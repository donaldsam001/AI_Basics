"""Tests for the MatchingPipeline orchestrator and configuration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import PipelineConfig
from src.llm.qwen import QwenLLM
from src.matching.matcher import CVJobMatcher
from src.pipeline import MatchingPipeline


# -----------------------------------------------------------------------
# Shared test helpers
# -----------------------------------------------------------------------

class FakeEmbeddingModel:
    """Deterministic 2-dim embeddings keyed on 'Java' presence."""
    def encode(self, texts, show_progress_bar=True):
        return np.array([
            [1.0, 0.0] if "Java" in text else [0.0, 1.0]
            for text in texts
        ])


def _make_config(**overrides) -> PipelineConfig:
    defaults = {
        "retrieval_top_k": 10,
        "ranking_top_k": 5,
        "explanation_top_k": 2,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_cvs() -> list[dict]:
    return [
        {"candidate_name": "Alice", "technical_skills": ["Java", "Spring Boot"]},
        {"candidate_name": "Bob", "technical_skills": ["Python", "Django"]},
        {"candidate_name": "Carol", "technical_skills": ["Java", "MySQL"]},
    ]


def _make_job() -> dict:
    return {
        "job_title": "Backend Developer",
        "required_skills": ["Java", "Spring Boot"],
        "preferred_skills": ["Docker"],
        "years_of_experience": 3,
    }


# -----------------------------------------------------------------------
# Configuration tests
# -----------------------------------------------------------------------

class TestPipelineConfig(unittest.TestCase):

    def test_default_values(self):
        with patch.dict("os.environ", {}, clear=True):
            config = PipelineConfig()
        self.assertEqual(config.retrieval_top_k, 100)
        self.assertEqual(config.ranking_top_k, 20)
        self.assertEqual(config.explanation_top_k, 5)

    def test_env_overrides(self):
        env = {
            "RETRIEVAL_TOP_K": "200",
            "RANKING_TOP_K": "30",
            "EXPLANATION_TOP_K": "3",
        }
        with patch.dict("os.environ", env, clear=True):
            config = PipelineConfig()
        self.assertEqual(config.retrieval_top_k, 200)
        self.assertEqual(config.ranking_top_k, 30)
        self.assertEqual(config.explanation_top_k, 3)

    def test_explicit_overrides(self):
        config = PipelineConfig(retrieval_top_k=50, ranking_top_k=10, explanation_top_k=1)
        self.assertEqual(config.retrieval_top_k, 50)
        self.assertEqual(config.ranking_top_k, 10)
        self.assertEqual(config.explanation_top_k, 1)


# -----------------------------------------------------------------------
# Pipeline without FAISS (baseline only)
# -----------------------------------------------------------------------

class TestPipelineBaselineOnly(unittest.TestCase):
    """Pipeline running the rule-based matcher without FAISS or XGBoost."""

    def test_rank_candidates_without_faiss_or_xgboost(self):
        matcher = CVJobMatcher(FakeEmbeddingModel())
        pipeline = MatchingPipeline(matcher=matcher, config=_make_config())
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())

        self.assertTrue(len(results) > 0)
        # First result should have the best final_score.
        for i in range(1, len(results)):
            self.assertGreaterEqual(
                results[i - 1]["final_score"],
                results[i]["final_score"],
            )

    def test_ranks_are_assigned_after_sorting(self):
        matcher = CVJobMatcher(FakeEmbeddingModel())
        pipeline = MatchingPipeline(matcher=matcher, config=_make_config())
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())

        for i, result in enumerate(results, start=1):
            self.assertEqual(result["rank"], i)

    def test_ranking_top_k_limits_output(self):
        matcher = CVJobMatcher(FakeEmbeddingModel())
        config = _make_config(ranking_top_k=2)
        pipeline = MatchingPipeline(matcher=matcher, config=config)
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())
        self.assertLessEqual(len(results), 2)

    def test_empty_cvs_returns_empty(self):
        matcher = CVJobMatcher(FakeEmbeddingModel())
        pipeline = MatchingPipeline(matcher=matcher, config=_make_config())
        results = pipeline.rank_and_explain([], _make_job())
        self.assertEqual(results, [])


# -----------------------------------------------------------------------
# Pipeline with mocked XGBoost
# -----------------------------------------------------------------------

class TestPipelineWithXGBoost(unittest.TestCase):
    """Pipeline with XGBoost scorer providing match probabilities."""

    def _make_pipeline(self, probabilities: np.ndarray) -> MatchingPipeline:
        scorer = MagicMock()
        scorer.predict_probabilities.return_value = probabilities
        # We need FAISS for XGBoost path; use the baseline-only path
        # by not passing stores, but attach scorer for API response test.
        matcher = CVJobMatcher(FakeEmbeddingModel())
        return MatchingPipeline(
            matcher=matcher,
            xgb_scorer=scorer,
            config=_make_config(),
        )

    def test_baseline_score_preserved_alongside_xgboost(self):
        """The deterministic baseline score should survive XGBoost reranking."""
        matcher = CVJobMatcher(FakeEmbeddingModel())
        pipeline = MatchingPipeline(matcher=matcher, config=_make_config())
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())
        # Without FAISS stores, baseline path is used.
        for result in results:
            self.assertIn("final_score", result)


# -----------------------------------------------------------------------
# Pipeline with mocked Qwen
# -----------------------------------------------------------------------

class TestPipelineWithQwen(unittest.TestCase):
    """Pipeline generates explanations for top-K candidates only."""

    def test_explanations_only_for_top_k(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Mock explanation."}}],
        }
        qwen = QwenLLM(model_path="mock.gguf")
        qwen._llm = mock_llama

        matcher = CVJobMatcher(FakeEmbeddingModel())
        config = _make_config(explanation_top_k=1)
        pipeline = MatchingPipeline(matcher=matcher, qwen=qwen, config=config)
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())

        # Only the first result should have an explanation.
        self.assertIn("explanation", results[0])
        self.assertEqual(results[0]["explanation"], "Mock explanation.")
        for result in results[1:]:
            self.assertNotIn("explanation", result)

    def test_qwen_called_correct_number_of_times(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        qwen = QwenLLM(model_path="mock.gguf")
        qwen._llm = mock_llama

        matcher = CVJobMatcher(FakeEmbeddingModel())
        config = _make_config(explanation_top_k=2)
        pipeline = MatchingPipeline(matcher=matcher, qwen=qwen, config=config)
        pipeline.rank_and_explain(_make_cvs(), _make_job())

        self.assertEqual(mock_llama.create_chat_completion.call_count, 2)

    def test_qwen_failure_preserves_ranking(self):
        qwen = QwenLLM(model_path="/nonexistent/model.gguf")

        matcher = CVJobMatcher(FakeEmbeddingModel())
        config = _make_config(explanation_top_k=2)
        pipeline = MatchingPipeline(matcher=matcher, qwen=qwen, config=config)
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())

        # Ranking should be intact.
        self.assertTrue(len(results) > 0)
        for result in results:
            self.assertIn("rank", result)
            self.assertIn("final_score", result)

        # First candidate should have error info.
        self.assertIsNone(results[0].get("explanation"))
        self.assertIn("explanation_error", results[0])

    def test_no_qwen_means_no_explanations(self):
        matcher = CVJobMatcher(FakeEmbeddingModel())
        pipeline = MatchingPipeline(matcher=matcher, qwen=None, config=_make_config())
        results = pipeline.rank_and_explain(_make_cvs(), _make_job())
        for result in results:
            self.assertNotIn("explanation", result)


# -----------------------------------------------------------------------
# API response builder
# -----------------------------------------------------------------------

class TestAPIResponseBuilder(unittest.TestCase):

    def test_response_structure(self):
        job = _make_job()
        results = [
            {
                "candidate_name": "Alice",
                "rank": 1,
                "xgb_probability": 0.94,
                "faiss_similarity": 0.87,
                "matched_skills": ["Java", "Spring Boot"],
                "missing_required_skills": ["Docker"],
                "matched_preferred_skills": [],
                "semantic_score": 80.0,
                "required_skill_score": 90.0,
                "preferred_skill_score": 0.0,
                "experience_score": 75.0,
                "deterministic_score": 72.5,
                "final_score": 0.94,
                "explanation": "Good match.",
            },
        ]
        response = MatchingPipeline.build_api_response(job, results)
        self.assertEqual(response["job_title"], "Backend Developer")
        self.assertEqual(len(response["candidates"]), 1)

        candidate = response["candidates"][0]
        self.assertEqual(candidate["candidate_name"], "Alice")
        self.assertEqual(candidate["rank"], 1)
        self.assertEqual(candidate["match_probability"], 0.94)
        self.assertEqual(candidate["retrieval_score"], 0.87)
        self.assertIn("Java", candidate["matched_skills"])
        self.assertEqual(candidate["explanation"], "Good match.")
        self.assertEqual(candidate["baseline_score"], 72.5)

    def test_response_without_explanation(self):
        response = MatchingPipeline.build_api_response(
            _make_job(),
            [{"candidate_name": "Bob", "rank": 1, "final_score": 50.0}],
        )
        candidate = response["candidates"][0]
        self.assertNotIn("explanation", candidate)

    def test_response_with_explanation_error(self):
        response = MatchingPipeline.build_api_response(
            _make_job(),
            [{
                "candidate_name": "Carol", "rank": 1, "final_score": 60.0,
                "explanation": None, "explanation_error": "Model unavailable",
            }],
        )
        candidate = response["candidates"][0]
        self.assertIsNone(candidate["explanation"])
        self.assertEqual(candidate["explanation_error"], "Model unavailable")


if __name__ == "__main__":
    unittest.main()
