"""Tests for the Qwen explanation layer.

Tests cover:
- QwenLLM wrapper configuration and lazy loading
- Evidence builder from pipeline results
- Prompt builder content and structure
- Integration flow (FAISS → XGBoost → Qwen)
- Graceful degradation when Qwen is unavailable
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.llm.evidence import build_evidence
from src.llm.prompts import build_matching_explanation_prompt
from src.llm.qwen import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_THREADS,
    QwenLLM,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _sample_candidate() -> dict:
    return {
        "candidate_name": "Nong Van Sam",
        "years_of_experience": 2,
        "technical_skills": ["Java", "Spring Boot", "MySQL", "REST API", "Git"],
        "programming_languages": ["Java"],
        "frameworks": ["Spring Boot", "Spring Security"],
        "databases": ["MySQL"],
        "projects": [
            {
                "name": "Airline Ticket Sales Platform",
                "description": "REST API backend with JWT authentication",
            },
        ],
        "education": [{"degree": "Bachelor of Computer Science"}],
    }


def _sample_job() -> dict:
    return {
        "job_title": "Backend Developer",
        "required_skills": ["Java", "Spring Boot", "MySQL", "REST API", "Git", "Kubernetes"],
        "preferred_skills": ["Docker", "Redis"],
        "years_of_experience": 3,
        "responsibilities": ["Design and implement REST APIs"],
        "education": ["Bachelor"],
    }


def _sample_match_result() -> dict:
    return {
        "candidate_name": "Nong Van Sam",
        "job_title": "Backend Developer",
        "semantic_score": 82.5,
        "required_skill_score": 83.33,
        "preferred_skill_score": 0.0,
        "experience_score": 66.67,
        "final_score": 0.91,
        "xgb_probability": 0.91,
        "faiss_similarity": 0.8432,
        "deterministic_score": 75.0,
        "matched_skills": ["Java", "Spring Boot", "MySQL", "REST API", "Git"],
        "missing_required_skills": ["Kubernetes"],
        "matched_preferred_skills": [],
        "recommendation": "Strong Match",
    }


# -----------------------------------------------------------------------
# QwenLLM wrapper tests
# -----------------------------------------------------------------------

class TestQwenLLMConfiguration(unittest.TestCase):
    """QwenLLM respects explicit, environment, and default configuration."""

    def test_defaults_when_no_arguments_or_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            llm = QwenLLM()
        self.assertEqual(llm.model_path, DEFAULT_MODEL_PATH)
        self.assertEqual(llm.n_ctx, DEFAULT_CONTEXT_SIZE)
        self.assertEqual(llm.n_threads, DEFAULT_THREADS)
        self.assertEqual(llm.temperature, DEFAULT_TEMPERATURE)
        self.assertEqual(llm.max_tokens, DEFAULT_MAX_TOKENS)

    def test_explicit_arguments_override_defaults(self):
        llm = QwenLLM(
            model_path="/custom/model.gguf",
            n_ctx=4096,
            n_threads=2,
            temperature=0.5,
            max_tokens=200,
        )
        self.assertEqual(llm.model_path, "/custom/model.gguf")
        self.assertEqual(llm.n_ctx, 4096)
        self.assertEqual(llm.n_threads, 2)
        self.assertEqual(llm.temperature, 0.5)
        self.assertEqual(llm.max_tokens, 200)

    def test_environment_variables_are_used_when_no_arguments(self):
        env = {
            "QWEN_MODEL_PATH": "/env/model.gguf",
            "QWEN_CONTEXT_SIZE": "2048",
            "QWEN_THREADS": "8",
            "QWEN_TEMPERATURE": "0.1",
            "QWEN_MAX_TOKENS": "300",
        }
        with patch.dict("os.environ", env, clear=True):
            llm = QwenLLM()
        self.assertEqual(llm.model_path, "/env/model.gguf")
        self.assertEqual(llm.n_ctx, 2048)
        self.assertEqual(llm.n_threads, 8)
        self.assertEqual(llm.temperature, 0.1)
        self.assertEqual(llm.max_tokens, 300)

    def test_explicit_arguments_override_environment(self):
        env = {"QWEN_MODEL_PATH": "/env/model.gguf", "QWEN_THREADS": "16"}
        with patch.dict("os.environ", env, clear=True):
            llm = QwenLLM(model_path="/explicit/model.gguf", n_threads=2)
        self.assertEqual(llm.model_path, "/explicit/model.gguf")
        self.assertEqual(llm.n_threads, 2)

    def test_model_is_not_loaded_on_init(self):
        llm = QwenLLM(model_path="nonexistent.gguf")
        self.assertFalse(llm.is_loaded)

    def test_missing_model_file_raises_clear_error(self):
        llm = QwenLLM(model_path="/nonexistent/path/model.gguf")
        with self.assertRaises(FileNotFoundError) as ctx:
            llm.generate("test prompt")
        self.assertIn("Qwen model not found", str(ctx.exception))

    def test_invalid_env_var_falls_back_to_default(self):
        env = {"QWEN_CONTEXT_SIZE": "not_a_number", "QWEN_TEMPERATURE": "bad"}
        with patch.dict("os.environ", env, clear=True):
            llm = QwenLLM()
        self.assertEqual(llm.n_ctx, DEFAULT_CONTEXT_SIZE)
        self.assertEqual(llm.temperature, DEFAULT_TEMPERATURE)


class TestQwenLLMGeneration(unittest.TestCase):
    """QwenLLM.generate returns a string from the mocked model."""

    def test_generate_returns_string(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "This is a mock explanation."}}],
        }

        llm = QwenLLM(model_path="mock.gguf")
        llm._llm = mock_llama  # Bypass lazy loading.

        result = llm.generate("Explain this candidate.")
        self.assertIsInstance(result, str)
        self.assertEqual(result, "This is a mock explanation.")

    def test_generate_strips_whitespace(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "  padded output  \n"}}],
        }

        llm = QwenLLM(model_path="mock.gguf")
        llm._llm = mock_llama

        self.assertEqual(llm.generate("prompt"), "padded output")

    def test_generate_handles_empty_content(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": ""}}],
        }

        llm = QwenLLM(model_path="mock.gguf")
        llm._llm = mock_llama

        self.assertEqual(llm.generate("prompt"), "")

    def test_generate_handles_none_content(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": None}}],
        }

        llm = QwenLLM(model_path="mock.gguf")
        llm._llm = mock_llama

        self.assertEqual(llm.generate("prompt"), "")

    def test_system_prompt_enforces_evidence_grounding(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }

        llm = QwenLLM(model_path="mock.gguf")
        llm._llm = mock_llama
        llm.generate("test")

        call_args = mock_llama.create_chat_completion.call_args
        messages = call_args[1].get("messages") or call_args[0][0] if call_args[0] else call_args[1]["messages"]
        system_msg = messages[0]["content"]
        self.assertIn("Only explain information provided in the evidence", system_msg)
        self.assertIn("Do not invent", system_msg)
        self.assertIn("Do not make a hiring decision", system_msg)

    def test_temperature_and_max_tokens_are_forwarded(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }

        llm = QwenLLM(model_path="mock.gguf", temperature=0.1, max_tokens=100)
        llm._llm = mock_llama
        llm.generate("test")

        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.1)
        self.assertEqual(call_kwargs["max_tokens"], 100)

    def test_model_instance_is_reused(self):
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }

        llm = QwenLLM(model_path="mock.gguf")
        llm._llm = mock_llama

        llm.generate("prompt 1")
        llm.generate("prompt 2")

        self.assertEqual(mock_llama.create_chat_completion.call_count, 2)
        # The model should still be the same instance.
        self.assertIs(llm._llm, mock_llama)


# -----------------------------------------------------------------------
# Evidence builder tests
# -----------------------------------------------------------------------

class TestBuildEvidence(unittest.TestCase):
    """build_evidence produces a compact, correctly structured evidence dict."""

    def test_evidence_has_required_structure(self):
        evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=_sample_match_result(),
            rank=1,
        )
        self.assertIn("candidate", evidence)
        self.assertIn("job", evidence)
        self.assertIn("model_result", evidence)

    def test_candidate_fields(self):
        evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=_sample_match_result(),
        )
        candidate = evidence["candidate"]
        self.assertEqual(candidate["name"], "Nong Van Sam")
        self.assertIn("Java", candidate["matched_required_skills"])
        self.assertIn("Kubernetes", candidate["missing_required_skills"])
        self.assertEqual(candidate["experience"], "2 years")
        self.assertIsInstance(candidate["projects"], list)
        self.assertIsInstance(candidate["education"], list)

    def test_job_fields(self):
        evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=_sample_match_result(),
        )
        job = evidence["job"]
        self.assertEqual(job["title"], "Backend Developer")
        self.assertIn("Java", job["required_skills"])
        self.assertEqual(job["experience_requirement"], "3 years")

    def test_model_result_fields(self):
        evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=_sample_match_result(),
            rank=1,
        )
        model = evidence["model_result"]
        self.assertEqual(model["xgboost_probability"], 0.91)
        self.assertEqual(model["rank"], 1)
        self.assertIn("faiss_similarity", model)

    def test_evidence_without_xgboost_uses_final_score(self):
        """When XGBoost is not used, evidence uses the deterministic score."""
        result = _sample_match_result()
        del result["xgb_probability"]
        result["final_score"] = 75.0
        evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=result,
        )
        self.assertEqual(evidence["model_result"]["xgboost_probability"], 0.75)

    def test_evidence_does_not_include_raw_cv_text(self):
        candidate = _sample_candidate()
        candidate["raw_text"] = "A very long raw CV text..." * 100
        evidence = build_evidence(
            candidate=candidate,
            job=_sample_job(),
            match_result=_sample_match_result(),
        )
        # The evidence should not contain the raw text anywhere.
        import json
        evidence_str = json.dumps(evidence)
        self.assertNotIn("A very long raw CV text", evidence_str)

    def test_evidence_with_empty_candidate(self):
        evidence = build_evidence(
            candidate={},
            job=_sample_job(),
            match_result={"final_score": 50.0},
        )
        self.assertEqual(evidence["candidate"]["name"], "Unknown Candidate")
        self.assertEqual(evidence["candidate"]["experience"], "Not specified")


# -----------------------------------------------------------------------
# Prompt builder tests
# -----------------------------------------------------------------------

class TestBuildMatchingExplanationPrompt(unittest.TestCase):
    """build_matching_explanation_prompt produces a well-structured prompt."""

    def setUp(self):
        self.evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=_sample_match_result(),
            rank=1,
        )
        self.prompt = build_matching_explanation_prompt(self.evidence)

    def test_prompt_contains_candidate_info(self):
        self.assertIn("Nong Van Sam", self.prompt)

    def test_prompt_contains_job_info(self):
        self.assertIn("Backend Developer", self.prompt)

    def test_prompt_contains_xgboost_probability(self):
        self.assertIn("0.91", self.prompt)

    def test_prompt_contains_matched_skills(self):
        for skill in ["Java", "Spring Boot", "MySQL", "REST API", "Git"]:
            self.assertIn(skill, self.prompt)

    def test_prompt_contains_missing_skills(self):
        self.assertIn("Kubernetes", self.prompt)

    def test_prompt_contains_experience(self):
        self.assertIn("2 years", self.prompt)

    def test_prompt_contains_projects(self):
        self.assertIn("Airline Ticket Sales Platform", self.prompt)

    def test_prompt_contains_education(self):
        self.assertIn("Bachelor", self.prompt)

    def test_prompt_contains_anti_hallucination_instructions(self):
        self.assertIn("Do not invent information", self.prompt)
        self.assertIn("Do not decide whether the candidate should be hired", self.prompt)
        self.assertIn("Do not change the ranking", self.prompt)
        self.assertIn("Use only the supplied evidence", self.prompt)

    def test_prompt_explains_xgboost_role(self):
        self.assertIn("XGBoost", self.prompt)
        self.assertIn("Your task is ONLY to explain the model result", self.prompt)

    def test_prompt_contains_explanation_instructions(self):
        self.assertIn("Why the candidate received this ranking", self.prompt)
        self.assertIn("Which job requirements are satisfied", self.prompt)
        self.assertIn("Which requirements are missing", self.prompt)
        self.assertIn("What areas the candidate could improve", self.prompt)

    def test_prompt_contains_rank(self):
        self.assertIn("#1", self.prompt)

    def test_prompt_contains_faiss_similarity(self):
        self.assertIn("FAISS similarity", self.prompt)

    def test_prompt_with_empty_evidence(self):
        """The builder should not crash even with minimal evidence."""
        prompt = build_matching_explanation_prompt({})
        self.assertIsInstance(prompt, str)
        self.assertIn("Explain:", prompt)


# -----------------------------------------------------------------------
# Integration tests (mocked LLM, real evidence + prompt pipeline)
# -----------------------------------------------------------------------

class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end evidence → prompt → explanation pipeline with mocked LLM."""

    def test_full_pipeline_produces_explanation(self):
        """Build evidence, generate a prompt, and get an explanation."""
        evidence = build_evidence(
            candidate=_sample_candidate(),
            job=_sample_job(),
            match_result=_sample_match_result(),
            rank=1,
        )
        prompt = build_matching_explanation_prompt(evidence)

        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": (
                "The candidate has a strong match for the Backend Developer "
                "role, with an XGBoost match probability of 0.91."
            )}}],
        }

        qwen = QwenLLM(model_path="mock.gguf")
        qwen._llm = mock_llama

        explanation = qwen.generate(prompt)
        self.assertIn("0.91", explanation)
        self.assertIn("Backend Developer", explanation)

    def test_qwen_failure_does_not_affect_ranking(self):
        """Ranking results should remain intact even when Qwen fails."""
        ranking_results = [
            {"candidate_name": "Alice", "xgb_probability": 0.95, "rank": 1},
            {"candidate_name": "Bob", "xgb_probability": 0.80, "rank": 2},
        ]

        # Simulate Qwen failure.
        qwen = QwenLLM(model_path="/nonexistent/model.gguf")

        explanation = None
        explanation_error = None
        try:
            evidence = build_evidence(
                candidate=_sample_candidate(),
                job=_sample_job(),
                match_result=ranking_results[0],
                rank=1,
            )
            prompt = build_matching_explanation_prompt(evidence)
            explanation = qwen.generate(prompt)
        except (FileNotFoundError, RuntimeError) as error:
            explanation_error = str(error)

        # Ranking is unaffected.
        self.assertEqual(ranking_results[0]["xgb_probability"], 0.95)
        self.assertEqual(ranking_results[1]["xgb_probability"], 0.80)
        self.assertIsNone(explanation)
        self.assertIsNotNone(explanation_error)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("faiss"),
        "faiss-cpu is not installed",
    )
    def test_removing_qwen_does_not_change_xgboost_ranking(self):
        """XGBoost ranking is independent of Qwen."""
        import numpy as np
        from src.matching.matcher import CVJobMatcher

        class FakeModel:
            def encode(self, texts, show_progress_bar=True):
                return np.array([[1.0, 0.0] if "Java" in text else [0.0, 1.0] for text in texts])

        class FakeXGBScorer:
            def predict_probabilities(self, feature_rows):
                return np.array([0.3, 0.9])

        import json
        from tempfile import TemporaryDirectory

        from src.vector_store import FAISSStore, MetadataStore

        with TemporaryDirectory() as directory:
            metadata_path = f"{directory}/metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump([
                    {"faiss_id": 0, "source_index": 0},
                    {"faiss_id": 1, "source_index": 1},
                ], handle)

            store = FAISSStore(2)
            store.add(np.array([[1.0, 0.0], [0.0, 1.0]]))

            cvs = [
                {"candidate_name": "Low Score", "technical_skills": ["Python"]},
                {"candidate_name": "High Score", "technical_skills": ["Java"]},
            ]
            job = {"job_title": "Java role", "required_skills": ["Java"]}

            matcher = CVJobMatcher(FakeModel())
            results = matcher.rank_retrieved_candidates(
                cvs, job, store, MetadataStore(metadata_path),
                k=2, xgb_scorer=FakeXGBScorer(),
            )

            # XGBoost ranking should be deterministic.
            self.assertEqual(results[0]["candidate_name"], "High Score")
            self.assertEqual(results[0]["xgb_probability"], 0.9)
            self.assertEqual(results[1]["candidate_name"], "Low Score")
            self.assertEqual(results[1]["xgb_probability"], 0.3)

            # Now use the evidence + prompt pipeline (Qwen mocked) —
            # the ranking must not change.
            evidence = build_evidence(
                candidate=cvs[1],
                job=job,
                match_result=results[0],
                rank=1,
            )
            prompt = build_matching_explanation_prompt(evidence)

            # Verify that prompts contain the correct ranking info.
            self.assertIn("#1", prompt)
            self.assertIn("0.9", prompt)

            # Ranking still the same.
            self.assertEqual(results[0]["xgb_probability"], 0.9)
            self.assertEqual(results[1]["xgb_probability"], 0.3)


if __name__ == "__main__":
    unittest.main()
