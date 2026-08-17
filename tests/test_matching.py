"""Unit tests for text, scoring, embeddings, and batched candidate ranking."""

from __future__ import annotations

import unittest
import json
from tempfile import TemporaryDirectory

import numpy as np

from src.matching.matcher import CVJobMatcher, calculate_final_score
from src.embeddings import EmbeddingCache
from src.matching.similarity import (
    calculate_experience_score, calculate_required_skill_score, cosine_similarity,
    similarity_to_score,
)
from src.preprocessing import build_cv_embedding_text, build_job_embedding_text, preprocess_job
from src.vector_store import FAISSStore, MetadataStore


class FakeModel:
    def encode(self, texts, show_progress_bar=True):
        return np.array([[1.0, 0.0] if "Java" in text else [0.0, 1.0] for text in texts])


class MatchingTests(unittest.TestCase):
    def test_cv_text_excludes_personal_data(self):
        text = build_cv_embedding_text({"candidate_name": "Ada", "current_position": "Engineer", "technical_skills": ["Python"]})
        self.assertIn("Engineer", text)
        self.assertNotIn("Ada", text)

    def test_job_preprocessing_and_text(self):
        job = preprocess_job("Requires 3 years of Java", "Backend Developer")
        job["required_skills"] = ["Java"]
        self.assertEqual(job["years_of_experience"], 3.0)
        self.assertIn("Required Skills", build_job_embedding_text(job))

    def test_similarity_and_score(self):
        self.assertEqual(cosine_similarity(np.array([1, 0]), np.array([1, 0])), 1.0)
        self.assertEqual(similarity_to_score(-1), 0.0)
        self.assertEqual(similarity_to_score(1), 100.0)

    def test_skill_aliases_duplicates_and_empty_requirement(self):
        self.assertEqual(calculate_required_skill_score(["K8s", "Postgres"], ["kubernetes", "PostgreSQL", "Kubernetes"]), 100.0)
        self.assertEqual(calculate_required_skill_score([], []), 100.0)

    def test_experience_edge_cases(self):
        self.assertEqual(calculate_experience_score(3, 2), 100.0)
        self.assertEqual(calculate_experience_score(1, 4), 25.0)
        self.assertEqual(calculate_experience_score("unknown", 3), 0.0)

    def test_final_score(self):
        result = calculate_final_score(80, 75, 50, 100)
        self.assertEqual(result["final_score"], 77.5)

    def test_candidate_ranking_batches_cvs(self):
        matcher = CVJobMatcher(FakeModel())
        job = {"job_title": "Java role", "required_skills": ["Java"]}
        cvs = [
            {"candidate_name": "Java Dev", "technical_skills": ["Java"]},
            {"candidate_name": "Python Dev", "technical_skills": ["Python"]},
        ]
        ranked = matcher.rank_candidates(cvs, job)
        self.assertEqual(ranked[0]["candidate_name"], "Java Dev")
        self.assertEqual(ranked[0]["required_skill_score"], 100.0)

    def test_embedding_cache_reuses_unchanged_text(self):
        class CountingModel(FakeModel):
            def __init__(self):
                self.calls = 0

            def encode(self, texts, show_progress_bar=True):
                self.calls += 1
                return super().encode(texts, show_progress_bar)

        with TemporaryDirectory() as directory:
            model = CountingModel()
            cache = EmbeddingCache(directory)
            cache.encode(["Java"], model, "cv")
            cache.encode(["Java"], model, "cv")
            self.assertEqual(model.calls, 1)

    def test_faiss_candidates_are_reranked_by_xgboost_probability(self):
        class FakeScorer:
            def predict_probabilities(self, feature_rows):
                self.feature_rows = feature_rows
                return np.array([0.1, 0.9])

        with TemporaryDirectory() as directory:
            metadata_path = f"{directory}/metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump([
                    {"faiss_id": 0, "source_index": 0},
                    {"faiss_id": 1, "source_index": 1},
                ], handle)
            store = FAISSStore(2)
            store.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
            matcher = CVJobMatcher(FakeModel())
            results = matcher.rank_retrieved_candidates(
                [
                    {"candidate_name": "Java Dev", "technical_skills": ["Java"]},
                    {"candidate_name": "Python Dev", "technical_skills": ["Python"]},
                ],
                {"job_title": "Java role", "required_skills": ["Java"]},
                store, MetadataStore(metadata_path), k=2, xgb_scorer=FakeScorer(),
            )
            self.assertEqual(results[0]["candidate_name"], "Python Dev")
            self.assertEqual(results[0]["xgb_probability"], 0.9)
            self.assertIn("faiss_similarity", results[0])


if __name__ == "__main__":
    unittest.main()
