"""Hybrid, explainable CV-job matching."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.embeddings import EmbeddingCache, EmbeddingModel
from src.preprocessing import build_cv_embedding_text, build_job_embedding_text

from .similarity import (
    calculate_experience_score, calculate_skill_match, similarity_to_score,
)


MATCHING_WEIGHTS = {
    "semantic": 0.50, "required_skills": 0.30,
    "preferred_skills": 0.10, "experience": 0.10,
}
RECOMMENDATION_THRESHOLDS = (
    (90.0, "Excellent Match"), (75.0, "Strong Match"),
    (60.0, "Moderate Match"), (40.0, "Weak Match"), (0.0, "Poor Match"),
)


def _candidate_skills(cv: dict[str, Any]) -> list[Any]:
    fields = (
        "technical_skills", "programming_languages", "frameworks", "libraries",
        "databases", "cloud_platforms", "devops_tools", "testing_tools", "keywords",
    )
    return [skill for field in fields for skill in cv.get(field, [])]


def _job_skills(job: dict[str, Any], field: str) -> list[Any]:
    return list(job.get(field, []))


def calculate_final_score(
    semantic_score: float, required_skill_score: float, preferred_skill_score: float,
    experience_score: float, weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute a deterministic weighted ranking score from 0 through 100."""
    active_weights = weights or MATCHING_WEIGHTS
    values = {
        "semantic": semantic_score, "required_skills": required_skill_score,
        "preferred_skills": preferred_skill_score, "experience": experience_score,
    }
    missing = set(values) - set(active_weights)
    if missing:
        raise ValueError(f"Missing matching weights: {sorted(missing)}")
    final = sum(float(values[key]) * float(active_weights[key]) for key in values)
    return {
        "semantic_score": round(float(semantic_score), 2),
        "required_skill_score": round(float(required_skill_score), 2),
        "preferred_skill_score": round(float(preferred_skill_score), 2),
        "experience_score": round(float(experience_score), 2),
        "final_score": round(max(0.0, min(100.0, final)), 2),
    }


def recommendation(score: float, thresholds: Iterable[tuple[float, str]] = RECOMMENDATION_THRESHOLDS) -> str:
    for minimum, label in thresholds:
        if score >= minimum:
            return label
    return "Poor Match"


class CVJobMatcher:
    """Reuse a single model while scoring one job against one or many CVs."""

    def __init__(self, embedding_model: EmbeddingModel | None = None,
                 weights: dict[str, float] | None = None,
                 embedding_cache: EmbeddingCache | None = None) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.weights = weights or MATCHING_WEIGHTS.copy()
        self.embedding_cache = embedding_cache

    def _encode(self, texts: list[str], namespace: str) -> np.ndarray:
        if self.embedding_cache:
            return self.embedding_cache.encode(texts, self.embedding_model, namespace)
        return self.embedding_model.encode(texts)

    def encode_cv(self, cv: dict[str, Any]) -> np.ndarray:
        return self._encode([build_cv_embedding_text(cv)], "cv")[0]

    def encode_job(self, job: dict[str, Any]) -> np.ndarray:
        return self._encode([build_job_embedding_text(job)], "job")[0]

    def match(self, cv: dict[str, Any], job: dict[str, Any],
              cv_embedding: np.ndarray | None = None,
              job_embedding: np.ndarray | None = None,
              faiss_similarity: float | None = None) -> dict[str, Any]:
        cv_embedding = self.encode_cv(cv) if cv_embedding is None else cv_embedding
        job_embedding = self.encode_job(job) if job_embedding is None else job_embedding
        semantic = similarity_to_score(float(np.dot(cv_embedding, job_embedding)))
        skills = _candidate_skills(cv)
        required = calculate_skill_match(skills, _job_skills(job, "required_skills"))
        preferred = calculate_skill_match(skills, _job_skills(job, "preferred_skills"))
        experience = calculate_experience_score(
            cv.get("years_of_experience"), job.get("years_of_experience"),
        )
        scores = calculate_final_score(semantic, required["score"], preferred["score"], experience, self.weights)
        result = {
            "candidate_name": cv.get("candidate_name") or "Unknown Candidate",
            "job_title": job.get("job_title") or "Untitled Job", **scores,
            "matched_skills": required["matched"],
            "missing_required_skills": required["missing"],
            "matched_preferred_skills": preferred["matched"],
            "recommendation": recommendation(scores["final_score"]),
        }
        if faiss_similarity is not None:
            # This is cosine similarity returned by FAISS, never a probability.
            result["faiss_similarity"] = round(float(faiss_similarity), 6)
        return result

    def rank_candidates(self, cvs: list[dict[str, Any]], job: dict[str, Any]) -> list[dict[str, Any]]:
        """Embed all CVs in one batch and the job once, then rank descending."""
        if not cvs:
            return []
        cv_texts = [build_cv_embedding_text(cv) for cv in cvs]
        cv_embeddings = self._encode(cv_texts, "cv")
        job_embedding = self.encode_job(job)
        results = [self.match(cv, job, embedding, job_embedding)
                   for cv, embedding in zip(cvs, cv_embeddings, strict=True)]
        return sorted(results, key=lambda result: result["final_score"], reverse=True)

    def rank_retrieved_candidates(
        self, cvs: list[dict[str, Any]], job: dict[str, Any], vector_store: Any,
        metadata_store: Any, k: int = 50, xgb_scorer: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve with FAISS, then optionally rerank only those candidates."""
        if not cvs:
            return []
        metadata_store.validate_index_size(vector_store.size)
        job_embedding = self.encode_job(job)
        faiss_scores, faiss_ids = vector_store.search(job_embedding, k)
        source_indices = [metadata_store.get_candidate(int(faiss_id))["source_index"] for faiss_id in faiss_ids]
        if any(not isinstance(index, int) or not 0 <= index < len(cvs) for index in source_indices):
            raise ValueError("FAISS metadata contains a source_index outside the loaded CV dataset")
        retrieved_cvs = [cvs[index] for index in source_indices]
        cv_embeddings = self._encode([build_cv_embedding_text(cv) for cv in retrieved_cvs], "cv")
        results = [
            self.match(cv, job, embedding, job_embedding, float(faiss_score))
            for cv, embedding, faiss_score in zip(retrieved_cvs, cv_embeddings, faiss_scores, strict=True)
        ]
        if xgb_scorer is None:
            return sorted(results, key=lambda result: result["final_score"], reverse=True)

        # Import here to avoid a package-import cycle with matching helpers.
        from src.features import build_cv_jd_features

        feature_rows = [
            build_cv_jd_features(cv, job, semantic_score=result["semantic_score"],
                                  retrieval_score=result["faiss_similarity"])
            for cv, result in zip(retrieved_cvs, results, strict=True)
        ]
        probabilities = xgb_scorer.predict_probabilities(feature_rows)
        for result, features, probability in zip(results, feature_rows, probabilities, strict=True):
            result["deterministic_score"] = result["final_score"]
            result["xgb_probability"] = round(float(probability), 6)
            result["final_score"] = round(float(probability), 6)
            result["education_score"] = round(features["education_score"] * 100.0, 2)
        return sorted(results, key=lambda result: result["xgb_probability"], reverse=True)


def rank_candidates(cvs: list[dict[str, Any]], job_description: dict[str, Any],
                    embedding_model: EmbeddingModel | None = None) -> list[dict[str, Any]]:
    return CVJobMatcher(embedding_model).rank_candidates(cvs, job_description)
