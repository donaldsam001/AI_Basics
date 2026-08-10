"""Similarity and deterministic score components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


SKILL_ALIASES = {
    "springboot": "spring boot", "postgres": "postgresql", "k8s": "kubernetes",
    "gitlab ci": "gitlab ci/cd", "gitlab cicd": "gitlab ci/cd",
}


def normalize_skill(skill: Any) -> str:
    value = " ".join(str(skill or "").strip().casefold().split())
    return SKILL_ALIASES.get(value, value)


def calculate_skill_match(candidate_skills: Iterable[Any], target_skills: Iterable[Any]) -> dict[str, Any]:
    """Return an explainable exact canonical skill match percentage."""
    candidate = {normalize_skill(item) for item in candidate_skills if normalize_skill(item)}
    required: dict[str, str] = {}
    for item in target_skills:
        original = str(item).strip()
        normalized = normalize_skill(original)
        if normalized and normalized not in required:
            required[normalized] = original
    matched_keys = [key for key in required if key in candidate]
    matched = [required[key] for key in matched_keys]
    missing = [required[key] for key in required if key not in candidate]
    score = 100.0 * len(matched) / len(required) if required else 100.0
    return {"score": round(score, 2), "matched": matched, "missing": missing}


def calculate_required_skill_score(candidate_skills: Iterable[Any], required_skills: Iterable[Any]) -> float:
    return float(calculate_skill_match(candidate_skills, required_skills)["score"])


def calculate_experience_score(candidate_years: Any, required_years: Any) -> float:
    """Score experience proportionally; an unspecified requirement is neutral."""
    try:
        required = float(required_years or 0)
    except (TypeError, ValueError):
        required = 0.0
    try:
        candidate = float(candidate_years or 0)
    except (TypeError, ValueError):
        candidate = 0.0
    if required <= 0:
        return 100.0
    return round(max(0.0, min(100.0, candidate / required * 100.0)), 2)


def cosine_similarity(cv_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
    """Cosine similarity, optimized for normalized MPNet embeddings."""
    cv = np.asarray(cv_embedding, dtype=float)
    job = np.asarray(job_embedding, dtype=float)
    if cv.size == 0 or job.size == 0 or cv.shape != job.shape:
        raise ValueError("Embeddings must be non-empty vectors with the same shape.")
    denominator = float(np.linalg.norm(cv) * np.linalg.norm(job))
    return float(np.dot(cv, job) / denominator) if denominator else 0.0


def similarity_to_score(similarity: float) -> float:
    """Map cosine [-1, 1] to an interpretable semantic match score [0, 100].

    This is a relevance scale for ranking, not a hiring probability.
    """
    return round(max(0.0, min(100.0, (float(similarity) + 1.0) * 50.0)), 2)
