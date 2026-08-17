"""Backward-compatible feature imports.

New code should import from :mod:`src.features` directly.
"""

from src.features.feature_engineering import build_cv_jd_features


def build_features(result: dict) -> list[float]:
    """Retain the historical compact matching feature vector."""
    return [
        float(result.get("semantic_score", 0.0)) / 100.0,
        float(result.get("required_skill_score", 0.0)) / 100.0,
        float(result.get("preferred_skill_score", 0.0)) / 100.0,
        float(result.get("experience_score", 0.0)) / 100.0,
        float(result.get("faiss_similarity", result.get("retrieval_score", 0.0))),
    ]
