"""Reusable, numeric-only features for CV suitability and CV--JD matching.

The bundled training data labels a CV in isolation.  Consequently
``build_candidate_features`` is the feature set used by the default training
command.  ``build_cv_jd_features`` is available for a future dataset with
pair-level CV--JD labels and is also used to expose explanation fields during
reranking.  Keeping those paths separate prevents inventing pair labels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from src.matching.similarity import calculate_experience_score, calculate_skill_match, normalize_skill


CANDIDATE_FEATURE_NAMES = [
    "years_experience",
    "has_portfolio",
    "skill_count",
    "raw_text_length",
    "education_level",
]

_EDUCATION_LEVELS = {
    "high school": 0,
    "secondary school": 0,
    "associate": 1,
    "associate degree": 1,
    "bachelor": 2,
    "bachelors": 2,
    "bachelor's": 2,
    "undergraduate": 2,
    "master": 3,
    "masters": 3,
    "master's": 3,
    "mba": 3,
    "phd": 4,
    "ph d": 4,
    "ph.d": 4,
    "doctorate": 4,
    "doctoral": 4,
}
_TITLE_WORD_RE = re.compile(r"[a-z0-9+#.]+")


def _as_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value or "")


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number >= 0 else 0.0


def _boolean(value: Any) -> float:
    if isinstance(value, str):
        return float(value.strip().casefold() in {"1", "true", "yes", "y"})
    return float(bool(value))


def parse_skills(value: Any) -> list[str]:
    """Parse comma/pipe separated skills into unique canonical names."""
    if value is None:
        return []
    values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = re.split(r"[,|;\n]", str(value))
    seen: set[str] = set()
    result = []
    for item in values:
        normalized = normalize_skill(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def education_level(value: Any) -> float:
    """Map known degrees to their meaningful academic order; unknown is -1."""
    text = " ".join(_as_text(value).casefold().replace(".", " ").split())
    if not text:
        return -1.0
    for degree, level in _EDUCATION_LEVELS.items():
        if degree.replace(".", "") in text:
            return float(level)
    return -1.0


def candidate_skills(candidate: dict[str, Any]) -> list[str]:
    fields = (
        "skills", "technical_skills", "programming_languages", "frameworks", "libraries",
        "databases", "cloud_platforms", "devops_tools", "testing_tools", "keywords",
    )
    return parse_skills([item for field in fields for item in parse_skills(candidate.get(field))])


def _candidate_years(candidate: dict[str, Any]) -> float:
    return _number(candidate.get("years_experience", candidate.get("years_of_experience")))


def _candidate_title(candidate: dict[str, Any]) -> str:
    return _as_text(candidate.get("current_title") or candidate.get("current_position"))


def _candidate_degree(candidate: dict[str, Any]) -> Any:
    return candidate.get("highest_degree", candidate.get("education"))


def build_candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Build the CV-level features supported by ``ml_resume_dataset_4500.csv``."""
    raw_text = _as_text(candidate.get("raw_text", candidate.get("cleaned_text", "")))
    return {
        "years_experience": _candidate_years(candidate),
        "has_portfolio": _boolean(candidate.get("has_portfolio", candidate.get("portfolio"))),
        "skill_count": float(len(candidate_skills(candidate))),
        "raw_text_length": float(len(raw_text)),
        "education_level": education_level(_candidate_degree(candidate)),
    }


def _title_similarity(candidate_title: str, job_title: str) -> float:
    candidate_words = set(_TITLE_WORD_RE.findall(candidate_title.casefold()))
    job_words = set(_TITLE_WORD_RE.findall(job_title.casefold()))
    if not candidate_words or not job_words:
        return 0.0
    return len(candidate_words & job_words) / len(candidate_words | job_words)


def _education_score(candidate_degree: Any, required_education: Any) -> float:
    required = education_level(required_education)
    candidate = education_level(candidate_degree)
    if required < 0:
        return 1.0
    if candidate < 0:
        return 0.0
    return min(candidate / max(required, 1.0), 1.0)


def build_cv_jd_features(
    candidate: dict[str, Any], job: dict[str, Any], *, semantic_score: float = 0.0,
    retrieval_score: float | None = None,
) -> dict[str, float]:
    """Build numeric comparison features for one candidate and one job.

    ``semantic_score`` is expected on the existing 0--100 matching scale; it
    is stored as 0--1 for model-friendly consistency.  FAISS similarity is
    kept separately because it is a retrieval signal, not a probability.
    """
    candidate_features = build_candidate_features(candidate)
    skills = candidate_skills(candidate)
    required = parse_skills(job.get("required_skills"))
    preferred = parse_skills(job.get("preferred_skills"))
    required_match = calculate_skill_match(skills, required)
    preferred_match = calculate_skill_match(skills, preferred)
    matched_required = len(required_match["matched"])
    years = _candidate_years(candidate)
    required_years = _number(job.get("years_of_experience"))
    portfolio_required = "portfolio" in _as_text(job.get("summary")).casefold()
    semantic = max(0.0, min(1.0, _number(semantic_score) / 100.0))
    features = {
        **candidate_features,
        "semantic_score": semantic,
        "required_skill_score": required_match["score"] / 100.0,
        "preferred_skill_score": preferred_match["score"] / 100.0,
        "skill_overlap": float(matched_required),
        "skill_match_ratio": matched_required / len(required) if required else 1.0,
        "matched_skill_count": float(matched_required),
        "missing_skill_count": float(len(required_match["missing"])),
        "experience_score": calculate_experience_score(years, required_years) / 100.0,
        "experience_gap": years - required_years,
        "education_score": _education_score(_candidate_degree(candidate), job.get("education")),
        "title_similarity": _title_similarity(_candidate_title(candidate), _as_text(job.get("job_title"))),
        "title_match": float(_candidate_title(candidate).casefold() == _as_text(job.get("job_title")).casefold()
                             and bool(_candidate_title(candidate))),
        "portfolio_match": candidate_features["has_portfolio"] if portfolio_required else 1.0,
    }
    if retrieval_score is not None:
        features["faiss_similarity"] = float(retrieval_score)
    return {name: float(value) for name, value in features.items()}
