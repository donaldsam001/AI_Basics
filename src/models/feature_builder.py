"""Canonical pair-level feature builder — the single source of truth.

This module defines ONE feature pipeline used identically during:
  - XGBoost training
  - XGBoost inference (FAISS → reranking)
  - offline evaluation
  - end-to-end tests

Design goals
------------
* ``build_pair_features`` accepts CV dict, job dict, and retrieval scores.
* The returned dict always contains exactly ``PAIR_FEATURE_NAMES`` keys.
* The ordering of ``PAIR_FEATURE_NAMES`` is deterministic and authoritative.
* ``validate_feature_row`` raises ``ValueError`` when any expected feature
  is absent — never silently fills with 0.
* All features are plain Python floats (no numpy scalars).

Label provenance note
---------------------
The ``shortlisted`` column in ``ats_resume_dataset_elite_v3.csv`` is
**algorithmically derived**: ``shortlisted = (final_score >= 0.4)`` where
``final_score`` is a linear composite of ``skill_match_score``,
``experience_match``, and ``education_match``.  It is **NOT** a genuine
human recruiter outcome.  Any trained model learns to reproduce the source
scoring rule, not real hiring decisions.  All documentation and outputs
must reflect this limitation.
"""

from __future__ import annotations

import re
from typing import Any

from src.matching.similarity import (
    calculate_experience_score,
    calculate_skill_match,
    normalize_skill,
)


# ---------------------------------------------------------------------------
# Canonical ordered feature list — training & inference must match exactly.
# ---------------------------------------------------------------------------

PAIR_FEATURE_NAMES: list[str] = [
    # ── Semantic / retrieval ────────────────────────────────────────────────
    "semantic_similarity",          # cosine similarity (0–1 scale)
    "faiss_similarity",             # FAISS retrieval score (0–1)
    # ── Required skill features ─────────────────────────────────────────────
    "required_skill_score",         # fraction of required skills matched (0–1)
    "matched_required_skill_count", # absolute count of matched required skills
    "missing_required_skill_count", # absolute count of missing required skills
    "total_required_skill_count",   # total required skills in job
    # ── Preferred skill features ────────────────────────────────────────────
    "preferred_skill_score",        # fraction of preferred skills matched (0–1)
    "matched_preferred_skill_count",# absolute count matched preferred
    # ── Combined skill coverage ─────────────────────────────────────────────
    "candidate_skill_count",        # total distinct skills on CV
    "skill_coverage_ratio",         # matched_required / (required + 1)
    # ── Experience features ─────────────────────────────────────────────────
    "candidate_experience_years",   # years of experience from CV
    "required_experience_years",    # years required by job
    "experience_gap",               # candidate - required (signed)
    "experience_satisfies",         # 1.0 if candidate >= required else 0.0
    "experience_score",             # proportional 0–1 score
    # ── Education features ───────────────────────────────────────────────────
    "education_level_score",        # ordinal level of candidate degree (0–4; -1=unknown)
    "education_meets_requirement",  # 1.0 if candidate level >= required level
    # ── CV content signals ───────────────────────────────────────────────────
    "has_certification",            # 1 if certifications field is non-empty
    "certification_count",          # count of certifications
    "has_portfolio",                # 1 if portfolio indicated
    "resume_text_length",           # char length of resume text
    # ── Title relevance ──────────────────────────────────────────────────────
    "title_similarity",             # word-overlap Jaccard between titles
]

# Validate no duplicates at import time.
assert len(PAIR_FEATURE_NAMES) == len(set(PAIR_FEATURE_NAMES)), \
    "PAIR_FEATURE_NAMES contains duplicates"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EDUCATION_LEVELS: dict[str, int] = {
    "high school": 0, "secondary school": 0,
    "associate": 1, "associate degree": 1,
    "bachelor": 2, "bachelors": 2, "bachelor's": 2, "undergraduate": 2,
    "master": 3, "masters": 3, "master's": 3, "mba": 3,
    "phd": 4, "ph d": 4, "ph.d": 4, "doctorate": 4, "doctoral": 4,
}
_WORD_RE = re.compile(r"[a-z0-9+#.]+")


def _as_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value or "")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result >= 0 else default
    except (TypeError, ValueError):
        return default


def _as_signed_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _education_level(value: Any) -> float:
    """Map known degree names to ordinal integer; unknown → -1.0."""
    text = " ".join(_as_text(value).casefold().replace(".", " ").split())
    if not text:
        return -1.0
    for degree, level in _EDUCATION_LEVELS.items():
        if degree.replace(".", "") in text:
            return float(level)
    return -1.0


def _parse_skills(value: Any) -> list[str]:
    """Return a deduplicated, normalized list of skills."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: list[Any] = list(value)
    else:
        items = re.split(r"[,|;\n]", str(value))
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normed = normalize_skill(item)
        if normed and normed not in seen:
            seen.add(normed)
            result.append(normed)
    return result


def _candidate_skills(cv: dict[str, Any]) -> list[str]:
    fields = (
        "skills", "resume_skills", "technical_skills", "programming_languages",
        "frameworks", "libraries", "databases", "cloud_platforms",
        "devops_tools", "testing_tools", "keywords",
    )
    raw: list[Any] = [item for field in fields for item in _parse_skills(cv.get(field))]
    return _parse_skills(raw)


def _candidate_years(cv: dict[str, Any]) -> float:
    for key in ("experience_years", "years_experience", "years_of_experience"):
        val = cv.get(key)
        if val is not None:
            return _as_float(val)
    return 0.0


def _candidate_certifications(cv: dict[str, Any]) -> tuple[int, int]:
    """Return (has_cert, cert_count)."""
    raw = cv.get("certifications", cv.get("certification", ""))
    text = _as_text(raw).strip()
    if not text or text.casefold() in ("none", "n/a", ""):
        return 0, 0
    count = len(re.split(r"[,|;\n]", text))
    return 1, max(1, count)


def _title_similarity(cv_title: str, job_title: str) -> float:
    cv_words = set(_WORD_RE.findall(cv_title.casefold()))
    job_words = set(_WORD_RE.findall(job_title.casefold()))
    if not cv_words or not job_words:
        return 0.0
    return len(cv_words & job_words) / len(cv_words | job_words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pair_features(
    cv: dict[str, Any],
    job: dict[str, Any],
    *,
    semantic_similarity: float = 0.0,
    faiss_similarity: float = 0.0,
) -> dict[str, float]:
    """Build the canonical XGBoost feature vector for one CV ↔ Job pair.

    Parameters
    ----------
    cv:
        Parsed CV dictionary (from resume parser or CSV row).
    job:
        Parsed job description dictionary.
    semantic_similarity:
        Cosine similarity between CV and job MPNet embeddings, in [0, 1].
        If you have the raw cosine similarity from the matcher (0–100 scale),
        divide by 100 before passing here.
    faiss_similarity:
        Raw FAISS inner-product / cosine score returned by FAISSStore.

    Returns
    -------
    dict[str, float]
        Feature dict whose keys are exactly ``PAIR_FEATURE_NAMES`` in order.
        All values are Python ``float``.
    """
    # ── Skills ───────────────────────────────────────────────────────────────
    candidate_skills = _candidate_skills(cv)
    required_skills = _parse_skills(job.get("required_skills"))
    preferred_skills = _parse_skills(job.get("preferred_skills"))

    req_match = calculate_skill_match(candidate_skills, required_skills)
    pref_match = calculate_skill_match(candidate_skills, preferred_skills)

    matched_req = len(req_match["matched"])
    missing_req = len(req_match["missing"])
    total_req = len(required_skills)
    matched_pref = len(pref_match["matched"])

    req_score = req_match["score"] / 100.0  # → [0, 1]
    pref_score = pref_match["score"] / 100.0

    # ── Experience ───────────────────────────────────────────────────────────
    cand_years = _candidate_years(cv)
    req_years_raw = job.get("job_experience_required", job.get("years_of_experience"))
    req_years = _as_float(req_years_raw)
    exp_gap = cand_years - req_years
    exp_satisfies = 1.0 if cand_years >= req_years else 0.0
    exp_score = calculate_experience_score(cand_years, req_years) / 100.0

    # ── Education ────────────────────────────────────────────────────────────
    cand_edu = _education_level(
        cv.get("education_level", cv.get("highest_degree", cv.get("education")))
    )
    job_edu = _education_level(
        job.get("education_level", job.get("education"))
    )
    edu_meets = 1.0 if (cand_edu < 0 or job_edu < 0 or cand_edu >= job_edu) else 0.0

    # ── Certifications ───────────────────────────────────────────────────────
    has_cert, cert_count = _candidate_certifications(cv)

    # ── Portfolio ─────────────────────────────────────────────────────────────
    has_portfolio_raw = cv.get("has_portfolio", cv.get("portfolio"))
    if isinstance(has_portfolio_raw, str):
        has_portfolio = float(has_portfolio_raw.strip().casefold() in {"1", "true", "yes", "y"})
    else:
        has_portfolio = float(bool(has_portfolio_raw))

    # ── Resume text length ───────────────────────────────────────────────────
    resume_text = _as_text(cv.get("resume_text", cv.get("raw_text", cv.get("cleaned_text", ""))))
    resume_len = float(len(resume_text))

    # ── Titles ───────────────────────────────────────────────────────────────
    cand_title = _as_text(cv.get("current_title", cv.get("current_position", "")))
    job_title = _as_text(job.get("job_title", ""))
    title_sim = _title_similarity(cand_title, job_title)

    # ── Skill coverage ratio ──────────────────────────────────────────────────
    skill_coverage = matched_req / (total_req + 1)  # +1 avoids div-by-zero

    # ── Assemble in PAIR_FEATURE_NAMES order ──────────────────────────────────
    features: dict[str, float] = {
        "semantic_similarity":           float(max(0.0, min(1.0, semantic_similarity))),
        "faiss_similarity":              float(faiss_similarity),
        "required_skill_score":          float(req_score),
        "matched_required_skill_count":  float(matched_req),
        "missing_required_skill_count":  float(missing_req),
        "total_required_skill_count":    float(total_req),
        "preferred_skill_score":         float(pref_score),
        "matched_preferred_skill_count": float(matched_pref),
        "candidate_skill_count":         float(len(candidate_skills)),
        "skill_coverage_ratio":          float(skill_coverage),
        "candidate_experience_years":    float(cand_years),
        "required_experience_years":     float(req_years),
        "experience_gap":                float(exp_gap),
        "experience_satisfies":          float(exp_satisfies),
        "experience_score":              float(exp_score),
        "education_level_score":         float(cand_edu),
        "education_meets_requirement":   float(edu_meets),
        "has_certification":             float(has_cert),
        "certification_count":           float(cert_count),
        "has_portfolio":                 float(has_portfolio),
        "resume_text_length":            float(resume_len),
        "title_similarity":              float(title_sim),
    }

    # Sanity check — keys must equal PAIR_FEATURE_NAMES exactly.
    assert set(features) == set(PAIR_FEATURE_NAMES), (
        f"build_pair_features produced unexpected keys. "
        f"Extra: {set(features) - set(PAIR_FEATURE_NAMES)}, "
        f"Missing: {set(PAIR_FEATURE_NAMES) - set(features)}"
    )
    return features


def validate_feature_row(row: dict[str, float], *, allow_extra: bool = False) -> None:
    """Validate that *row* contains all expected features and no spurious ones.

    Parameters
    ----------
    row:
        Feature dict to validate.
    allow_extra:
        If ``False`` (default), raises ``ValueError`` when unexpected keys
        are present.  Set to ``True`` only for debugging.

    Raises
    ------
    ValueError
        When required features are missing or (if not allow_extra) unexpected
        features are present.
    """
    expected = set(PAIR_FEATURE_NAMES)
    present = set(row)
    missing = expected - present
    if missing:
        raise ValueError(
            f"Missing required XGBoost features: {sorted(missing)}\n"
            f"Expected {len(expected)} features: {PAIR_FEATURE_NAMES}"
        )
    if not allow_extra:
        extra = present - expected
        if extra:
            raise ValueError(
                f"Unexpected XGBoost features detected (schema drift): {sorted(extra)}\n"
                f"Expected exactly: {PAIR_FEATURE_NAMES}"
            )


def feature_vector(row: dict[str, float]) -> list[float]:
    """Convert *row* to an ordered list following ``PAIR_FEATURE_NAMES``.

    Always validates the row first — never silently fills missing values.
    """
    validate_feature_row(row)
    return [row[name] for name in PAIR_FEATURE_NAMES]
