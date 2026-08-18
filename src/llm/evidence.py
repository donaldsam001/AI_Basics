"""Structured evidence builder for the Qwen explanation layer.

This module converts matching pipeline results into a compact evidence
dictionary that carries only the information Qwen needs to generate a
grounded, hallucination-resistant explanation.

Full CVs and full job descriptions are intentionally excluded.  The
evidence object reuses data already produced by the FAISS retrieval,
feature engineering, and XGBoost scoring stages so that no duplicate
extraction or embedding work is performed.
"""

from __future__ import annotations

from typing import Any

from src.features.feature_engineering import candidate_skills, parse_skills
from src.preprocessing.common import clean_text, display_items


def _format_experience(candidate: dict[str, Any]) -> str:
    """Build a concise experience string from candidate data."""
    years = candidate.get("years_of_experience", candidate.get("years_experience"))
    if years not in (None, "", 0, 0.0):
        return f"{years} years"
    return "Not specified"


def _format_projects(candidate: dict[str, Any]) -> list[str]:
    """Extract concise project descriptions from parsed CV data."""
    raw_projects = candidate.get("projects", [])
    return display_items(raw_projects)


def _format_education(candidate: dict[str, Any]) -> list[str]:
    """Extract education entries from parsed CV data."""
    education = candidate.get("education", [])
    return display_items(education)


def build_evidence(
    candidate: dict[str, Any],
    job: dict[str, Any],
    match_result: dict[str, Any],
    rank: int = 1,
) -> dict[str, Any]:
    """Build a compact evidence dictionary for the Qwen explanation layer.

    Parameters
    ----------
    candidate:
        The parsed CV dictionary as produced by the preprocessing pipeline.
    job:
        The structured job description dictionary.
    match_result:
        The matching result dictionary containing scores and skill analysis
        as returned by :meth:`CVJobMatcher.match` or
        :meth:`CVJobMatcher.rank_retrieved_candidates`.
    rank:
        The candidate's rank position (1-based) in the final ranking.

    Returns
    -------
    dict
        A structured evidence object containing only the fields required
        for explanation generation.
    """
    # Candidate evidence -------------------------------------------------
    candidate_name = (
        match_result.get("candidate_name")
        or candidate.get("candidate_name")
        or "Unknown Candidate"
    )

    matched_skills = match_result.get("matched_skills", [])
    missing_skills = match_result.get("missing_required_skills", [])

    # Collect skills the candidate has that match *preferred* requirements.
    matched_preferred = match_result.get("matched_preferred_skills", [])

    candidate_evidence = {
        "name": clean_text(candidate_name),
        "matched_required_skills": list(matched_skills),
        "missing_required_skills": list(missing_skills),
        "matched_preferred_skills": list(matched_preferred),
        "experience": _format_experience(candidate),
        "projects": _format_projects(candidate),
        "education": _format_education(candidate),
    }

    # Job evidence -------------------------------------------------------
    job_evidence = {
        "title": clean_text(job.get("job_title", "Untitled Job")),
        "required_skills": [str(skill) for skill in job.get("required_skills", [])],
        "preferred_skills": [str(skill) for skill in job.get("preferred_skills", [])],
        "experience_requirement": (
            f"{job.get('years_of_experience')} years"
            if job.get("years_of_experience") not in (None, "", 0, 0.0)
            else "Not specified"
        ),
        "responsibilities": display_items(job.get("responsibilities", [])),
    }

    # Model result -------------------------------------------------------
    xgb_probability = match_result.get("xgb_probability")
    if xgb_probability is None:
        # Fall back to the deterministic final score when XGBoost is absent.
        xgb_probability = match_result.get("final_score", 0.0)
        # Normalise the 0-100 deterministic score to 0-1 for consistency.
        if xgb_probability > 1.0:
            xgb_probability = round(xgb_probability / 100.0, 4)

    model_result = {
        "xgboost_probability": round(float(xgb_probability), 4),
        "rank": rank,
    }

    # Optional FAISS similarity for additional context.
    faiss_sim = match_result.get("faiss_similarity")
    if faiss_sim is not None:
        model_result["faiss_similarity"] = round(float(faiss_sim), 4)

    return {
        "candidate": candidate_evidence,
        "job": job_evidence,
        "model_result": model_result,
    }
