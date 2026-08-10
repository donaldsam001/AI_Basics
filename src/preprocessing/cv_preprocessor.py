"""CV parsing and semantic text construction."""

from __future__ import annotations

from typing import Any

from resume_parser import parse_resume

from .common import clean_text, section


def preprocess_cv(raw_text: str) -> dict[str, Any]:
    """Extract the repository's documented structured CV fields from raw text."""
    return parse_resume(clean_text(raw_text))


def build_cv_embedding_text(cv: dict[str, Any]) -> str:
    """Build matching-focused CV text, intentionally excluding personal details."""
    parts = []
    if clean_text(cv.get("current_position")):
        parts.append(f"Position: {clean_text(cv['current_position'])}")
    years = cv.get("years_of_experience")
    if years not in (None, "", 0, 0.0):
        parts.append(f"Experience:\n{years} years")
    labels = (
        ("Programming Languages", "programming_languages"),
        ("Frameworks", "frameworks"), ("Libraries", "libraries"),
        ("Databases", "databases"), ("Cloud", "cloud_platforms"),
        ("DevOps", "devops_tools"), ("Technical Skills", "technical_skills"),
        ("Domains", "domains"), ("Education", "education"),
        ("Certifications", "certifications"), ("Projects", "projects"),
        ("Work Experience", "work_experience"), ("Achievements", "achievements"),
    )
    parts.extend(section(label, cv.get(key, [])) for label, key in labels)
    return "\n\n".join(part for part in parts if part)
