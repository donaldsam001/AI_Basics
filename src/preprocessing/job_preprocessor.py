"""Job-description preprocessing and semantic text construction."""

from __future__ import annotations

import re
from typing import Any

from .common import as_list, clean_text, section


_EXPERIENCE_RE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*\+?\s*years?")


def preprocess_job(raw_text: str, job_title: str = "") -> dict[str, Any]:
    """Create a safe structured job record from CSV-style raw job text.

    Callers with richer data can pass a dictionary directly to the text builder.
    """
    text = clean_text(raw_text)
    experience = _EXPERIENCE_RE.search(text)
    return {
        "job_title": clean_text(job_title), "summary": text,
        "required_skills": [], "preferred_skills": [],
        "programming_languages": [], "frameworks": [], "libraries": [],
        "databases": [], "cloud_platforms": [], "devops_tools": [],
        "years_of_experience": float(experience.group(1)) if experience else 0,
        "education": [], "responsibilities": [], "qualifications": [],
        "keywords": [],
    }


def build_job_embedding_text(job: dict[str, Any]) -> str:
    """Build a structured semantic representation of a job description."""
    parts = []
    if clean_text(job.get("job_title")):
        parts.append(f"Job Title:\n{clean_text(job['job_title'])}")
    if clean_text(job.get("summary")):
        parts.append(f"Summary:\n{clean_text(job['summary'])}")
    labels = (
        ("Required Skills", "required_skills"), ("Preferred Skills", "preferred_skills"),
        ("Programming Languages", "programming_languages"), ("Frameworks", "frameworks"),
        ("Libraries", "libraries"), ("Databases", "databases"),
        ("Cloud", "cloud_platforms"), ("DevOps", "devops_tools"),
        ("Education", "education"), ("Responsibilities", "responsibilities"),
        ("Qualifications", "qualifications"), ("Technical Keywords", "keywords"),
    )
    parts.extend(section(label, as_list(job.get(key))) for label, key in labels)
    years = job.get("years_of_experience")
    if years not in (None, "", 0, 0.0):
        parts.append(f"Experience:\n{years}+ years")
    return "\n\n".join(part for part in parts if part)
