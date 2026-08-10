"""Public parser facade that composes independent extraction modules."""

from __future__ import annotations

import re
from typing import Any

from resume_parser.extractors.achievements import extract_achievements
from resume_parser.extractors.certification import extract_certifications
from resume_parser.extractors.education import extract_education, extract_gpa
from resume_parser.extractors.experience import (
    current_position, estimate_years_of_experience, extract_work_experience,
)
from resume_parser.extractors.projects import extract_projects
from resume_parser.extractors.skills import extract_skill_groups
from resume_parser.utils import clean_value, normalize_whitespace, unique_sorted


def extract_candidate_name(text: str) -> str | None:
    """Use the first plausible, non-contact line as a conservative name heuristic."""
    blocked = re.compile(r"(?i)(resume|curriculum|vitae|email|phone|linkedin|github|summary|objective)")
    for line in normalize_whitespace(text).splitlines()[:8]:
        candidate = clean_value(line)
        if not candidate or blocked.search(candidate) or "@" in candidate or re.search(r"\d{3}", candidate):
            continue
        if re.fullmatch(r"[A-Za-zÀ-ÿ.' -]{2,80}", candidate) and 1 < len(candidate.split()) <= 5:
            return candidate
    return None


def _job_title_keywords(work_experience: list[dict[str, Any]]) -> list[str]:
    return [str(item["position"]) for item in work_experience if item.get("position")]


class ResumeParser:
    """Rule-based parser. Dictionaries and extractors can be replaced independently."""

    def parse(self, raw_text: str) -> dict[str, Any]:
        text = normalize_whitespace(raw_text)
        skills = extract_skill_groups(text)
        education = extract_education(text)
        work_experience = extract_work_experience(text)
        certifications = extract_certifications(text)
        projects = extract_projects(text)
        achievements = extract_achievements(text)
        certification_names = [item["certification_name"] for item in certifications if item.get("certification_name")]
        keywords = unique_sorted(
            skills["technical_skills"] + skills["soft_skills"] + skills["domains"] +
            _job_title_keywords(work_experience) + certification_names
        )
        # Keep this explicit rather than expanding ``skills`` so callers always
        # receive the documented response shape and order.
        return {
            "candidate_name": extract_candidate_name(text),
            "current_position": current_position(work_experience),
            "years_of_experience": estimate_years_of_experience(text, work_experience),
            "education": education,
            "gpa": extract_gpa(text),
            "technical_skills": skills["technical_skills"],
            "soft_skills": skills["soft_skills"],
            "programming_languages": skills["programming_languages"],
            "frameworks": skills["frameworks"],
            "libraries": skills["libraries"],
            "databases": skills["databases"],
            "cloud_platforms": skills["cloud_platforms"],
            "devops_tools": skills["devops_tools"],
            "operating_systems": skills["operating_systems"],
            "testing_tools": skills["testing_tools"],
            "certifications": certifications,
            "projects": projects,
            "work_experience": work_experience,
            "achievements": achievements,
            "industries": skills["industries"],
            "domains": skills["domains"],
            "spoken_languages": skills["spoken_languages"],
            "keywords": keywords,
        }


def parse_resume(raw_text: str) -> dict[str, Any]:
    """Parse a resume string into the documented response schema."""
    return ResumeParser().parse(raw_text)
