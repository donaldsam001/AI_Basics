"""Dictionary-driven skill extraction."""

from __future__ import annotations

from resume_parser.dictionaries.skills import (
    CLOUD_PLATFORMS, DATABASES, DEVOPS_TOOLS, DOMAINS, FRAMEWORKS, INDUSTRIES,
    LIBRARIES, OPERATING_SYSTEMS, PROGRAMMING_LANGUAGES, SOFT_SKILLS,
    SPOKEN_LANGUAGES, TECHNICAL_SKILLS, TESTING_TOOLS,
)
from resume_parser.utils import extract_items, unique_sorted


def extract_skill_groups(text: str) -> dict[str, list[str]]:
    """Extract every dictionary-controlled skill category from resume text."""
    groups = {
        "programming_languages": extract_items(text, PROGRAMMING_LANGUAGES),
        "frameworks": extract_items(text, FRAMEWORKS),
        "libraries": extract_items(text, LIBRARIES),
        "databases": extract_items(text, DATABASES),
        "cloud_platforms": extract_items(text, CLOUD_PLATFORMS),
        "devops_tools": extract_items(text, DEVOPS_TOOLS),
        "operating_systems": extract_items(text, OPERATING_SYSTEMS),
        "testing_tools": extract_items(text, TESTING_TOOLS),
        "soft_skills": extract_items(text, SOFT_SKILLS),
        "spoken_languages": extract_items(text, SPOKEN_LANGUAGES),
        "industries": extract_items(text, INDUSTRIES),
        "domains": extract_items(text, DOMAINS),
    }
    generic = extract_items(text, TECHNICAL_SKILLS)
    groups["technical_skills"] = unique_sorted(
        generic + groups["programming_languages"] + groups["frameworks"] +
        groups["libraries"] + groups["databases"] + groups["cloud_platforms"] +
        groups["devops_tools"] + groups["testing_tools"]
    )
    return groups
