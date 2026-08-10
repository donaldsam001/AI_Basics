"""Project extraction with technology matching."""

from __future__ import annotations

import re

from resume_parser.extractors.skills import extract_skill_groups
from resume_parser.utils import compact_dict, find_section, lines_to_blocks, unique_sorted


def extract_projects(text: str) -> list[dict[str, object]]:
    section = find_section(text, ("projects", "personal projects", "selected projects", "project experience"))
    entries = []
    for block in lines_to_blocks(section):
        lines = [line.strip("•- ") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0]
        name = re.split(r"\s*(?:\||—|–|-)\s*", first, maxsplit=1)[0]
        role_match = re.search(r"(?im)^\s*(?:role)\s*[:=-]?\s*([^,;|\n]+)", block)
        skills = extract_skill_groups(block)
        technologies = unique_sorted(
            skills["technical_skills"] + skills["programming_languages"] + skills["frameworks"]
        )
        description_lines = [line for line in lines[1:] if not re.match(r"(?i)^role\s*:", line)]
        entries.append(compact_dict(
            project_name=name,
            role=role_match.group(1) if role_match else None,
            technologies=technologies,
            description=" ".join(description_lines) or None,
        ))
    return entries
