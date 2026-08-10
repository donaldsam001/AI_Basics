"""Awards, scholarship, publication, and accomplishment extraction."""

from __future__ import annotations

import re

from resume_parser.utils import find_section, unique_sorted

ACHIEVEMENT_RE = re.compile(
    r"(?i)\b(?:award|scholarship|winner|won|first place|second place|third place|"
    r"finalist|publication|published|achievement|honou?r|prize)\b"
)


def extract_achievements(text: str) -> list[str]:
    section = find_section(text, ("achievements", "awards", "honors", "publications"))
    source = section or text
    return unique_sorted(
        line.strip("•- ") for line in source.splitlines() if ACHIEVEMENT_RE.search(line)
    )
