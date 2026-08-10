"""Education and GPA extraction heuristics."""

from __future__ import annotations

import re

from resume_parser.utils import compact_dict, extract_date_range, find_section, lines_to_blocks

DEGREE_RE = re.compile(
    r"(?i)\b(?:b\.?\s*(?:sc|s|a)|m\.?\s*(?:sc|s|a)|ph\.?d|doctorate|"
    r"bachelor(?:'s)?|master(?:'s)?|associate(?:'s)?|diploma)\b[^,;|\n]*"
)
INSTITUTION_RE = re.compile(
    r"(?i)\b(?:university|college|institute|school|academy)\b[^,;|\n]*|"
    r"[^,;|\n]*(?:university|college|institute|school|academy)\b"
)
GPA_RE = re.compile(r"(?i)\b(?:gpa|cgpa)\s*[:=-]?\s*(\d{1,2}(?:\.\d{1,2})?)(?:\s*/\s*\d{1,2}(?:\.\d+)?)?")


def extract_gpa(text: str) -> str | None:
    """Return the first explicitly-labelled GPA numeric value."""
    match = GPA_RE.search(text)
    return match.group(1) if match else None


def extract_education(text: str) -> list[dict[str, str | None]]:
    section = find_section(text, ("education", "academic background", "academics"))
    entries = []
    for block in lines_to_blocks(section):
        if not DEGREE_RE.search(block) and not INSTITUTION_RE.search(block):
            continue
        lines = block.splitlines()
        degree_match = DEGREE_RE.search(block)
        institution_match = INSTITUTION_RE.search(block)
        faculty_match = re.search(r"(?i)\b(?:faculty|school|college)\s+of\s+[^,;\n]+", block)
        major_match = re.search(r"(?i)\b(?:major(?:ing)?\s+in|field of study)\s*[:=-]?\s*([^,;\n]+)", block)
        if not major_match and degree_match:
            major_match = re.search(
                r"(?i)\b(?:bachelor|master|associate|b\.?\s*(?:sc|s|a)|"
                r"m\.?\s*(?:sc|s|a))[^\n]*?\bin\s+([A-Za-z][A-Za-z &/-]+)",
                degree_match.group(0),
            )
        start, end = extract_date_range(block)
        entries.append(compact_dict(
            university_name=institution_match.group(0) if institution_match else (lines[0] if lines else None),
            faculty=faculty_match.group(0) if faculty_match else None,
            degree=degree_match.group(0) if degree_match else None,
            major=major_match.group(1) if major_match else None,
            gpa=extract_gpa(block), start_date=start, end_date=end,
        ))
    return entries
