"""Work history, current title, and total experience estimation."""

from __future__ import annotations

import math
import re

from resume_parser.utils import compact_dict, extract_date_range, find_section, lines_to_blocks, merge_month_ranges

TITLE_RE = re.compile(
    r"(?i)\b(?:software|backend|frontend|full[ -]?stack|data|devops|cloud|"
    r"machine learning|qa|test|security|systems?|product)\s+(?:engineer|developer|scientist|analyst|manager|designer)\b"
)
DECLARED_YEARS_RE = re.compile(r"(?i)\b(\d+)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience\b")


def _looks_like_company(line: str) -> bool:
    return bool(re.search(r"(?i)\b(?:inc\.?|llc|ltd\.?|corp\.?|company|co\.?|technologies|solutions)\b", line))


def extract_work_experience(text: str) -> list[dict[str, object]]:
    section = find_section(text, ("work experience", "professional experience", "experience", "employment", "work history"))
    entries = []
    for block in lines_to_blocks(section):
        start, end = extract_date_range(block)
        lines = [line.strip("•- ") for line in block.splitlines() if line.strip()]
        if not lines or (not start and not any(TITLE_RE.search(line) for line in lines)):
            continue
        date_header = next(
            (line for line in lines if re.search(r"(?i)(?:19|20)\d{2}|present|current", line)),
            None,
        )
        header = [line for line in lines if line != date_header]
        header_parts = []
        if date_header:
            # Common CV format: "Position | Company | Jan 2021 - Present".
            undated = re.sub(
                r"(?i)\s*[|,]?\s*(?:[A-Za-z]{3,9}\.?\s+)?(?:19|20)\d{2}"
                r"\s*(?:-|–|—|to)\s*(?:(?:[A-Za-z]{3,9}\.?\s+)?(?:19|20)\d{2}|present|current|now)\s*$",
                "", date_header,
            ).strip(" |,-–—")
            header_parts = [part.strip() for part in re.split(r"\s*\|\s*", undated) if part.strip()]
        position_line = next(
            (line for line in header_parts + header if TITLE_RE.search(line)),
            (header_parts + header)[0] if header_parts or header else None,
        )
        company_line = next((line for line in header if line != position_line and _looks_like_company(line)), None)
        if not company_line and len(header_parts) > 1:
            company_line = next((line for line in header_parts if line != position_line), None)
        if not company_line and len(header) > 1:
            company_line = header[1] if position_line == header[0] else header[0]
        responsibilities = [
            line for line in lines
            if line not in {position_line, company_line, date_header}
        ]
        entries.append(compact_dict(
            company=company_line, position=position_line, start_date=start, end_date=end,
            responsibilities=responsibilities,
        ))
    return entries


def estimate_years_of_experience(text: str, work_experience: list[dict[str, object]]) -> int:
    """Use non-overlapping work ranges, falling back to a stated years value."""
    months = merge_month_ranges(
        (item.get("start_date"), item.get("end_date")) for item in work_experience
    )
    if months:
        return max(1, math.floor(months / 12))
    declared = DECLARED_YEARS_RE.search(text)
    return int(declared.group(1)) if declared else 0


def current_position(work_experience: list[dict[str, object]]) -> str | None:
    """Prefer the first ongoing position, then the most recently listed role."""
    if not work_experience:
        return None
    return work_experience[0].get("position")  # type: ignore[return-value]
