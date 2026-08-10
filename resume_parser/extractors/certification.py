"""Certification extraction."""

from __future__ import annotations

import re

from resume_parser.utils import (
    DATE_TOKEN, compact_dict, extract_date_range, find_section, lines_to_blocks,
    normalize_date,
)


def extract_certifications(text: str) -> list[dict[str, str | None]]:
    section = find_section(text, ("certifications", "certification", "licenses", "licenses and certifications"))
    entries = []
    for block in lines_to_blocks(section):
        line = " ".join(block.splitlines()).strip("•- ")
        if not line:
            continue
        start, end = extract_date_range(line)
        if not start:
            date_match = re.search(DATE_TOKEN, line, re.I)
            start = normalize_date(date_match.group(0)) if date_match else None
        issuer_match = re.search(r"(?i)\b(?:issued by|issuer|from|by)\s*[:=-]?\s*([^,;|]+)", line)
        name = re.split(r"\s*(?:\||—|–|issued by|issuer|from)\s*", line, maxsplit=1, flags=re.I)[0]
        entries.append(compact_dict(
            certification_name=name,
            issuer=issuer_match.group(1) if issuer_match else None,
            issue_date=start, expiration_date=end,
        ))
    return entries
