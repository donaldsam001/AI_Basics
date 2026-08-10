"""Shared text, section, date, and matching helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any


SECTION_TITLES = (
    "education|experience|employment|work history|professional experience|"
    "skills|technical skills|projects|certifications?|licenses?|"
    "achievements?|awards?|honors?|languages|publications?"
)
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3,
    "march": 3, "apr": 4, "april": 4, "may": 5, "jun": 6,
    "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def normalize_whitespace(text: str | None) -> str:
    """Collapse repeated whitespace while retaining line boundaries."""
    if not isinstance(text, str):
        return ""
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def clean_value(value: str | None) -> str | None:
    """Return a clean scalar, or None for an empty value."""
    cleaned = re.sub(r"\s+", " ", value or "").strip(" -–—|,:;")
    return cleaned or None


def unique_sorted(values: Iterable[str | None]) -> list[str]:
    """Deduplicate case-insensitively and return stable alphabetical output."""
    result: dict[str, str] = {}
    for value in values:
        cleaned = clean_value(value)
        if cleaned:
            result.setdefault(cleaned.casefold(), cleaned)
    return sorted(result.values(), key=str.casefold)


def extract_items(text: str, dictionary: Mapping[str, Iterable[str]]) -> list[str]:
    """Match dictionary synonyms and return canonical, sorted names.

    ``(?<!\\w)`` / ``(?!\\w)`` handles symbols such as C++ and C# better than
    a plain ``\\b`` boundary while still preventing partial-word matches.
    """
    if not isinstance(text, str):
        return []
    matched = []
    for canonical, synonyms in dictionary.items():
        variants = unique_sorted([canonical, *synonyms])
        if any(
            re.search(r"(?<!\w)" + re.escape(item) + r"(?!\w)", text, re.I)
            for item in variants
        ):
            matched.append(canonical)
    return unique_sorted(matched)


def find_section(text: str, headings: Iterable[str]) -> str:
    """Return text below a matching heading and before the next known heading."""
    normalized = normalize_whitespace(text)
    labels = "|".join(re.escape(heading) for heading in headings)
    start = re.search(rf"(?im)^\s*(?:{labels})\s*:?\s*$", normalized)
    if not start:
        return ""
    end = re.search(rf"(?im)^\s*(?:{SECTION_TITLES})\s*:?\s*$", normalized[start.end():])
    return normalized[start.end(): start.end() + end.start()] if end else normalized[start.end():]


def lines_to_blocks(section: str) -> list[str]:
    """Create logical blocks from blank lines or date-bearing header lines."""
    if not section:
        return []
    raw_blocks = re.split(r"\n\s*\n", section)
    blocks: list[str] = []
    date_line = re.compile(r"(?i)(?:19|20)\d{2}.*(?:-|–|—|to|present|current|now)")
    for raw in raw_blocks:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        current: list[str] = []
        for line in lines:
            if current and date_line.search(line) and len(current) >= 2:
                blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
    return blocks


def normalize_date(value: str | None) -> str | None:
    """Normalize a month/year or year to YYYY-MM where it is unambiguous."""
    if not value:
        return None
    candidate = value.strip().casefold().replace(".", "")
    if candidate in {"present", "current", "now", "ongoing"}:
        today = date.today()
        return f"{today.year:04d}-{today.month:02d}"
    numeric = re.fullmatch(r"(\d{1,2})[/-](\d{4})", candidate)
    if numeric:
        month, year = map(int, numeric.groups())
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else None
    named = re.fullmatch(r"([a-z]+)\s+(\d{4})", candidate)
    if named and named.group(1) in MONTHS:
        return f"{named.group(2)}-{MONTHS[named.group(1)]:02d}"
    year = re.fullmatch(r"((?:19|20)\d{2})", candidate)
    return f"{year.group(1)}-01" if year else None


DATE_TOKEN = (
    r"(?:[A-Za-z]{3,9}\.?\s+)?(?:19|20)\d{2}|"
    r"\d{1,2}[/-](?:19|20)\d{2}|present|current|now|ongoing"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:-|–|—|to)\s*(?P<end>{DATE_TOKEN})", re.I
)


def extract_date_range(text: str) -> tuple[str | None, str | None]:
    """Extract the first date range in text."""
    match = DATE_RANGE_RE.search(text)
    if not match:
        return None, None
    return normalize_date(match.group("start")), normalize_date(match.group("end"))


def month_index(value: str | None) -> int | None:
    if not value or not re.fullmatch(r"\d{4}-\d{2}", value):
        return None
    year, month = map(int, value.split("-"))
    return year * 12 + month - 1


def merge_month_ranges(ranges: Iterable[tuple[str | None, str | None]]) -> int:
    """Return duration of overlapping ranges only once, in whole months."""
    intervals = []
    for start, end in ranges:
        left, right = month_index(start), month_index(end)
        if left is not None and right is not None and right >= left:
            intervals.append((left, right))
    if not intervals:
        return 0
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start + 1 for start, end in merged)


def compact_dict(**values: Any) -> dict[str, Any]:
    """Retain a consistent schema while converting blank scalars to None."""
    return {key: (clean_value(value) if isinstance(value, str) else value)
            for key, value in values.items()}
