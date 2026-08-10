"""Shared lightweight normalisation utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def clean_text(value: Any) -> str:
    """Normalize whitespace without discarding meaningful technical punctuation."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def display_items(values: Iterable[Any]) -> list[str]:
    """Convert parser dictionaries or values to concise displayable strings."""
    items: list[str] = []
    for value in values:
        if isinstance(value, dict):
            parts = [clean_text(part) for part in value.values() if part]
            value = " ".join(parts)
        text = clean_text(value)
        if text and text not in items:
            items.append(text)
    return items


def section(label: str, values: Iterable[Any]) -> str:
    items = display_items(values)
    return f"{label}:\n{', '.join(items)}" if items else ""
