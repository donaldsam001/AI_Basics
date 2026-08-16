"""Candidate metadata associated with FAISS's positional identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the list stored alongside a FAISS index."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"FAISS metadata file does not exist: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read FAISS metadata {source}: it may be corrupted") from error
    if not isinstance(data, list):
        raise ValueError("FAISS metadata must be a JSON list")
    ids: set[int] = set()
    records: list[dict[str, Any]] = []
    for position, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Metadata record {position} must be an object")
        faiss_id = record.get("faiss_id")
        if isinstance(faiss_id, bool) or not isinstance(faiss_id, int) or faiss_id < 0:
            raise ValueError(f"Metadata record {position} has an invalid faiss_id")
        if faiss_id in ids:
            raise ValueError(f"Metadata contains duplicate faiss_id {faiss_id}")
        ids.add(faiss_id)
        records.append(record.copy())
    return records


class MetadataStore:
    """Look up candidate records by the integer ID returned by FAISS."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records = load_metadata(self.path)
        self._by_id = {record["faiss_id"]: record for record in self.records}

    def get_candidate(self, faiss_id: int) -> dict[str, Any]:
        try:
            return self._by_id[int(faiss_id)].copy()
        except (KeyError, TypeError, ValueError) as error:
            raise KeyError(f"No candidate metadata found for FAISS ID {faiss_id}") from error

    def validate_index_size(self, index_size: int) -> None:
        if index_size != len(self.records):
            raise ValueError(
                f"FAISS index contains {index_size} vectors but metadata contains {len(self.records)} records"
            )
        if set(self._by_id) != set(range(index_size)):
            raise ValueError("FAISS metadata IDs must exactly cover 0 through index size minus one")

    def __len__(self) -> int:
        return len(self.records)
