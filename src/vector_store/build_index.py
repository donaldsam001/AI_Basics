"""Build the persistent CV FAISS index from the existing embedding cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings import EmbeddingCache, EmbeddingModel
from src.main import load_cvs
from src.preprocessing import build_cv_embedding_text

from .faiss_store import FAISSStore


DEFAULT_CV_PATH = Path("data/preprocess/preprocessed_cvs.csv")
DEFAULT_EMBEDDING_PATH = Path("data/embeddings/cv_embeddings.npy")
DEFAULT_INDEX_PATH = Path("data/faiss/cv.index")
DEFAULT_METADATA_PATH = Path("data/faiss/cv_metadata.json")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_source_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CV CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CV CSV is empty: {path}")
    return rows


def _cached_vectors_for_texts(embedding_path: Path, texts: list[str]) -> np.ndarray | None:
    """Return cache vectors in CV-row order, including duplicate CV texts."""
    metadata_path = embedding_path.with_suffix(".json")
    if not embedding_path.exists() and not metadata_path.exists():
        return None
    if not embedding_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            "CV embedding cache is incomplete; expected both "
            f"{embedding_path} and {metadata_path}"
        )
    try:
        cached = np.load(embedding_path)
        locations = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load CV embedding cache at {embedding_path}") from error
    if cached.ndim != 2 or cached.shape[0] == 0:
        raise ValueError(f"CV embedding cache has invalid shape {cached.shape}")
    if not isinstance(locations, dict):
        raise ValueError("CV embedding cache metadata must be a JSON object")
    hashes = [_text_hash(text) for text in texts]
    if any(item not in locations for item in hashes):
        return None
    try:
        indices = [int(locations[item]) for item in hashes]
    except (TypeError, ValueError) as error:
        raise ValueError("CV embedding cache metadata contains invalid vector indices") from error
    if any(index < 0 or index >= len(cached) for index in indices):
        raise ValueError("CV embedding cache metadata refers to a missing vector")
    return np.asarray(cached[indices], dtype=np.float32)


def _metadata(rows: list[dict[str, str]], cvs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for source_index, (row, cv) in enumerate(zip(rows, cvs, strict=True)):
        candidate_id = (row.get("candidate_id") or "").strip() or f"candidate_{source_index:06d}"
        candidate_name = (row.get("candidate_name") or "").strip() or cv.get("candidate_name") or "Unknown Candidate"
        records.append({
            "faiss_id": source_index,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "source_index": source_index,
        })
    return records


def build_cv_index(
    cv_path: str | Path = DEFAULT_CV_PATH,
    embedding_path: str | Path = DEFAULT_EMBEDDING_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    *,
    rebuild: bool = False,
) -> FAISSStore:
    """Create a fresh CV index and its positional candidate metadata."""
    del rebuild  # Every build creates a fresh index, preventing duplicate vectors.
    cv_path, embedding_path = Path(cv_path), Path(embedding_path)
    index_path, metadata_path = Path(index_path), Path(metadata_path)
    print("Loading CVs...")
    rows = _load_source_rows(cv_path)
    cvs = load_cvs(str(cv_path))
    if len(cvs) != len(rows):
        raise ValueError(f"Loaded {len(cvs)} CVs but CSV contains {len(rows)} rows")
    print(f"Loaded {len(cvs)} CVs")
    texts = [build_cv_embedding_text(cv) for cv in cvs]

    print("Loading cached embeddings...")
    embeddings = _cached_vectors_for_texts(embedding_path, texts)
    if embeddings is None:
        print("Cache is missing required CV texts; generating only missing embeddings...")
        cache = EmbeddingCache(embedding_path.parent)
        embeddings = cache.encode(texts, EmbeddingModel(), "cv")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cvs):
        raise ValueError(
            f"Embedding count mismatch: loaded {embeddings.shape[0] if embeddings.ndim else 0} "
            f"embeddings for {len(cvs)} CVs"
        )
    print(f"Loaded embeddings: {embeddings.shape}")

    print("Building FAISS index...")
    store = FAISSStore(embeddings.shape[1])
    store.add(embeddings)
    records = _metadata(rows, cvs)
    if store.size != len(records):
        raise ValueError(f"Index contains {store.size} vectors but metadata contains {len(records)} records")
    print(f"Dimension: {store.dimension}\nVectors: {store.size}")
    store.save(index_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saving:\n  {index_path}\n  {metadata_path}\nFAISS index successfully built.")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persistent CV FAISS index.")
    parser.add_argument("--cv-path", default=DEFAULT_CV_PATH)
    parser.add_argument("--embedding-path", default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild a fresh index (the default behavior).")
    args = parser.parse_args()
    build_cv_index(args.cv_path, args.embedding_path, args.index_path, args.metadata_path, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
