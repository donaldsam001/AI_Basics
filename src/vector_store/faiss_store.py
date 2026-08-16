"""Validated persistence wrapper around an exact FAISS cosine-similarity index."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FAISSStore:
    """Store normalized embeddings in an ``IndexFlatIP`` index.

    Normalizing both indexed and query vectors means the returned inner-product
    scores are cosine similarities.
    """

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.index = faiss.IndexFlatIP(int(dimension))

    def _validate_embeddings(self, embeddings: np.ndarray, *, name: str) -> np.ndarray:
        vectors = np.array(embeddings, dtype=np.float32, copy=True)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError(f"{name} must be a one- or two-dimensional array")
        if vectors.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one vector")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"{name} dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"{name} must contain only finite values")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError(f"{name} must not contain zero vectors")
        return vectors

    def add(self, embeddings: np.ndarray) -> None:
        vectors = self._validate_embeddings(embeddings, name="embeddings")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k must be a positive integer")
        if self.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        query = self._validate_embeddings(query_embedding, name="query_embedding")
        if query.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one vector")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(int(k), self.size))
        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))

    @classmethod
    def load(cls, path: str | Path) -> "FAISSStore":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"FAISS index file does not exist: {source}")
        try:
            index = faiss.read_index(str(source))
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Unable to read FAISS index {source}: it may be corrupted") from error
        if index.d <= 0 or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(f"FAISS index {source} is not an inner-product index")
        store = cls(index.d)
        store.index = index
        return store

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self.index.d)
