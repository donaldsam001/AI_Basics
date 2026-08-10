"""Content-addressed, NumPy embedding cache for repeat CLI runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .model import EmbeddingModel


class EmbeddingCache:
    """Cache embeddings by SHA-256 of their source text within a namespace."""

    def __init__(self, directory: str | Path = "data/embeddings") -> None:
        self.directory = Path(directory)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str], model: EmbeddingModel, namespace: str) -> np.ndarray:
        """Return cached vectors and encode only source texts not already stored."""
        self.directory.mkdir(parents=True, exist_ok=True)
        vectors_path = self.directory / f"{namespace}_embeddings.npy"
        metadata_path = self.directory / f"{namespace}_embeddings.json"
        hashes = [self._hash(text) for text in texts]
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        vectors = np.load(vectors_path) if vectors_path.exists() else np.empty((0, 0))
        cache = {
            key: vectors[index] for key, index in metadata.items()
            if index < len(vectors)
        }
        missing = list(dict.fromkeys(text for text, key in zip(texts, hashes) if key not in cache))
        if missing:
            generated = model.encode(missing)
            for text, vector in zip(missing, generated, strict=True):
                cache[self._hash(text)] = vector
            all_hashes = sorted(cache)
            vectors = np.asarray([cache[key] for key in all_hashes])
            np.save(vectors_path, vectors)
            metadata_path.write_text(json.dumps({key: index for index, key in enumerate(all_hashes)}))
        return np.asarray([cache[key] for key in hashes])
