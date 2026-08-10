"""One reusable Sentence Transformers model instance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class EmbeddingModel:
    """Lazy CPU embedding wrapper for all-mpnet-base-v2.

    Importing this module does not require sentence-transformers. The dependency
    is only imported when an embedding is actually requested, keeping unit tests
    and text-only workflows lightweight.
    """

    model_name = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: Sequence[str], show_progress_bar: bool = True) -> np.ndarray:
        """Return normalized embeddings using a small CPU-safe batch size."""
        return np.asarray(self.model.encode(
            list(texts),
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        ))
