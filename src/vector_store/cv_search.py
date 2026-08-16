"""High-level CV retrieval using the project's job text representation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.embeddings import EmbeddingModel
from src.preprocessing import build_job_embedding_text, preprocess_job

from .faiss_store import FAISSStore
from .metadata_store import MetadataStore


class CVSearch:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: FAISSStore,
                 metadata_store: MetadataStore) -> None:
        metadata_store.validate_index_size(vector_store.size)
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def search(self, query_text: str, k: int = 10) -> list[dict[str, Any]]:
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        job_text = build_job_embedding_text(preprocess_job(query_text))
        query_embedding = self.embedding_model.encode([job_text], show_progress_bar=False)
        scores, ids = self.vector_store.search(query_embedding, k)
        results = []
        for score, faiss_id in zip(scores, ids, strict=True):
            candidate = self.metadata_store.get_candidate(int(faiss_id))
            results.append({**candidate, "score": float(score)})
        return sorted(results, key=lambda result: result["score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the persistent CV FAISS index.")
    parser.add_argument("query", nargs="?", default="Backend Java Developer with Spring Boot")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--index-path", default="data/faiss/cv.index")
    parser.add_argument("--metadata-path", default="data/faiss/cv_metadata.json")
    args = parser.parse_args()
    search = CVSearch(EmbeddingModel(), FAISSStore.load(Path(args.index_path)), MetadataStore(args.metadata_path))
    print(f"Query: {args.query}\n\nTop candidates:")
    for rank, candidate in enumerate(search.search(args.query, args.k), 1):
        print(f"\n{rank}. {candidate['candidate_name']}")
        print(f"   Candidate ID: {candidate['candidate_id']}")
        print(f"   Similarity: {candidate['score']:.4f}")


if __name__ == "__main__":
    main()
