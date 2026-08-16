"""Command-line CSV candidate ranker."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.embeddings import EmbeddingCache
from src.matching import CVJobMatcher
from src.preprocessing import preprocess_cv, preprocess_job


def _csv_records(path: str, kind: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path).fillna("")
    if frame.empty:
        raise ValueError(f"{kind} CSV is empty: {path}")
    return frame.to_dict("records")


def _value(row: dict[str, Any], *names: str) -> Any:
    """Read either documented snake_case or bundled dataset column names."""
    normalized = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.casefold())
        if value not in (None, ""):
            return value
    return ""


def _split_skills(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").replace("|", ",").split(",") if item.strip()]


def load_cvs(path: str) -> list[dict[str, Any]]:
    cvs = []
    for row in _csv_records(path, "CV"):
        parsed = preprocess_cv(_value(row, "cv_text", "raw_text"))
        parsed["candidate_name"] = _value(row, "candidate_name") or parsed.get("candidate_name")
        cvs.append(parsed)
    return cvs


def load_job(path: str, job_index: int = 0) -> dict[str, Any]:
    rows = _csv_records(path, "Job")
    if not 0 <= job_index < len(rows):
        raise ValueError(f"job_index must be between 0 and {len(rows) - 1}")
    row = rows[job_index]
    job = preprocess_job(
        _value(row, "job_description", "summary"),
        _value(row, "job_title", "job title"),
    )
    job["required_skills"] = _split_skills(_value(row, "required_skills", "required skills"))
    job["years_of_experience"] = _value(row, "years_of_experience", "experience years") or 0
    job["education"] = _split_skills(_value(row, "education", "education requirement"))
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank CVs against one job description.")
    parser.add_argument("--cv", required=True, help="CSV with candidate_name and cv_text columns")
    parser.add_argument("--job", required=True, help="CSV with job_title and job_description columns")
    parser.add_argument("--job-index", type=int, default=0,
                        help="Zero-based job row to rank against (default: 0)")
    parser.add_argument("--faiss-index", help="Optional FAISS index for candidate retrieval")
    parser.add_argument("--faiss-metadata", help="Metadata JSON for --faiss-index")
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Candidates to retrieve before detailed ranking (default: 50)")
    args = parser.parse_args()
    cvs, job = load_cvs(args.cv), load_job(args.job, args.job_index)
    print("Loading embedding model...")
    print("Model: sentence-transformers/all-mpnet-base-v2")
    print("\nEncoding job description and CVs...\n")
    matcher = CVJobMatcher(embedding_cache=EmbeddingCache())
    if args.faiss_index or args.faiss_metadata:
        if not args.faiss_index or not args.faiss_metadata:
            parser.error("--faiss-index and --faiss-metadata must be supplied together")
        from src.vector_store import FAISSStore, MetadataStore
        results = matcher.rank_retrieved_candidates(
            cvs, job, FAISSStore.load(args.faiss_index), MetadataStore(args.faiss_metadata), args.retrieval_k,
        )
    else:
        results = matcher.rank_candidates(cvs, job)
    print("=" * 50 + "\nCV MATCHING RESULTS\n" + "=" * 50)
    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. {result['candidate_name']}")
        print(f"   Final Score: {result['final_score']:.2f}")
        print(f"   Semantic: {result['semantic_score']:.2f}")
        print(f"   Required Skills: {result['required_skill_score']:.2f}")
        print(f"   Preferred Skills: {result['preferred_skill_score']:.2f}")
        print(f"   Experience: {result['experience_score']:.2f}")


if __name__ == "__main__":
    main()
