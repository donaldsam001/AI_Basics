"""Command-line CSV candidate ranker."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from src.embeddings import EmbeddingCache
from src.matching import CVJobMatcher
from src.preprocessing import preprocess_cv, preprocess_job


from pathlib import Path


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    candidates = [
        Path("data/preprocess") / p.name,
        Path("data") / p.name,
        Path("example_data") / p.name,
        Path("example_data/jd") / p.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return path


def _csv_records(path: str, kind: str) -> list[dict[str, Any]]:
    resolved = _resolve_path(path)
    frame = pd.read_csv(resolved).fillna("")
    if frame.empty:
        raise ValueError(f"{kind} CSV is empty: {resolved}")
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
        raw_text = _value(row, "cv_text", "raw_text")
        parsed = preprocess_cv(raw_text)
        cand_name = _value(row, "candidate_name")
        cand_id = _value(row, "candidate_id", "file_name")
        if not cand_name or cand_name.casefold() in ("experience years", "unknown candidate", "experience year"):
            cand_name = cand_id or parsed.get("candidate_name") or "Candidate"
        parsed["candidate_name"] = cand_name
        parsed["candidate_id"] = cand_id
        # Preserve source fields used by the optional supervised model.  The
        # parser remains authoritative for matching fields, while raw source
        # values retain information (such as portfolio presence) it does not
        # extract.
        parsed["raw_text"] = raw_text
        for field in ("years_experience", "highest_degree", "skills", "current_title", "has_portfolio"):
            value = _value(row, field)
            if value not in (None, ""):
                parsed[field] = value
        cvs.append(parsed)
    return cvs



def load_job(path: str, job_index: int = 0, job_title: str | None = None) -> dict[str, Any]:
    rows = _csv_records(path, "Job")
    selected_index = job_index
    if job_title:
        target = job_title.strip().casefold()
        matching_indices = [
            i for i, r in enumerate(rows)
            if target in str(_value(r, "job_title", "job title")).casefold()
        ]
        if matching_indices:
            selected_index = matching_indices[0]
        else:
            logging.warning("No job matching title '%s' found. Defaulting to index %d.", job_title, job_index)

    if not 0 <= selected_index < len(rows):
        raise ValueError(f"job_index must be between 0 and {len(rows) - 1}")
    row = rows[selected_index]
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
    parser.add_argument("--cv", default="preprocessed_cvs.csv", help="CSV with candidate_name and cv_text columns")
    parser.add_argument("--job", default="job_roles_IT_filtered.csv", help="CSV with job_title and job_description columns")
    parser.add_argument("--job-index", type=int, default=0,
                        help="Zero-based job row to rank against (default: 0)")
    parser.add_argument("--job-title", help="Job title to match against (overrides --job-index if matched)")
    parser.add_argument("--faiss-index", help="Optional FAISS index for candidate retrieval")
    parser.add_argument("--faiss-metadata", help="Metadata JSON for --faiss-index")
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Candidates to retrieve before detailed ranking (default: 50)")
    parser.add_argument("--xgb-model", help="Native XGBoost model used to rerank FAISS results")
    parser.add_argument("--xgb-features", help="Feature schema JSON; defaults to xgb_features.json beside the model")
    parser.add_argument("--qwen-model", help="Path to Qwen3-4B GGUF model for explanation generation")
    parser.add_argument("--ranking-top-k", type=int, default=20,
                        help="Maximum candidates to display after ranking (default: 20)")
    parser.add_argument("--explanation-top-k", type=int, default=5,
                        help="Top candidates to generate Qwen explanations for (default: 5)")
    args = parser.parse_args()
    cvs = load_cvs(args.cv)
    job = load_job(args.job, args.job_index, args.job_title)

    print(f"\nJob: {job.get('job_title', 'Untitled Job')}\n")

    matcher = CVJobMatcher(embedding_cache=EmbeddingCache())
    xgb_scorer = None
    if args.xgb_model:
        if not args.faiss_index:
            parser.error("--xgb-model requires FAISS retrieval; supply --faiss-index and --faiss-metadata")
        from src.models import XGBCandidateScorer
        xgb_scorer = XGBCandidateScorer.load(args.xgb_model, args.xgb_features)
    
    faiss_index_path = _resolve_path(args.faiss_index) if args.faiss_index else None
    faiss_metadata_path = _resolve_path(args.faiss_metadata) if args.faiss_metadata else None

    # Automatic fallback if default faiss index/metadata exist in default location
    if not faiss_index_path and Path("data/faiss/cv.index").is_file() and Path("data/faiss/cv_metadata.json").is_file():
        faiss_index_path = "data/faiss/cv.index"
        faiss_metadata_path = "data/faiss/cv_metadata.json"

    if faiss_index_path and faiss_metadata_path:
        from src.vector_store import FAISSStore, MetadataStore
        results = matcher.rank_retrieved_candidates(
            cvs, job, FAISSStore.load(faiss_index_path), MetadataStore(faiss_metadata_path), args.retrieval_k,
            xgb_scorer,
        )
    else:
        results = matcher.rank_candidates(cvs, job)
    
    # Trim to ranking top-K.
    results = results[: args.ranking_top_k]

    if results and "faiss_similarity" in results[0]:
        print("FAISS Top 5")
        print("-" * 40)
        print(f"{'Rank':<5} {'Candidate':<25} {'Score':<10}")
        print("-" * 40)
        for rank, res in enumerate(results[:5], 1):
            print(f"{rank:<5} {res['candidate_name']:<25} {res['faiss_similarity']:.4f}")
        print()

    if args.xgb_model and xgb_scorer:
        print("XGBoost")
        print("-" * 40)
        print(f"{'Rank':<5} {'Candidate':<25} {'Score':<10}")
        print("-" * 40)
        for rank, res in enumerate(results[:5], 1):
            print(f"{rank:<5} {res['candidate_name']:<25} {res.get('xgb_probability', 0.0):.4f}")
        print()
    else:
        print("CV MATCHING RESULTS (Baseline)")
        print("-" * 40)
        print(f"{'Rank':<5} {'Candidate':<25} {'Score':<10}")
        print("-" * 40)
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
            score = result.get("xgb_probability", result.get("final_score", 0.0))
            print(f"{rank:<5} {result['candidate_name']:<25} {score:.4f}")
        print()

    # --- Optional Qwen explanations for the top candidates --------------
    if args.qwen_model and results:
        _print_explanations(args.qwen_model, results, cvs, job, args.explanation_top_k)



def _print_explanations(
    model_path: str,
    results: list[dict[str, Any]],
    cvs: list[dict[str, Any]],
    job: dict[str, Any],
    explanation_top_k: int = 5,
) -> None:
    """Generate and print Qwen explanations for the top-ranked candidates.

    This is an optional, non-critical step.  If the model is unavailable or
    generation fails, the error is reported but the ranking output is not
    affected.
    """
    try:
        from src.llm import QwenLLM, build_evidence, build_matching_explanation_prompt
    except ImportError:
        logging.warning("llm module not available; skipping explanation generation.")
        return

    print("\n" + "=" * 50)
    print(f"QWEN EXPLANATIONS (Top {min(explanation_top_k, len(results))} Candidates)")
    print("=" * 50)

    try:
        qwen = QwenLLM(model_path=model_path)
    except Exception as error:
        print(f"\n[Qwen unavailable] {error}")
        return

    for result in results[:explanation_top_k]:
        candidate_name = result.get("candidate_name", "")
        rank = result.get("rank", "?")
        top_candidate = next(
            (cv for cv in cvs if cv.get("candidate_name") == candidate_name),
            cvs[0] if cvs else {},
        )

        evidence = build_evidence(
            candidate=top_candidate,
            job=job,
            match_result=result,
            rank=rank,
        )
        prompt = build_matching_explanation_prompt(evidence)

        print(f"\n--- #{rank} {candidate_name} ---")
        try:
            explanation = qwen.generate(prompt)
            print(explanation)
        except FileNotFoundError as error:
            print(f"[Explanation unavailable] {error}")
        except RuntimeError as error:
            print(f"[Explanation failed] {error}")
        except Exception as error:
            logging.exception("Unexpected error during Qwen explanation generation")
            print(f"[Explanation error] {error}")


if __name__ == "__main__":
    main()
