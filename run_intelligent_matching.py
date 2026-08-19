"""Executable script for the 9-Step CV-Job Intelligent Matching System architecture flow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.intelligent_matching import CVJobIntelligentMatchingSystem, parse_resume, extract_document_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 9-Step CV-Job Intelligent Matching System Architecture")
    parser.add_argument("--cv", default="example_data/docx", help="Directory or single PDF/DOCX file containing resumes")
    parser.add_argument("--job", default="data/preprocess/job_roles_IT_filtered.csv", help="Job description file or text")
    parser.add_argument("--retrieval-k", type=int, default=50, help="FAISS Top-K retrieval count")
    parser.add_argument("--explain-k", type=int, default=5, help="Qwen3-4B explanation count")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    # Step 1 & 2: Read & Parse CV Inputs (PDF / DOCX / Text)
    cv_path = Path(args.cv)
    cvs = []
    if cv_path.is_dir():
        for file in sorted(cv_path.glob("*")):
            if file.suffix.lower() in (".pdf", ".docx", ".doc", ".txt") and not file.name.startswith("."):
                cvs.append(parse_resume(file))
    elif cv_path.is_file():
        cvs.append(parse_resume(cv_path))
    else:
        cvs.append(parse_resume(args.cv))

    if not cvs:
        print(f"No valid resumes found at: {args.cv}")
        return

    # Step 1 & 2: Job input
    job_path = Path(args.job)
    if job_path.is_file() and job_path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(job_path).fillna("")
        row = df.iloc[0].to_dict()
        job = {
            "job_title": row.get("job_title", "Software Engineer"),
            "job_description": row.get("job_description", row.get("summary", "")),
            "required_skills": [s.strip() for s in str(row.get("required_skills", "")).split(",") if s.strip()],
            "years_of_experience": row.get("years_of_experience", 0),
        }
    elif job_path.is_file():
        job_text = extract_document_text(job_path)
        job = {"job_title": job_path.stem.replace("_", " ").title(), "job_description": job_text}
    else:
        job = {"job_title": "Target Position", "job_description": str(args.job)}

    # Initialize 9-Step Pipeline
    system = CVJobIntelligentMatchingSystem()
    results = system.process_matching(
        candidate_files_or_dicts=cvs,
        job_description_or_dict=job,
        top_k_retrieval=args.retrieval_k,
        top_k_explain=args.explain_k,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    print("\n" + "=" * 70)
    print(f"CV-JOB INTELLIGENT MATCHING SYSTEM: RESULTS ({results['job_title']})")
    print("=" * 70)
    print(f"Total Candidates Processed: {results['total_candidates']}")
    print(f"Top-K FAISS Retrieved: {results['retrieved_candidates']}\n")

    print(f"{'Rank':<5} {'Candidate Name':<30} {'XGB Match Prob':<16} {'FAISS Sim':<12}")
    print("-" * 70)

    for res in results["rankings"]:
        rank = res.get("rank", "-")
        cand_name = res.get("candidate_name", "Unknown")[:28]
        xgb_prob = res.get("xgb_probability", res.get("final_score", 0.0))
        faiss_sim = res.get("faiss_similarity", 0.0)
        print(f"{rank:<5} {cand_name:<30} {xgb_prob:<16.4f} {faiss_sim:<12.4f}")

    print("\n" + "=" * 70)
    print("SKILL GAP ANALYSIS & EXPLANATIONS (Top Candidates)")
    print("=" * 70)

    for res in results["rankings"][:args.explain_k]:
        print(f"\n[Rank #{res.get('rank')}] {res.get('candidate_name')}")
        gap = res.get("skill_gap_analysis", {})
        print(f"  - Matched Skills ({gap.get('matched_count', 0)}): {', '.join(gap.get('matched_skills', [])) or 'None'}")
        print(f"  - Missing Skills (Gap Analysis - {gap.get('missing_count', 0)}): {', '.join(gap.get('missing_skills', [])) or 'None'}")
        print(f"  - Skill Match Score: {gap.get('skill_match_score', 0.0)}%")
        if res.get("qwen_explanation"):
            print(f"  - Qwen Explanation: {res['qwen_explanation']}")


if __name__ == "__main__":
    main()
