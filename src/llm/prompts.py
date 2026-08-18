"""Prompt templates for the Qwen3-4B explanation layer.

The prompt builder converts a structured evidence dictionary into a
precisely formatted prompt that instructs Qwen to explain an XGBoost
ranking result using only the supplied evidence.

The prompt enforces several anti-hallucination and role-boundary constraints:

- Qwen is told that XGBoost produced the ranking.
- Qwen must only explain the result, not modify it.
- Qwen must not invent information.
- Qwen must not make hiring decisions.
- Missing requirements must be explicitly identified.
"""

from __future__ import annotations

from typing import Any


def _bullet_list(items: list[str], fallback: str = "None") -> str:
    """Format a list of items as a newline-separated bullet list."""
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)


def build_matching_explanation_prompt(evidence: dict[str, Any]) -> str:
    """Build a structured explanation prompt from an evidence dictionary.

    Parameters
    ----------
    evidence:
        A dictionary with ``candidate``, ``job``, and ``model_result``
        keys as produced by :func:`src.llm.evidence.build_evidence`.

    Returns
    -------
    str
        A fully formatted prompt ready for :meth:`QwenLLM.generate`.
    """
    candidate = evidence.get("candidate", {})
    job = evidence.get("job", {})
    model = evidence.get("model_result", {})

    # --- Preamble with strict role boundaries ---------------------------
    lines = [
        "You are an AI assistant explaining CV-job matching.",
        "",
        "The candidate ranking was calculated by an XGBoost model.",
        "Your task is ONLY to explain the model result.",
        "",
        "Do not decide whether the candidate should be hired.",
        "Do not change the ranking.",
        "Do not invent information.",
        "Do not invent skills, experience, education, projects, "
        "certifications, technologies, achievements, or responsibilities.",
        "If information is unavailable, state that it is not provided.",
        "Do not infer unsupported experience.",
        "Do not change the XGBoost ranking or probability.",
        "Do not make a hiring decision.",
        "",
        "---",
        "",
    ]

    # --- Candidate section ----------------------------------------------
    lines.append(f"Candidate:\n{candidate.get('name', 'Unknown')}")
    lines.append("")

    # --- Job section ----------------------------------------------------
    lines.append(f"Job:\n{job.get('title', 'Untitled Job')}")
    lines.append("")

    # --- Model result ---------------------------------------------------
    lines.append("Model result:")
    lines.append(f"Rank: #{model.get('rank', 'N/A')}")
    lines.append(f"Match probability: {model.get('xgboost_probability', 'N/A')}")
    if "faiss_similarity" in model:
        lines.append(f"FAISS similarity: {model['faiss_similarity']}")
    lines.append("")

    # --- Matched required skills ----------------------------------------
    lines.append("Matched required skills:")
    lines.append(_bullet_list(candidate.get("matched_required_skills", [])))
    lines.append("")

    # --- Missing required skills ----------------------------------------
    lines.append("Missing required skills:")
    lines.append(_bullet_list(candidate.get("missing_required_skills", [])))
    lines.append("")

    # --- Matched preferred skills (when present) -----------------------
    preferred = candidate.get("matched_preferred_skills", [])
    if preferred:
        lines.append("Matched preferred skills:")
        lines.append(_bullet_list(preferred))
        lines.append("")

    # --- Experience -----------------------------------------------------
    lines.append(f"Experience:\n{candidate.get('experience', 'Not specified')}")
    lines.append("")

    # --- Job experience requirement ------------------------------------
    lines.append(f"Job experience requirement:\n{job.get('experience_requirement', 'Not specified')}")
    lines.append("")

    # --- Projects -------------------------------------------------------
    lines.append("Relevant projects:")
    lines.append(_bullet_list(candidate.get("projects", [])))
    lines.append("")

    # --- Education ------------------------------------------------------
    lines.append("Education:")
    lines.append(_bullet_list(candidate.get("education", [])))
    lines.append("")

    # --- Responsibilities (from job) ------------------------------------
    responsibilities = job.get("responsibilities", [])
    if responsibilities:
        lines.append("Job responsibilities:")
        lines.append(_bullet_list(responsibilities))
        lines.append("")

    # --- Explanation instructions ---------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("Explain:")
    lines.append("")
    lines.append("1. Why the candidate received this ranking.")
    lines.append("2. Which job requirements are satisfied.")
    lines.append("3. Which requirements are missing.")
    lines.append("4. What areas the candidate could improve.")
    lines.append("")
    lines.append("Use only the supplied evidence.")

    return "\n".join(lines)
