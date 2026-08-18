"""Qwen3-4B explanation layer for CV-job matching results.

This module provides an optional LLM-based explanation component that
interprets and explains XGBoost ranking results using evidence extracted
from the FAISS retrieval and feature engineering pipeline.

The LLM is an explanation-only layer: it does not rank candidates, modify
scores, or make hiring decisions.
"""

from .evidence import build_evidence
from .prompts import build_matching_explanation_prompt
from .qwen import QwenLLM

__all__ = [
    "QwenLLM",
    "build_evidence",
    "build_matching_explanation_prompt",
]
