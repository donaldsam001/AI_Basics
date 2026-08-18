"""Evaluation utilities for the CV–Job matching pipeline.

Provides:

* **Retrieval metrics** – Recall@K for the FAISS retrieval stage.
* **Classification metrics** – Precision, Recall, F1, ROC-AUC, PR-AUC
  for XGBoost and the baseline scorer.
* **Ranking metrics** – Precision@K, Recall@K, NDCG@K, MRR for the
  final ranked list.
* **Baseline vs XGBoost comparison** – side-by-side metric tables.
"""

from .metrics import (
    classification_report,
    compare_baseline_vs_xgboost,
    ndcg_at_k,
    mrr,
    precision_at_k,
    ranking_report,
    recall_at_k,
    retrieval_recall_at_k,
    retrieval_report,
)

__all__ = [
    "classification_report",
    "compare_baseline_vs_xgboost",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "ranking_report",
    "recall_at_k",
    "retrieval_recall_at_k",
    "retrieval_report",
]
