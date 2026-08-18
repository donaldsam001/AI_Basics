"""Retrieval, classification, and ranking evaluation metrics.

All functions accept plain Python lists or NumPy arrays so they can be
used independently of the rest of the pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


# ======================================================================
# 1. Retrieval metrics — evaluate FAISS independently
# ======================================================================


def retrieval_recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
    k: int,
) -> float:
    """Recall@K: fraction of truly relevant items that appear in top-K.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of IDs returned by FAISS (best first).
    relevant_ids:
        Ground-truth set of relevant IDs.
    k:
        Cut-off depth.

    Returns
    -------
    float
        Recall in [0, 1].  Returns 0.0 when *relevant_ids* is empty.
    """
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    retrieved = set(list(retrieved_ids)[:k])
    return len(retrieved & relevant) / len(relevant)


def retrieval_report(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
    k_values: Sequence[int] = (5, 10, 20, 50, 100),
) -> dict[str, float]:
    """Compute Recall@K for multiple cut-offs in one call.

    Returns
    -------
    dict
        ``{"recall@5": 0.6, "recall@10": 0.8, ...}``
    """
    return {
        f"recall@{k}": round(retrieval_recall_at_k(retrieved_ids, relevant_ids, k), 4)
        for k in k_values
    }


# ======================================================================
# 2. Classification metrics — evaluate XGBoost / baseline predictions
# ======================================================================


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute core classification metrics for binary labels.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_pred:
        Predicted binary labels (0/1).
    y_prob:
        Predicted positive-class probabilities (for ROC-AUC and PR-AUC).

    Returns
    -------
    dict
        ``precision``, ``recall``, ``f1``, and optionally ``roc_auc``
        and ``pr_auc``.
    """
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=float)
        try:
            result["roc_auc"] = round(float(_roc_auc(true, prob)), 4)
        except ValueError:
            pass  # Single-class samples, AUC is undefined.
        try:
            result["pr_auc"] = round(float(_pr_auc(true, prob)), 4)
        except ValueError:
            pass
    return result


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC without importing sklearn."""
    if len(set(y_true)) < 2:
        raise ValueError("ROC-AUC requires both positive and negative samples")
    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    n_pos = int(np.sum(sorted_true == 1))
    n_neg = int(np.sum(sorted_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both positive and negative samples")
    tpr_sum = 0.0
    fp_count = 0
    for label in sorted_true:
        if label == 0:
            fp_count += 1
        else:
            tpr_sum += fp_count
    return 1.0 - tpr_sum / (n_pos * n_neg)


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Approximate PR-AUC using the trapezoidal rule."""
    if len(set(y_true)) < 2:
        raise ValueError("PR-AUC requires both positive and negative samples")
    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    n_pos = int(np.sum(sorted_true == 1))
    if n_pos == 0:
        raise ValueError("PR-AUC requires positive samples")
    precisions = []
    recalls = []
    tp = 0
    for i, label in enumerate(sorted_true, start=1):
        if label == 1:
            tp += 1
        precisions.append(tp / i)
        recalls.append(tp / n_pos)
    # Trapezoidal approximation.
    auc = 0.0
    for i in range(1, len(recalls)):
        auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0
    return auc


# ======================================================================
# 3. Ranking metrics — evaluate final ranked list
# ======================================================================


def precision_at_k(
    ranked_labels: Sequence[int],
    k: int,
) -> float:
    """Precision@K: fraction of the top-K that are relevant.

    Parameters
    ----------
    ranked_labels:
        Binary relevance labels in rank order (best first).
    k:
        Cut-off depth.
    """
    top = list(ranked_labels)[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(
    ranked_labels: Sequence[int],
    k: int,
    total_relevant: int | None = None,
) -> float:
    """Recall@K: fraction of all relevant items found in top-K.

    Parameters
    ----------
    ranked_labels:
        Binary relevance labels in rank order.
    k:
        Cut-off depth.
    total_relevant:
        Total relevant items in the collection.  Defaults to the sum of
        all labels in *ranked_labels*.
    """
    top = list(ranked_labels)[:k]
    total = total_relevant if total_relevant is not None else sum(ranked_labels)
    return sum(top) / total if total else 0.0


def ndcg_at_k(
    ranked_labels: Sequence[int | float],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Parameters
    ----------
    ranked_labels:
        Relevance scores (binary or graded) in rank order.
    k:
        Cut-off depth.
    """
    top = list(ranked_labels)[:k]
    dcg = sum(label / math.log2(i + 2) for i, label in enumerate(top))
    ideal = sorted(ranked_labels, reverse=True)[:k]
    idcg = sum(label / math.log2(i + 2) for i, label in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(
    ranked_labels: Sequence[int],
) -> float:
    """Mean Reciprocal Rank: 1 / position of the first relevant item.

    Returns 0.0 if no relevant item is found.
    """
    for i, label in enumerate(ranked_labels, start=1):
        if label:
            return 1.0 / i
    return 0.0


def ranking_report(
    ranked_labels: Sequence[int],
    k_values: Sequence[int] = (5, 10),
    total_relevant: int | None = None,
) -> dict[str, float]:
    """Compute a suite of ranking metrics at multiple K values.

    Returns
    -------
    dict
        Keys like ``precision@5``, ``recall@5``, ``ndcg@5``, ``mrr``.
    """
    result: dict[str, float] = {}
    for k in k_values:
        result[f"precision@{k}"] = round(precision_at_k(ranked_labels, k), 4)
        result[f"recall@{k}"] = round(recall_at_k(ranked_labels, k, total_relevant), 4)
        result[f"ndcg@{k}"] = round(ndcg_at_k(ranked_labels, k), 4)
    result["mrr"] = round(mrr(ranked_labels), 4)
    return result


# ======================================================================
# 4. Baseline vs XGBoost comparison
# ======================================================================


def compare_baseline_vs_xgboost(
    y_true: Sequence[int],
    baseline_scores: Sequence[float],
    xgboost_probs: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Side-by-side comparison of baseline deterministic and XGBoost.

    Both systems are evaluated at the same *threshold* to produce binary
    predictions.

    Returns
    -------
    dict
        ``{"baseline": {...}, "xgboost": {...}}`` each containing
        classification metrics.
    """
    baseline_pred = [1 if s >= threshold else 0 for s in baseline_scores]
    xgboost_pred = [1 if p >= threshold else 0 for p in xgboost_probs]
    return {
        "baseline": classification_report(y_true, baseline_pred, list(baseline_scores)),
        "xgboost": classification_report(y_true, xgboost_pred, list(xgboost_probs)),
    }
