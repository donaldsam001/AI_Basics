"""Tests for evaluation metrics (retrieval, classification, ranking)."""

from __future__ import annotations

import unittest

from src.evaluation.metrics import (
    classification_report,
    compare_baseline_vs_xgboost,
    mrr,
    ndcg_at_k,
    precision_at_k,
    ranking_report,
    recall_at_k,
    retrieval_recall_at_k,
    retrieval_report,
)


# -----------------------------------------------------------------------
# Retrieval metrics
# -----------------------------------------------------------------------

class TestRetrievalRecall(unittest.TestCase):

    def test_perfect_recall(self):
        self.assertEqual(retrieval_recall_at_k([1, 2, 3], [1, 2, 3], k=3), 1.0)

    def test_partial_recall(self):
        self.assertAlmostEqual(retrieval_recall_at_k([1, 2, 3, 4, 5], [1, 5], k=3), 0.5)

    def test_no_relevant_items(self):
        self.assertEqual(retrieval_recall_at_k([1, 2, 3], [], k=3), 0.0)

    def test_k_larger_than_retrieved(self):
        self.assertEqual(retrieval_recall_at_k([1, 2], [1, 2, 3], k=10), 2 / 3)

    def test_report_multiple_k(self):
        report = retrieval_report([1, 2, 3, 4, 5], [1, 3, 5], k_values=[2, 3, 5])
        self.assertIn("recall@2", report)
        self.assertIn("recall@3", report)
        self.assertIn("recall@5", report)
        self.assertEqual(report["recall@5"], 1.0)


# -----------------------------------------------------------------------
# Classification metrics
# -----------------------------------------------------------------------

class TestClassificationReport(unittest.TestCase):

    def test_perfect_classification(self):
        report = classification_report([1, 1, 0, 0], [1, 1, 0, 0])
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)

    def test_all_wrong(self):
        report = classification_report([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertEqual(report["precision"], 0.0)
        self.assertEqual(report["recall"], 0.0)

    def test_with_probabilities(self):
        report = classification_report(
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            y_prob=[0.9, 0.1, 0.8, 0.2],
        )
        self.assertIn("roc_auc", report)
        self.assertIn("pr_auc", report)
        self.assertGreater(report["roc_auc"], 0.5)

    def test_empty_inputs(self):
        report = classification_report([], [])
        self.assertEqual(report["f1"], 0.0)

    def test_single_class_no_auc(self):
        report = classification_report([1, 1], [1, 1], y_prob=[0.9, 0.8])
        # AUC should not be present when only one class exists.
        self.assertNotIn("roc_auc", report)


# -----------------------------------------------------------------------
# Ranking metrics
# -----------------------------------------------------------------------

class TestRankingMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        labels = [1, 0, 1, 0, 0]
        self.assertAlmostEqual(precision_at_k(labels, 2), 0.5)
        self.assertAlmostEqual(precision_at_k(labels, 3), 2 / 3)

    def test_recall_at_k(self):
        labels = [1, 0, 1, 0, 0]
        self.assertEqual(recall_at_k(labels, 2, total_relevant=2), 0.5)
        self.assertEqual(recall_at_k(labels, 3, total_relevant=2), 1.0)

    def test_ndcg_at_k_perfect(self):
        # Perfect ordering: all relevant first.
        self.assertAlmostEqual(ndcg_at_k([1, 1, 0, 0], 4), 1.0)

    def test_ndcg_at_k_imperfect(self):
        # Relevant items not at top.
        score = ndcg_at_k([0, 1, 1, 0], 4)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_mrr_first_position(self):
        self.assertEqual(mrr([1, 0, 0]), 1.0)

    def test_mrr_second_position(self):
        self.assertEqual(mrr([0, 1, 0]), 0.5)

    def test_mrr_no_relevant(self):
        self.assertEqual(mrr([0, 0, 0]), 0.0)

    def test_ranking_report(self):
        labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        report = ranking_report(labels, k_values=[5, 10])
        self.assertIn("precision@5", report)
        self.assertIn("recall@5", report)
        self.assertIn("ndcg@5", report)
        self.assertIn("precision@10", report)
        self.assertIn("mrr", report)
        self.assertEqual(report["mrr"], 1.0)


# -----------------------------------------------------------------------
# Baseline vs XGBoost comparison
# -----------------------------------------------------------------------

class TestBaselineVsXGBoost(unittest.TestCase):

    def test_comparison_returns_both_systems(self):
        comparison = compare_baseline_vs_xgboost(
            y_true=[1, 0, 1, 0],
            baseline_scores=[0.8, 0.3, 0.9, 0.2],
            xgboost_probs=[0.9, 0.1, 0.85, 0.15],
        )
        self.assertIn("baseline", comparison)
        self.assertIn("xgboost", comparison)
        self.assertIn("precision", comparison["baseline"])
        self.assertIn("precision", comparison["xgboost"])

    def test_xgboost_can_outperform_baseline(self):
        comparison = compare_baseline_vs_xgboost(
            y_true=[1, 0, 1, 0, 1, 0],
            baseline_scores=[0.6, 0.5, 0.4, 0.3, 0.7, 0.55],
            xgboost_probs=[0.9, 0.1, 0.85, 0.05, 0.95, 0.15],
            threshold=0.5,
        )
        # With these engineered scores, XGBoost should be better.
        self.assertGreaterEqual(
            comparison["xgboost"]["f1"],
            comparison["baseline"]["f1"],
        )


if __name__ == "__main__":
    unittest.main()
