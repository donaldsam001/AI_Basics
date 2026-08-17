"""Feature engineering and supervised scorer tests."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.features import build_candidate_features, build_cv_jd_features, education_level, parse_skills
from src.models.xgboost_model import XGBCandidateScorer


class FeatureEngineeringTests(unittest.TestCase):
    def test_skill_parsing_normalizes_aliases_and_duplicates(self):
        self.assertEqual(parse_skills(" Python, K8s | postgres, python "), ["python", "kubernetes", "postgresql"])

    def test_education_encoding_is_ordinal_but_unknown_is_explicit(self):
        self.assertEqual(education_level("Bachelor of Science"), 2.0)
        self.assertEqual(education_level("Ph.D. in Computer Science"), 4.0)
        self.assertEqual(education_level("Professional certificate"), -1.0)

    def test_candidate_features_are_numeric_and_exclude_labels(self):
        features = build_candidate_features({
            "years_experience": 3, "has_portfolio": "yes", "skills": "Python, Python, SQL",
            "highest_degree": "Masters", "raw_text": "A short resume",
            "label": 1,
        })
        self.assertEqual(set(features), {"years_experience", "has_portfolio", "skill_count", "raw_text_length", "education_level"})
        self.assertEqual(features["skill_count"], 2.0)
        self.assertTrue(all(isinstance(value, float) for value in features.values()))

    def test_pair_features_handle_empty_job_requirements(self):
        features = build_cv_jd_features(
            {"skills": "Python", "years_of_experience": 2, "education": "Bachelor"},
            {"job_title": "Python Developer", "required_skills": [], "years_of_experience": 0, "education": []},
            semantic_score=84,
        )
        self.assertEqual(features["semantic_score"], 0.84)
        self.assertEqual(features["skill_match_ratio"], 1.0)
        self.assertEqual(features["experience_score"], 1.0)

    def test_scorer_uses_persisted_feature_order(self):
        class FakeModel:
            def predict_proba(self, matrix):
                self.matrix = matrix
                return np.column_stack((np.zeros(len(matrix)), np.full(len(matrix), 0.75)))

        model = FakeModel()
        scorer = XGBCandidateScorer(model, ["second", "first"])
        np.testing.assert_allclose(scorer.predict_probabilities([{"first": 1, "second": 2}]), [0.75])
        np.testing.assert_allclose(model.matrix, [[2.0, 1.0]])


@unittest.skipUnless(__import__("importlib").util.find_spec("xgboost"), "xgboost is not installed")
class XGBoostTrainingTests(unittest.TestCase):
    def test_training_persistence_loading_and_prediction(self):
        from src.models.train_xgboost import train_xgboost

        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.csv"
            with dataset.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "years_experience", "highest_degree", "skills", "current_title", "has_portfolio", "raw_text", "label",
                ])
                writer.writeheader()
                for index in range(20):
                    positive = index % 2
                    writer.writerow({
                        "years_experience": 8 if positive else 1,
                        "highest_degree": "Masters" if positive else "High School",
                        "skills": "Python, SQL" if positive else "Communication",
                        "current_title": "Engineer", "has_portfolio": bool(positive),
                        "raw_text": "technical project " * (8 if positive else 1), "label": positive,
                    })
            model_path = Path(directory) / "model.json"
            features_path = Path(directory) / "features.json"
            importance_path = Path(directory) / "importance.csv"
            _, metrics = train_xgboost(dataset, model_path, features_path, importance_path,
                                       model_options={"n_estimators": 2, "max_depth": 2})
            self.assertIn("roc_auc", metrics)
            self.assertTrue(importance_path.is_file())
            scorer = XGBCandidateScorer.load(model_path, features_path)
            prediction = scorer.predict_probabilities([build_candidate_features({"skills": "Python"})])
            self.assertEqual(prediction.shape, (1,))


if __name__ == "__main__":
    unittest.main()
