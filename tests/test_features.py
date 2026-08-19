"""Feature engineering and supervised scorer tests."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.features import build_cv_jd_features, education_level, parse_skills
from src.models.feature_builder import PAIR_FEATURE_NAMES, build_pair_features
from src.models.xgboost_model import XGBCandidateScorer


class FeatureEngineeringTests(unittest.TestCase):
    def test_skill_parsing_normalizes_aliases_and_duplicates(self):
        self.assertEqual(parse_skills(" Python, K8s | postgres, python "), ["python", "kubernetes", "postgresql"])

    def test_education_encoding_is_ordinal_but_unknown_is_explicit(self):
        self.assertEqual(education_level("Bachelor of Science"), 2.0)
        self.assertEqual(education_level("Ph.D. in Computer Science"), 4.0)
        self.assertEqual(education_level("Professional certificate"), -1.0)

    def test_pair_features_handle_empty_job_requirements(self):
        features = build_cv_jd_features(
            {"skills": "Python", "years_of_experience": 2, "education": "Bachelor"},
            {"job_title": "Python Developer", "required_skills": [], "years_of_experience": 0, "education": []},
            semantic_score=84,
        )
        self.assertEqual(features["semantic_score"], 0.84)
        self.assertEqual(features["skill_match_ratio"], 1.0)
        self.assertEqual(features["experience_score"], 1.0)

    def test_canonical_pair_features_produce_all_names(self):
        """build_pair_features returns exactly PAIR_FEATURE_NAMES."""
        feat = build_pair_features(
            {"resume_skills": "Python, SQL", "experience_years": 3, "education_level": "Bachelors"},
            {"required_skills": "Python", "job_experience_required": 2},
            semantic_similarity=0.8,
            faiss_similarity=0.7,
        )
        self.assertEqual(set(feat.keys()), set(PAIR_FEATURE_NAMES))

    def test_scorer_uses_persisted_feature_order(self):
        class FakeModel:
            def predict_proba(self, matrix):
                self.matrix = matrix
                return np.column_stack((np.zeros(len(matrix)), np.full(len(matrix), 0.75)))

        model = FakeModel()
        scorer = XGBCandidateScorer(model, ["second", "first"])
        np.testing.assert_allclose(scorer.predict_probabilities([{"first": 1, "second": 2}]), [0.75])
        np.testing.assert_allclose(model.matrix, [[2.0, 1.0]])


def _write_elite_dataset(directory: str) -> Path:
    """Write a minimal elite-format CSV for integration testing."""
    fields = [
        "resume_id", "resume_text", "resume_skills", "experience_years",
        "education_level", "projects", "certifications", "job_role",
        "required_skills", "job_experience_required", "job_description",
        "skill_match_score", "experience_match", "education_match",
        "final_score", "shortlisted", "similarity_score",
    ]
    dataset = Path(directory) / "elite_mini.csv"
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(20):
            positive = index % 2
            writer.writerow({
                "resume_id": f"R{index:03d}",
                "resume_text": "Experienced Python developer" * (2 if positive else 1),
                "resume_skills": "Python, SQL, PyTorch" if positive else "Java, CSS",
                "experience_years": 6 if positive else 1,
                "education_level": "Bachelors" if positive else "High School",
                "projects": "ML project" if positive else "Website",
                "certifications": "AWS Certified" if positive else "",
                "job_role": "Software Engineer",
                "required_skills": "Python, SQL",
                "job_experience_required": 3,
                "job_description": "Build Python backends",
                "skill_match_score": 0.8 if positive else 0.2,
                "experience_match": 1.0 if positive else 0.0,
                "education_match": 1.0 if positive else 0.0,
                "final_score": 0.85 if positive else 0.15,
                "shortlisted": positive,
                "similarity_score": 0.75 if positive else 0.25,
            })
    return dataset


@unittest.skipUnless(__import__("importlib").util.find_spec("xgboost"), "xgboost is not installed")
class XGBoostTrainingTests(unittest.TestCase):
    """Tests for the refactored train_xgboost using canonical pair feature builder."""

    def test_training_persistence_loading_and_prediction(self):
        """Train with elite dataset, save pipeline, load, and infer with canonical features."""
        from src.models.train_xgboost import train_xgboost

        with TemporaryDirectory() as directory:
            dataset = _write_elite_dataset(directory)
            pipeline_path = Path(directory) / "pipeline.joblib"
            schema_path = Path(directory) / "schema.json"
            metadata_path = Path(directory) / "meta.json"
            importance_path = Path(directory) / "importance.csv"

            _, metrics = train_xgboost(
                dataset_path=dataset,
                pipeline_path=pipeline_path,
                schema_path=schema_path,
                metadata_path=metadata_path,
                importance_path=importance_path,
                model_options={"n_estimators": 2, "max_depth": 2},
            )
            self.assertIn("roc_auc", metrics)
            self.assertTrue(importance_path.is_file())

            # Load and infer using canonical feature builder.
            scorer = XGBCandidateScorer.load_pipeline(pipeline_path, schema_path)
            feat = build_pair_features(
                {"resume_skills": "Python", "experience_years": 3, "education_level": "Bachelors"},
                {"required_skills": "Python, SQL", "job_experience_required": 2},
            )
            prediction = scorer.predict_probabilities([feat])
            self.assertEqual(prediction.shape, (1,))
            self.assertTrue(0.0 <= float(prediction[0]) <= 1.0)

    def test_elite_dataset_training_and_preprocessing(self):
        from src.models.train_xgboost import train_xgboost

        with TemporaryDirectory() as directory:
            dataset = _write_elite_dataset(directory)
            pipeline_path = Path(directory) / "pipeline.joblib"
            schema_path = Path(directory) / "schema.json"
            metadata_path = Path(directory) / "meta.json"
            importance_path = Path(directory) / "importance.csv"
            _, metrics = train_xgboost(
                dataset_path=dataset,
                pipeline_path=pipeline_path,
                schema_path=schema_path,
                metadata_path=metadata_path,
                importance_path=importance_path,
                model_options={"n_estimators": 2, "max_depth": 2},
            )
            self.assertIn("roc_auc", metrics)
            self.assertIn("pr_auc", metrics)
            self.assertTrue(importance_path.is_file())
            self.assertTrue(pipeline_path.is_file())


if __name__ == "__main__":
    unittest.main()
