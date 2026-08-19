"""Regression tests for XGBoost feature contract.

Covers all 7 test cases specified in the architectural fix:

Test 1  — Training/inference feature equality
Test 2  — Feature ordering is deterministic
Test 3  — Missing feature raises explicit ValueError
Test 4  — Unexpected feature raises ValueError (schema drift detection)
Test 5  — Same input produces same feature vector (determinism)
Test 6  — Different jobs → different pair features (job-specificity)
Test 7  — End-to-end: FAISS → feature builder → XGBoost → ranking
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np

from src.models.feature_builder import (
    PAIR_FEATURE_NAMES,
    build_pair_features,
    feature_vector,
    validate_feature_row,
)
from src.models.xgboost_model import XGBCandidateScorer


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

_CV = {
    "resume_text": "Experienced Python backend developer with 5 years experience.",
    "resume_skills": "Python, FastAPI, PostgreSQL, Docker",
    "experience_years": 5,
    "education_level": "Bachelors",
    "certifications": "AWS Certified Developer",
    "has_portfolio": True,
}

_JOB_A = {
    "job_title": "Backend Engineer",
    "job_role": "Backend Developer",
    "required_skills": "Python, FastAPI, PostgreSQL",
    "job_experience_required": 3,
    "education_level": "Bachelors",
}

_JOB_B = {
    "job_title": "Data Scientist",
    "job_role": "Data Scientist",
    "required_skills": "Python, TensorFlow, Keras, pandas, scikit-learn",
    "job_experience_required": 4,
    "education_level": "Masters",
}


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Training/inference feature equality
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureNameEquality(unittest.TestCase):
    """Training feature names must equal inference feature names."""

    def test_build_pair_features_produces_pair_feature_names(self):
        """Inference feature builder produces exactly PAIR_FEATURE_NAMES."""
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.82, faiss_similarity=0.75)
        self.assertEqual(set(feat.keys()), set(PAIR_FEATURE_NAMES),
                         "build_pair_features keys must equal PAIR_FEATURE_NAMES")

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("xgboost"),
        "xgboost not installed"
    )
    def test_training_feature_names_match_inference_feature_names(self):
        """Train the model; the saved schema must use exactly PAIR_FEATURE_NAMES."""
        from src.models.train_xgboost import train_xgboost

        with TemporaryDirectory() as tmpdir:
            dataset = _write_mini_elite_dataset(tmpdir)
            pipeline_path = Path(tmpdir) / "pipeline.joblib"
            schema_path = Path(tmpdir) / "schema.json"
            metadata_path = Path(tmpdir) / "meta.json"
            importance_path = Path(tmpdir) / "imp.csv"

            train_xgboost(
                dataset_path=dataset,
                pipeline_path=pipeline_path,
                schema_path=schema_path,
                metadata_path=metadata_path,
                importance_path=importance_path,
                model_options={"n_estimators": 2, "max_depth": 2},
            )

            schema = json.loads(schema_path.read_text())
            self.assertEqual(
                schema["feature_names"], PAIR_FEATURE_NAMES,
                "Saved schema feature_names must equal PAIR_FEATURE_NAMES"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Feature ordering is deterministic
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureOrdering(unittest.TestCase):
    """PAIR_FEATURE_NAMES order must be stable and consistent."""

    def test_no_duplicates_in_pair_feature_names(self):
        self.assertEqual(len(PAIR_FEATURE_NAMES), len(set(PAIR_FEATURE_NAMES)))

    def test_feature_vector_order_matches_pair_feature_names(self):
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.7, faiss_similarity=0.6)
        vec = feature_vector(feat)
        self.assertEqual(len(vec), len(PAIR_FEATURE_NAMES))
        for i, name in enumerate(PAIR_FEATURE_NAMES):
            self.assertAlmostEqual(vec[i], feat[name], places=9,
                                   msg=f"Position {i} ({name}) mismatch")

    def test_scorer_respects_canonical_order(self):
        """Scorer must consume features in PAIR_FEATURE_NAMES order."""
        class RecordingModel:
            def predict_proba(self, matrix):
                self.last_matrix = matrix
                return np.column_stack([np.zeros(len(matrix)), np.ones(len(matrix)) * 0.9])

        model = RecordingModel()
        scorer = XGBCandidateScorer(model, PAIR_FEATURE_NAMES)
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.8, faiss_similarity=0.7)
        scorer.predict_probabilities([feat])
        expected_vec = [feat[n] for n in PAIR_FEATURE_NAMES]
        np.testing.assert_array_almost_equal(model.last_matrix[0], expected_vec)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Missing feature raises explicit ValueError
# ──────────────────────────────────────────────────────────────────────────────

class TestMissingFeatureRaisesError(unittest.TestCase):
    """Inference must raise ValueError when required features are absent."""

    def test_validate_feature_row_raises_on_missing(self):
        incomplete = {"semantic_similarity": 0.8}  # almost all features missing
        with self.assertRaises(ValueError) as ctx:
            validate_feature_row(incomplete)
        self.assertIn("Missing required XGBoost features", str(ctx.exception))
        # Check that at least one missing feature name is in the error message.
        self.assertIn("required_skill_score", str(ctx.exception))

    def test_scorer_predict_raises_on_missing_feature(self):
        class DummyModel:
            def predict_proba(self, matrix):
                return np.column_stack([np.zeros(len(matrix)), np.ones(len(matrix))])

        scorer = XGBCandidateScorer(DummyModel(), PAIR_FEATURE_NAMES)
        bad_row = {"semantic_similarity": 0.5}   # missing all other features
        with self.assertRaises(ValueError) as ctx:
            scorer.predict_probabilities([bad_row])
        self.assertIn("missing required", str(ctx.exception).lower())

    def test_error_message_lists_specific_missing_features(self):
        feat = build_pair_features(_CV, _JOB_A)
        feat_copy = dict(feat)
        del feat_copy["faiss_similarity"]
        del feat_copy["required_skill_score"]
        with self.assertRaises(ValueError) as ctx:
            validate_feature_row(feat_copy)
        msg = str(ctx.exception)
        self.assertIn("faiss_similarity", msg)
        self.assertIn("required_skill_score", msg)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Unexpected feature raises ValueError (schema drift detection)
# ──────────────────────────────────────────────────────────────────────────────

class TestUnexpectedFeatureRaisesError(unittest.TestCase):
    """Inference must raise ValueError when spurious features appear."""

    def test_validate_feature_row_raises_on_extra_feature(self):
        feat = build_pair_features(_CV, _JOB_A)
        feat["this_feature_was_never_trained"] = 99.0
        with self.assertRaises(ValueError) as ctx:
            validate_feature_row(feat)
        self.assertIn("Unexpected XGBoost features", str(ctx.exception))
        self.assertIn("this_feature_was_never_trained", str(ctx.exception))

    def test_allow_extra_flag_suppresses_extra_check(self):
        feat = build_pair_features(_CV, _JOB_A)
        feat["extra_debug_feature"] = 1.0
        # Should not raise when allow_extra=True.
        try:
            validate_feature_row(feat, allow_extra=True)
        except ValueError as exc:
            self.fail(f"validate_feature_row raised unexpectedly: {exc}")

    def test_scorer_raises_on_extra_feature(self):
        class DummyModel:
            def predict_proba(self, matrix):
                return np.column_stack([np.zeros(len(matrix)), np.ones(len(matrix))])

        scorer = XGBCandidateScorer(DummyModel(), PAIR_FEATURE_NAMES)
        feat = build_pair_features(_CV, _JOB_A)
        feat["rogue_feature"] = 42.0
        with self.assertRaises(ValueError) as ctx:
            scorer.predict_probabilities([feat])
        self.assertIn("unexpected features", str(ctx.exception).lower())


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Same input → same feature vector (determinism)
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureDeterminism(unittest.TestCase):
    """Identical inputs must produce identical feature vectors."""

    def test_same_cv_same_job_same_scores_same_vector(self):
        feat_a = build_pair_features(_CV, _JOB_A, semantic_similarity=0.75, faiss_similarity=0.70)
        feat_b = build_pair_features(_CV, _JOB_A, semantic_similarity=0.75, faiss_similarity=0.70)
        self.assertEqual(feat_a, feat_b)

    def test_feature_vector_function_is_deterministic(self):
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.6)
        vec_1 = feature_vector(feat)
        vec_2 = feature_vector(feat)
        self.assertEqual(vec_1, vec_2)

    def test_scorer_determinism(self):
        class ConstantModel:
            def predict_proba(self, matrix):
                return np.column_stack([np.zeros(len(matrix)), np.full(len(matrix), 0.85)])

        scorer = XGBCandidateScorer(ConstantModel(), PAIR_FEATURE_NAMES)
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.8)
        prob_a = scorer.predict_probabilities([feat])
        prob_b = scorer.predict_probabilities([feat])
        np.testing.assert_array_equal(prob_a, prob_b)


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — Different job → different pair features (job-specificity)
# ──────────────────────────────────────────────────────────────────────────────

class TestJobSpecificFeatures(unittest.TestCase):
    """Same CV + different jobs must produce different feature vectors.

    This is the CRITICAL test.  If the features are identical regardless of the
    job, XGBoost predictions are not job-specific and the reranking is invalid.
    """

    def test_different_jobs_produce_different_feature_vectors(self):
        feat_a = build_pair_features(_CV, _JOB_A, semantic_similarity=0.8, faiss_similarity=0.7)
        feat_b = build_pair_features(_CV, _JOB_B, semantic_similarity=0.5, faiss_similarity=0.4)
        self.assertNotEqual(feat_a, feat_b,
                             "Same CV + different jobs MUST produce different features")

    def test_required_skill_score_differs_by_job(self):
        feat_backend = build_pair_features(_CV, _JOB_A)
        feat_ds = build_pair_features(_CV, _JOB_B)
        # CV has Python/FastAPI/PostgreSQL/Docker — good match for Backend, poor for Data Scientist.
        self.assertGreater(
            feat_backend["required_skill_score"],
            feat_ds["required_skill_score"],
            "Backend job should score higher required_skill_score for this CV"
        )

    def test_experience_gap_differs_by_job(self):
        feat_a = build_pair_features(_CV, _JOB_A)  # requires 3 years
        feat_b = build_pair_features(_CV, _JOB_B)  # requires 4 years
        # CV has 5 years: gap_A = 2, gap_B = 1
        self.assertAlmostEqual(feat_a["experience_gap"], 2.0, places=5)
        self.assertAlmostEqual(feat_b["experience_gap"], 1.0, places=5)

    def test_scorer_gives_different_probabilities_for_different_jobs(self):
        """XGBoost must NOT return the same probability for every job."""
        class LinearModel:
            """Returns probability = sum(row) / (sum(row) + 1) — varies with input."""
            def predict_proba(self, matrix):
                probs = np.sum(matrix, axis=1)
                probs = probs / (probs + 1.0)
                return np.column_stack([1 - probs, probs])

        scorer = XGBCandidateScorer(LinearModel(), PAIR_FEATURE_NAMES)
        feat_a = build_pair_features(_CV, _JOB_A, semantic_similarity=0.8, faiss_similarity=0.7)
        feat_b = build_pair_features(_CV, _JOB_B, semantic_similarity=0.5, faiss_similarity=0.4)
        prob_a = scorer.predict_probabilities([feat_a])[0]
        prob_b = scorer.predict_probabilities([feat_b])[0]
        self.assertNotAlmostEqual(float(prob_a), float(prob_b), places=5,
                                  msg="Probabilities must differ for different jobs")


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — End-to-end: FAISS → feature builder → XGBoost → ranking
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipeline(unittest.TestCase):
    """Integration test: FAISS retrieval → build_pair_features → XGBoost."""

    def _make_scorer_prefer_backend(self):
        """Scorer that returns high probability iff required_skill_score > 0.5."""
        class SkillBasedModel:
            def predict_proba(self, matrix):
                # required_skill_score is feature index 2 (PAIR_FEATURE_NAMES[2]).
                idx = PAIR_FEATURE_NAMES.index("required_skill_score")
                scores = matrix[:, idx]
                probs = np.where(scores > 0.5, 0.9, 0.2)
                return np.column_stack([1 - probs, probs])

        return XGBCandidateScorer(SkillBasedModel(), PAIR_FEATURE_NAMES)

    def test_end_to_end_ranking_is_job_specific(self):
        """Verify the pipeline: match → build_pair_features → XGBoost → sort."""
        from src.matching.matcher import CVJobMatcher

        class FakeEmbedding:
            def encode(self, texts, show_progress_bar=True):
                return np.array([[1.0, 0.0] for _ in texts])

        cvs = [
            {
                "candidate_name": "Backend Alice",
                "resume_skills": "Python, FastAPI, PostgreSQL",
                "experience_years": 5,
                "education_level": "Bachelors",
            },
            {
                "candidate_name": "DS Bob",
                "resume_skills": "TensorFlow, Keras, pandas",
                "experience_years": 4,
                "education_level": "Masters",
            },
        ]
        job = {
            "job_title": "Backend Engineer",
            "required_skills": "Python, FastAPI, PostgreSQL",
            "job_experience_required": 3,
        }

        from src.vector_store import FAISSStore, MetadataStore
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.json"
            meta_path.write_text(json.dumps([
                {"faiss_id": 0, "source_index": 0},
                {"faiss_id": 1, "source_index": 1},
            ]))
            store = FAISSStore(2)
            store.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
            meta_store = MetadataStore(str(meta_path))

            matcher = CVJobMatcher(FakeEmbedding())
            scorer = self._make_scorer_prefer_backend()
            results = matcher.rank_retrieved_candidates(
                cvs, job, store, meta_store, k=2, xgb_scorer=scorer
            )

        self.assertEqual(len(results), 2)
        # Backend Alice should rank first because she has Python/FastAPI/PostgreSQL.
        self.assertEqual(results[0]["candidate_name"], "Backend Alice")
        self.assertGreater(results[0]["xgb_probability"], results[1]["xgb_probability"])
        # Verify all results have required fields.
        for r in results:
            self.assertIn("xgb_probability", r)
            self.assertIn("deterministic_score", r)
            self.assertIn("faiss_similarity", r)

    def test_pipeline_produces_valid_probabilities(self):
        """All XGBoost probabilities must be in [0, 1]."""
        class BoundedModel:
            def predict_proba(self, matrix):
                # Output deliberately in [0, 1].
                probs = np.clip(np.abs(np.sin(np.sum(matrix, axis=1))), 0, 1)
                return np.column_stack([1 - probs, probs])

        scorer = XGBCandidateScorer(BoundedModel(), PAIR_FEATURE_NAMES)
        cvs_list = [_CV, dict(_CV, experience_years=1, resume_skills="Java")]
        job = _JOB_A
        rows = [
            build_pair_features(cv, job, semantic_similarity=0.7, faiss_similarity=0.6)
            for cv in cvs_list
        ]
        probs = scorer.predict_probabilities(rows)
        self.assertTrue(np.all(probs >= 0.0), "All probabilities must be >= 0")
        self.assertTrue(np.all(probs <= 1.0), "All probabilities must be <= 1")


# ──────────────────────────────────────────────────────────────────────────────
# Bonus: feature content tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureValues(unittest.TestCase):
    """Spot-check specific feature values for correctness."""

    def test_semantic_similarity_clamped_to_unit_interval(self):
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=1.5)
        self.assertEqual(feat["semantic_similarity"], 1.0)
        feat2 = build_pair_features(_CV, _JOB_A, semantic_similarity=-0.3)
        self.assertEqual(feat2["semantic_similarity"], 0.0)

    def test_experience_gap_is_signed(self):
        cv_junior = dict(_CV, experience_years=1)
        feat = build_pair_features(cv_junior, _JOB_A)  # requires 3 years
        self.assertAlmostEqual(feat["experience_gap"], -2.0, places=5)
        self.assertEqual(feat["experience_satisfies"], 0.0)

    def test_experience_satisfies_when_overskilled(self):
        feat = build_pair_features(_CV, _JOB_A)  # 5 years vs 3 required
        self.assertEqual(feat["experience_satisfies"], 1.0)
        self.assertAlmostEqual(feat["experience_gap"], 2.0, places=5)

    def test_certification_parsing(self):
        feat = build_pair_features(_CV, _JOB_A)
        self.assertEqual(feat["has_certification"], 1.0)
        self.assertGreaterEqual(feat["certification_count"], 1.0)

    def test_no_certification(self):
        cv_no_cert = dict(_CV, certifications=None)
        feat = build_pair_features(cv_no_cert, _JOB_A)
        self.assertEqual(feat["has_certification"], 0.0)
        self.assertEqual(feat["certification_count"], 0.0)

    def test_all_feature_values_are_float(self):
        feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.8, faiss_similarity=0.6)
        for name, value in feat.items():
            self.assertIsInstance(value, float, f"Feature {name!r} is not float")


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end training integration
# ──────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    __import__("importlib").util.find_spec("xgboost"),
    "xgboost not installed"
)
class TestTrainingIntegration(unittest.TestCase):
    """Verify the full retrain → save → load → infer round trip."""

    def test_retrain_save_load_infer_round_trip(self):
        from src.models.train_xgboost import train_xgboost

        with TemporaryDirectory() as tmpdir:
            dataset = _write_mini_elite_dataset(tmpdir)
            pipeline_path = Path(tmpdir) / "pipeline.joblib"
            schema_path = Path(tmpdir) / "schema.json"
            meta_path = Path(tmpdir) / "meta.json"
            imp_path = Path(tmpdir) / "imp.csv"

            _, metrics = train_xgboost(
                dataset_path=dataset,
                pipeline_path=pipeline_path,
                schema_path=schema_path,
                metadata_path=meta_path,
                importance_path=imp_path,
                model_options={"n_estimators": 3, "max_depth": 2},
            )

            self.assertIn("roc_auc", metrics)
            self.assertTrue(pipeline_path.is_file())
            self.assertTrue(schema_path.is_file())

            # Load and infer — must use canonical features.
            scorer = XGBCandidateScorer.load_pipeline(pipeline_path, schema_path)
            feat = build_pair_features(_CV, _JOB_A, semantic_similarity=0.8, faiss_similarity=0.7)
            probs = scorer.predict_probabilities([feat])
            self.assertEqual(probs.shape, (1,))
            self.assertTrue(0.0 <= float(probs[0]) <= 1.0)

    def test_schema_version_is_persisted(self):
        from src.models.train_xgboost import train_xgboost, SCHEMA_VERSION

        with TemporaryDirectory() as tmpdir:
            dataset = _write_mini_elite_dataset(tmpdir)
            schema_path = Path(tmpdir) / "schema.json"
            train_xgboost(
                dataset_path=dataset,
                schema_path=schema_path,
                pipeline_path=Path(tmpdir) / "p.joblib",
                metadata_path=Path(tmpdir) / "m.json",
                importance_path=Path(tmpdir) / "i.csv",
                model_options={"n_estimators": 2, "max_depth": 2},
            )
            schema = json.loads(schema_path.read_text())
            self.assertEqual(schema["schema_version"], SCHEMA_VERSION)
            self.assertIn("label_provenance", schema)
            self.assertIn("ALGORITHMIC", schema["label_provenance"])

    def test_label_provenance_documented_in_metadata(self):
        from src.models.train_xgboost import train_xgboost

        with TemporaryDirectory() as tmpdir:
            dataset = _write_mini_elite_dataset(tmpdir)
            meta_path = Path(tmpdir) / "meta.json"
            train_xgboost(
                dataset_path=dataset,
                metadata_path=meta_path,
                pipeline_path=Path(tmpdir) / "p.joblib",
                schema_path=Path(tmpdir) / "s.json",
                importance_path=Path(tmpdir) / "i.csv",
                model_options={"n_estimators": 2, "max_depth": 2},
            )
            metadata = json.loads(meta_path.read_text())
            self.assertIn("label_provenance", metadata)
            self.assertIn("ALGORITHMIC", metadata["label_provenance"])
            self.assertIn("split_strategy", metadata)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: write a minimal elite-format CSV for testing
# ──────────────────────────────────────────────────────────────────────────────

def _write_mini_elite_dataset(directory: str) -> Path:
    """Write a small synthetic elite-format CSV for integration tests."""
    fields = [
        "resume_id", "resume_text", "resume_skills", "experience_years",
        "education_level", "projects", "certifications", "job_role",
        "required_skills", "job_experience_required", "job_description",
        "skill_match_score", "experience_match", "education_match",
        "final_score", "shortlisted", "similarity_score",
    ]
    rows = []
    for i in range(30):
        positive = i % 2
        rows.append({
            "resume_id": f"R{i:03d}",
            "resume_text": "Experienced Python and ML developer building algorithms" * (2 if positive else 1),
            "resume_skills": "Python, SQL, PyTorch" if positive else "Java, CSS",
            "experience_years": 6 if positive else 1,
            "education_level": "Bachelors" if positive else "High School",
            "projects": "Neural network optimization" if positive else "Website",
            "certifications": "AWS Certified" if positive else "",
            "job_role": ["Software Engineer", "Data Scientist", "Backend Developer"][i % 3],
            "required_skills": "Python, SQL" if positive else "Java, SQL",
            "job_experience_required": 3,
            "job_description": "Build high performance Python backends and ML models",
            "skill_match_score": 0.8 if positive else 0.2,
            "experience_match": 1.0 if positive else 0.0,
            "education_match": 1.0 if positive else 0.0,
            "final_score": 0.85 if positive else 0.15,
            "shortlisted": positive,
            "similarity_score": 0.75 if positive else 0.25,
        })

    dataset = Path(directory) / "elite_mini.csv"
    with dataset.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return dataset


if __name__ == "__main__":
    unittest.main()
