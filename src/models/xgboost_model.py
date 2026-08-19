"""Loading and inference helpers for the persisted XGBoost pipeline.

Strict feature contract
-----------------------
Inference MUST use the exact same features as training.
Any mismatch raises ``ValueError`` — there is NO silent zero-filling.

Usage
-----
The scorer can load from two artifact formats:

1. **Preferred** — joblib Pipeline saved by the new training script:
   ``XGBCandidateScorer.load_pipeline("data/models/xgb_pipeline.joblib",
                                       "data/models/xgb_schema.json")``

2. **Legacy** — raw XGBoost JSON + feature schema (for backward compat):
   ``XGBCandidateScorer.load("data/models/cv_job_xgb.json",
                              "data/models/xgb_features.json")``

In both cases the public ``predict_probabilities`` method:
  1. Calls ``validate_feature_row`` — raises if any required feature is missing.
  2. Assembles the matrix in the exact order defined by ``PAIR_FEATURE_NAMES``.
  3. Asserts returned probabilities are in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.models.feature_builder import PAIR_FEATURE_NAMES, validate_feature_row


def _xgboost_classifier() -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is required for supervised reranking. "
            "Install requirements.txt first."
        ) from error
    return XGBClassifier


class XGBCandidateScorer:
    """XGBoost model plus strict, ordered feature schema enforcement.

    The feature names stored in this instance are the authoritative contract.
    Any call to ``predict_probabilities`` that supplies a feature dict not
    matching ``self.feature_names`` raises ``ValueError`` immediately.
    """

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        if not feature_names or any(
            not isinstance(name, str) or not name for name in feature_names
        ):
            raise ValueError(
                "XGBoost feature schema must contain non-empty feature names"
            )
        self.model = model
        self.feature_names = list(feature_names)
        # Keep a set for O(1) membership checks.
        self._feature_set = set(feature_names)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def load_pipeline(
        cls,
        pipeline_path: str | Path,
        schema_path: str | Path | None = None,
    ) -> "XGBCandidateScorer":
        """Load from a joblib Pipeline artifact (preferred).

        Parameters
        ----------
        pipeline_path:
            Path to ``xgb_pipeline.joblib``.
        schema_path:
            Optional path to ``xgb_schema.json``.  If omitted, defaults to
            ``<pipeline_dir>/xgb_schema.json``.
        """
        try:
            import joblib  # noqa: PLC0415
        except ImportError as error:
            raise RuntimeError("joblib is required. Install requirements.txt.") from error

        pipeline_path = Path(pipeline_path)
        if not pipeline_path.is_file():
            raise FileNotFoundError(
                f"XGBoost pipeline file does not exist: {pipeline_path}"
            )
        schema_path = (
            Path(schema_path)
            if schema_path
            else pipeline_path.parent / "xgb_schema.json"
        )

        feature_names = cls._load_feature_names(schema_path)
        pipeline = joblib.load(pipeline_path)
        return cls(pipeline, feature_names)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        features_path: str | Path | None = None,
    ) -> "XGBCandidateScorer":
        """Load from a raw XGBoost JSON model + feature schema (legacy).

        Parameters
        ----------
        model_path:
            Path to ``cv_job_xgb.json``.
        features_path:
            Optional path to ``xgb_features.json``.  Defaults to
            ``<model_dir>/xgb_features.json``.
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"XGBoost model file does not exist: {model_path}"
            )
        features_path = (
            Path(features_path)
            if features_path
            else model_path.with_name("xgb_features.json")
        )

        feature_names = cls._load_feature_names(features_path)
        classifier = _xgboost_classifier()()
        classifier.load_model(str(model_path))
        return cls(classifier, feature_names)

    @staticmethod
    def _load_feature_names(features_path: Path) -> list[str]:
        if not features_path.is_file():
            raise FileNotFoundError(
                f"XGBoost feature schema does not exist: {features_path}"
            )
        try:
            schema = json.loads(features_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Unable to read XGBoost feature schema: {features_path}"
            ) from error
        feature_names = schema.get("feature_names") if isinstance(schema, dict) else None
        if not feature_names:
            raise ValueError(
                f"Feature schema at {features_path} has no 'feature_names' key"
            )
        return list(feature_names)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_probabilities(
        self, feature_rows: list[dict[str, float]]
    ) -> np.ndarray:
        """Return P(match=1) for each feature row.

        Parameters
        ----------
        feature_rows:
            List of feature dicts produced by ``build_pair_features``.
            Every dict must contain all keys in ``self.feature_names``
            with no extras (strict mode).

        Raises
        ------
        ValueError
            Immediately if any row is missing a required feature or contains
            an unexpected feature.  No silent zero-filling ever occurs.
        """
        if not feature_rows:
            return np.array([], dtype=float)

        # Validate all rows before building the matrix.
        for i, row in enumerate(feature_rows):
            missing = set(self.feature_names) - set(row)
            if missing:
                raise ValueError(
                    f"Feature row {i} is missing required XGBoost features: "
                    f"{sorted(missing)}\n"
                    f"Expected: {self.feature_names}"
                )
            extra = set(row) - self._feature_set
            if extra:
                raise ValueError(
                    f"Feature row {i} contains unexpected features "
                    f"(schema drift): {sorted(extra)}"
                )

        # Build matrix in canonical order — explicit, never .get() with default.
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_names] for row in feature_rows],
            dtype=float,
        )
        probabilities = np.asarray(
            self.model.predict_proba(matrix)[:, 1], dtype=float
        )

        # Sanity check on probability range.
        if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
            bad = probabilities[(probabilities < 0.0) | (probabilities > 1.0)]
            raise RuntimeError(
                f"predict_proba returned {len(bad)} value(s) outside [0, 1]: {bad}"
            )

        return probabilities
