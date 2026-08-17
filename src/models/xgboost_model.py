"""Loading and inference helpers for the persisted XGBoost scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _xgboost_classifier() -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("XGBoost is required for supervised reranking. Install requirements.txt first.") from error
    return XGBClassifier


class XGBCandidateScorer:
    """Native-XGBoost model plus its explicit, ordered numeric feature schema."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        if not feature_names or any(not isinstance(name, str) or not name for name in feature_names):
            raise ValueError("XGBoost feature schema must contain non-empty feature names")
        self.model = model
        self.feature_names = feature_names

    @classmethod
    def load(cls, model_path: str | Path, features_path: str | Path | None = None) -> "XGBCandidateScorer":
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBoost model file does not exist: {model_path}")
        features_path = Path(features_path) if features_path else model_path.with_name("xgb_features.json")
        if not features_path.is_file():
            raise FileNotFoundError(f"XGBoost feature schema does not exist: {features_path}")
        try:
            schema = json.loads(features_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read XGBoost feature schema: {features_path}") from error
        feature_names = schema.get("feature_names") if isinstance(schema, dict) else None
        classifier = _xgboost_classifier()()
        classifier.load_model(str(model_path))
        return cls(classifier, feature_names)

    def predict_probabilities(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        if not feature_rows:
            return np.array([], dtype=float)
        matrix = np.asarray(
            [[float(row.get(name, 0.0)) for name in self.feature_names] for row in feature_rows], dtype=float,
        )
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)
