"""Train the CV-level XGBoost suitability model without label leakage.

The bundled dataset's ``label`` means a CV is suitable (1) or unsuitable (0)
in its source dataset.  It has no job-description identifier or pair label, so
this command deliberately trains only CV-level features.  It must not be read
as a calibrated CV--JD suitability model until pair-labelled data is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import CANDIDATE_FEATURE_NAMES, build_candidate_features
from src.models.xgboost_model import _xgboost_classifier


DEFAULT_MODEL_PATH = Path("data/models/cv_job_xgb.json")
DEFAULT_FEATURES_PATH = Path("data/models/xgb_features.json")
DEFAULT_IMPORTANCE_PATH = Path("data/models/xgb_feature_importance.csv")


def build_training_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ordered numeric features and never include the target column."""
    rows = [build_candidate_features(record) for record in frame.to_dict("records")]
    return pd.DataFrame(rows, columns=CANDIDATE_FEATURE_NAMES, dtype=float)


def train_xgboost(
    dataset_path: str | Path, output_path: str | Path = DEFAULT_MODEL_PATH,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    importance_path: str | Path = DEFAULT_IMPORTANCE_PATH, *, test_size: float = 0.2,
    random_state: int = 42, model_options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    frame = pd.read_csv(dataset_path)
    required = {"label", "years_experience", "highest_degree", "skills", "current_title", "has_portfolio", "raw_text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {', '.join(missing)}")
    labels = frame["label"]
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("label must contain only binary 0 and 1 values")
    features = build_training_features(frame)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels.astype(int), test_size=test_size, stratify=labels, random_state=random_state,
    )
    positive = int((train_y == 1).sum())
    negative = int((train_y == 0).sum())
    options = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary:logistic",
        "eval_metric": "logloss", "random_state": random_state, "n_jobs": 1,
    }
    if positive and negative > positive:
        options["scale_pos_weight"] = negative / positive
    options.update(model_options or {})
    model = _xgboost_classifier()(**options)
    # A NumPy matrix makes the saved native model independent of pandas internals.
    model.fit(train_x.to_numpy(), train_y.to_numpy())
    predictions = model.predict(test_x.to_numpy())
    probabilities = model.predict_proba(test_x.to_numpy())[:, 1]
    metrics = {
        "label_counts": {str(key): int(value) for key, value in labels.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision": float(precision_score(test_y, predictions, zero_division=0)),
        "recall": float(recall_score(test_y, predictions, zero_division=0)),
        "f1": float(f1_score(test_y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
        "classification_report": classification_report(test_y, predictions, zero_division=0),
    }
    output_path, features_path, importance_path = Path(output_path), Path(features_path), Path(importance_path)
    for path in (output_path, features_path, importance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    features_path.write_text(json.dumps({
        "feature_names": CANDIDATE_FEATURE_NAMES,
        "label_semantics": "CV-level source-dataset suitability: 1=suitable, 0=unsuitable.",
        "pair_label_limitation": "No CV--JD pair labels were available; pair features are not trained by this model.",
    }, indent=2), encoding="utf-8")
    importance = pd.DataFrame({"feature_name": CANDIDATE_FEATURE_NAMES, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(importance_path, index=False)
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CV-level XGBoost suitability model.")
    parser.add_argument("--dataset", default="example_data/ml_resume_dataset_4500.csv")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--importance-output", default=str(DEFAULT_IMPORTANCE_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    _, metrics = train_xgboost(args.dataset, args.output, args.features_output, args.importance_output,
                               test_size=args.test_size)
    print("Label distribution:", metrics["label_counts"])
    print(metrics["classification_report"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print(f"Saved model: {args.output}")
    print(f"Saved feature schema: {args.features_output}")
    print(f"Saved feature importance: {args.importance_output}")


if __name__ == "__main__":
    main()
