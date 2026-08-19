"""Train the XGBoost suitability model from ats_resume_dataset_elite_v3.csv.

Architecture
------------
One canonical feature pipeline is used for training AND inference:

    CV + Job row  →  build_pair_features()  →  feature vector  →  XGBoost

The complete artifact written to disk is:

    data/models/
    ├── xgb_pipeline.joblib   — sklearn Pipeline (preprocessor + XGBoost)
    ├── xgb_schema.json       — versioned feature contract
    └── xgb_metadata.json     — training metadata / metrics

Label provenance
----------------
``shortlisted`` in ats_resume_dataset_elite_v3.csv is ALGORITHMICALLY
DERIVED: ``shortlisted = (final_score >= 0.4)`` where ``final_score`` is a
linear composite of ``skill_match_score``, ``experience_match``, and
``education_match``.  It is NOT a real recruiter outcome.  The trained model
therefore learns to reproduce the source scoring rule — it is a proof-of-
concept demonstrating the pipeline architecture, NOT a validated hiring model.

Data split strategy
-------------------
Because every ``resume_id`` is unique (no repeated candidates), we split at
the row level with stratification on ``job_role`` to ensure every role is
represented proportionally in train / validation / test splits.

Leakage prevention
------------------
Any preprocessing fitted on observed data (e.g. sklearn scalers or encoders)
is fitted ONLY on the training partition and then applied to validation and
test without refitting.  The current feature pipeline uses purely deterministic
arithmetic so no fitted preprocessing is required — XGBoost operates directly
on the float feature matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_builder import (
    PAIR_FEATURE_NAMES,
    build_pair_features,
    validate_feature_row,
)
from src.models.xgboost_model import _xgboost_classifier

# ──────────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ELITE_DATASET_PATH = Path("data/ats_resume_dataset_elite_v3.csv")
DEFAULT_PIPELINE_PATH = Path("data/models/xgb_pipeline.joblib")
DEFAULT_SCHEMA_PATH = Path("data/models/xgb_schema.json")
DEFAULT_METADATA_PATH = Path("data/models/xgb_metadata.json")
DEFAULT_IMPORTANCE_PATH = Path("data/models/xgb_feature_importance.csv")

# Kept for backward compatibility (some tests reference these).
DEFAULT_MODEL_PATH = Path("data/models/cv_job_xgb.json")
DEFAULT_FEATURES_PATH = Path("data/models/xgb_features.json")

SCHEMA_VERSION = "2.0"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset → canonical features
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_cv(row: dict[str, Any]) -> dict[str, Any]:
    """Map one CSV row to a CV dict understood by ``build_pair_features``."""
    return {
        "resume_text":      row.get("resume_text", ""),
        "resume_skills":    row.get("resume_skills", ""),
        "experience_years": row.get("experience_years", 0),
        "education_level":  row.get("education_level", ""),
        "projects":         row.get("projects", ""),
        "certifications":   row.get("certifications", ""),
        "has_portfolio":    False,  # not present in the elite dataset
    }


def _row_to_job(row: dict[str, Any]) -> dict[str, Any]:
    """Map one CSV row to a job dict understood by ``build_pair_features``."""
    return {
        "job_title":               row.get("job_role", ""),
        "job_role":                row.get("job_role", ""),
        "required_skills":         row.get("required_skills", ""),
        "job_experience_required": row.get("job_experience_required", 0),
        "job_description":         row.get("job_description", ""),
        "education_level":         "",  # not available per-job in this dataset
    }


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    semantic_col: str = "similarity_score",
    faiss_col: str | None = None,
) -> pd.DataFrame:
    """Convert the elite CSV into the canonical feature matrix.

    NOTE: ``similarity_score`` in the elite CSV has essentially zero variance
    (std ≈ 2.8e-16); it is used as a stand-in for semantic similarity during
    training.  During inference the real MPNet cosine similarity is used.

    Parameters
    ----------
    df:
        Raw elite dataset DataFrame.
    semantic_col:
        Column name in *df* to use as ``semantic_similarity`` (0–1 scale).
    faiss_col:
        Column name in *df* to use as ``faiss_similarity``, or ``None`` to
        default to 0.0 (appropriate when FAISS is not run during training).

    Returns
    -------
    pd.DataFrame
        Columns = ``PAIR_FEATURE_NAMES``.  Index preserved from *df*.
    """
    rows: list[dict[str, float]] = []
    for record in df.to_dict("records"):
        cv = _row_to_cv(record)
        job = _row_to_job(record)

        sem = float(record.get(semantic_col, 0.0) or 0.0)
        # similarity_score in the CSV is on a 0–1 scale already.
        faiss = float(record.get(faiss_col, 0.0) or 0.0) if faiss_col else 0.0

        feat = build_pair_features(cv, job, semantic_similarity=sem, faiss_similarity=faiss)
        # Strict validation — build_pair_features always produces the full set,
        # but we re-validate here as an integration-level safety net.
        validate_feature_row(feat)
        rows.append(feat)

    return pd.DataFrame(rows, columns=PAIR_FEATURE_NAMES, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    dataset_path: str | Path = DEFAULT_ELITE_DATASET_PATH,
    pipeline_path: str | Path = DEFAULT_PIPELINE_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    importance_path: str | Path = DEFAULT_IMPORTANCE_PATH,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
    model_options: dict[str, Any] | None = None,
    # Legacy compat — output_path and features_path not used but accepted so
    # existing callers (e.g. test_features.py) don't break.
    output_path: str | Path | None = None,
    features_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train XGBoost using the canonical pair feature builder.

    Returns
    -------
    (pipeline, metrics) where ``pipeline`` is the fitted sklearn Pipeline
    and ``metrics`` is a dict of evaluation results on the held-out test split.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    # ── Validate required columns ─────────────────────────────────────────────
    elite_required = {
        "shortlisted", "resume_skills", "experience_years",
        "job_experience_required", "job_role",
    }
    missing_cols = elite_required - set(frame.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_cols)}"
        )

    # ── Build feature matrix using canonical feature builder ──────────────────
    print(f"Building feature matrix for {len(frame)} rows using PAIR_FEATURE_NAMES …")
    X = build_feature_matrix(frame)
    y = frame["shortlisted"].astype(int)

    # ── Stratified split: train / validation / test ───────────────────────────
    # Stratify on job_role to keep class and role balance across splits.
    # Because resume_id is unique (6000 unique IDs confirmed), there is no
    # candidate duplication across splits.
    stratify_col = frame["job_role"] if "job_role" in frame.columns else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )
    stratify_val = (
        frame.loc[X_trainval.index, "job_role"] if stratify_col is not None else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size / (1.0 - test_size),
        stratify=stratify_val,
        random_state=random_state,
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train positive rate: {y_train.mean():.3f}")

    # ── Class balance ─────────────────────────────────────────────────────────
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    # ── Build sklearn Pipeline (scaler + XGBoost) ─────────────────────────────
    # StandardScaler is fitted ONLY on training data (no leakage).
    xgb_options: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": 1,
    }
    xgb_options.update(model_options or {})

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", _xgboost_classifier()(**xgb_options)),
    ])

    pipeline.fit(X_train, y_train)

    # ── Validation metrics ────────────────────────────────────────────────────
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_pred = pipeline.predict(X_val)
    val_metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
        "val_f1":      float(f1_score(y_val, val_pred, zero_division=0)),
    }
    print(f"  Val ROC-AUC: {val_metrics['val_roc_auc']:.4f}  "
          f"Val F1: {val_metrics['val_f1']:.4f}")

    # ── Test metrics ──────────────────────────────────────────────────────────
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = pipeline.predict(X_test)

    # Probability validity check.
    assert np.all((test_proba >= 0.0) & (test_proba <= 1.0)), \
        "predict_proba returned values outside [0, 1]"

    metrics: dict[str, Any] = {
        "label_counts":          {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_size":            len(X_train),
        "val_size":              len(X_val),
        "test_size":             len(X_test),
        "accuracy":              float(accuracy_score(y_test, test_pred)),
        "precision":             float(precision_score(y_test, test_pred, zero_division=0)),
        "recall":                float(recall_score(y_test, test_pred, zero_division=0)),
        "f1":                    float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc":               float(roc_auc_score(y_test, test_proba)),
        "pr_auc":                float(average_precision_score(y_test, test_proba)),
        "confusion_matrix":      confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(y_test, test_pred, zero_division=0),
        **val_metrics,
        "label_provenance": (
            "ALGORITHMIC (not real recruiter data): "
            "shortlisted = (final_score >= 0.4) where final_score is a "
            "linear composite of skill_match_score + experience_match + education_match. "
            "XGBoost is learning to reproduce the source scoring rule."
        ),
        "split_strategy": (
            "Row-level stratified split on job_role. "
            "resume_id is unique per row — no candidate duplication across splits."
        ),
    }

    # ── Persist artifacts ─────────────────────────────────────────────────────
    for path in (pipeline_path, schema_path, metadata_path, importance_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Complete sklearn Pipeline (preprocessor + model).
    joblib.dump(pipeline, pipeline_path)
    print(f"  Saved pipeline: {pipeline_path}")

    # 2. Feature schema (authoritative contract).
    schema = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "XGBClassifier",
        "target": "shortlisted",
        "feature_names": PAIR_FEATURE_NAMES,
        "feature_count": len(PAIR_FEATURE_NAMES),
        "label_provenance": metrics["label_provenance"],
        "builder_module": "src.models.feature_builder",
        "builder_function": "build_pair_features",
    }
    Path(schema_path).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"  Saved schema: {schema_path}")

    # 3. Training metadata.
    Path(metadata_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved metadata: {metadata_path}")

    # 4. Feature importances.
    xgb_model = pipeline.named_steps["model"]
    importance = pd.DataFrame({
        "feature_name": PAIR_FEATURE_NAMES,
        "importance": xgb_model.feature_importances_,
    })
    importance.sort_values("importance", ascending=False).to_csv(importance_path, index=False)
    print(f"  Saved importances: {importance_path}")

    # 5. Legacy artifacts (backward compat for old XGBCandidateScorer.load()).
    # Write the raw XGBoost model JSON and a features file that the old scorer
    # can still load when running tests from before the refactor.
    if output_path is not None:
        xgb_model.save_model(str(output_path))
    legacy_features_path = features_path or DEFAULT_FEATURES_PATH
    Path(legacy_features_path).parent.mkdir(parents=True, exist_ok=True)
    Path(legacy_features_path).write_text(
        json.dumps({"feature_names": PAIR_FEATURE_NAMES, **schema}, indent=2),
        encoding="utf-8",
    )

    return pipeline, metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost candidate suitability model."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_ELITE_DATASET_PATH))
    parser.add_argument("--pipeline-output", default=str(DEFAULT_PIPELINE_PATH))
    parser.add_argument("--schema-output", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--importance-output", default=str(DEFAULT_IMPORTANCE_PATH))
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    _, metrics = train_xgboost(
        dataset_path=args.dataset,
        pipeline_path=args.pipeline_output,
        schema_path=args.schema_output,
        metadata_path=args.metadata_output,
        importance_path=args.importance_output,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    print("\n=== Training Results ===")
    print("Label distribution:", metrics["label_counts"])
    print(metrics["classification_report"])
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"\nLabel provenance: {metrics['label_provenance']}")


if __name__ == "__main__":
    main()
