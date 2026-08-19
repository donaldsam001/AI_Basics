"""Train the XGBoost suitability model without label leakage.

Supports candidate-job pair datasets (e.g. ats_resume_dataset_elite_v3.csv)
and candidate-only datasets (e.g. ml_resume_dataset_4500.csv).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CANDIDATE_FEATURE_NAMES, build_candidate_features
from src.models.xgboost_model import _xgboost_classifier

DEFAULT_ELITE_DATASET_PATH = Path("data/ats_resume_dataset_elite_v3.csv")
DEFAULT_LEGACY_DATASET_PATH = Path("example_data/ml_resume_dataset_4500.csv")
DEFAULT_MODEL_PATH = Path("data/models/cv_job_xgb.json")
DEFAULT_FEATURES_PATH = Path("data/models/xgb_features.json")
DEFAULT_IMPORTANCE_PATH = Path("data/models/xgb_feature_importance.csv")


def preprocess_elite_dataset(
    df: pd.DataFrame, recompute_similarity: bool = True
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    """Preprocess elite ATS resume dataset (ats_resume_dataset_elite_v3.csv).

    Engineers features, cleans missing certifications, recomputes semantic similarity
    to fix constant similarity_score values, adds text length features, and excludes target
    leakage columns ('final_score', 'resume_id').
    """
    frame = df.copy()

    # 1. Handle missing values in certifications (2,058 missing entries)
    frame["certifications"] = frame["certifications"].fillna("None")
    frame["has_certification"] = (frame["certifications"] != "None").astype(int)
    frame["cert_count"] = frame["certifications"].apply(
        lambda x: 0 if str(x).strip().casefold() in ["none", ""] else len(re.split(r"[,|;\n]", str(x)))
    )

    # 2. Count parsed skill lists
    frame["num_resume_skills"] = frame["resume_skills"].apply(lambda x: len(str(x).split(",")))
    frame["num_required_skills"] = frame["required_skills"].apply(lambda x: len(str(x).split(",")))

    # 3. Experience delta
    frame["experience_diff"] = frame["experience_years"] - frame["job_experience_required"]

    # 4. Recompute semantic similarity (Fix Issue 3) & extract text length features (Fix Issue 5)
    if recompute_similarity or ("similarity_score" in frame and frame["similarity_score"].std() == 0):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        resume_combined = frame["resume_text"].fillna("") + " " + frame["projects"].fillna("")
        job_combined = frame["job_description"].fillna("") + " " + frame["required_skills"].fillna("")
        vectorizer.fit(resume_combined + " " + job_combined)
        r_vecs = vectorizer.transform(resume_combined)
        j_vecs = vectorizer.transform(job_combined)
        sims = np.array([cosine_similarity(r_vecs[i], j_vecs[i])[0][0] for i in range(len(frame))])
        frame["similarity_score"] = sims

    frame["resume_text_len"] = frame["resume_text"].fillna("").apply(len)
    frame["job_desc_len"] = frame["job_description"].fillna("").apply(len)

    # 5. Define feature columns and target (Avoid target leakage: drop 'final_score', 'resume_id')
    categorical_cols = ["education_level", "job_role"]
    feature_cols = [
        "experience_years",
        "job_experience_required",
        "experience_diff",
        "skill_match_score",
        "experience_match",
        "education_match",
        "similarity_score",
        "has_certification",
        "cert_count",
        "num_resume_skills",
        "num_required_skills",
        "resume_text_len",
        "job_desc_len",
        "education_level",
        "job_role",
    ]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    X = frame[feature_cols]
    y = frame["shortlisted"]
    return X, y, feature_cols, categorical_cols, numeric_cols


def build_training_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ordered numeric features for legacy dataset and never include the target column."""
    rows = [build_candidate_features(record) for record in frame.to_dict("records")]
    return pd.DataFrame(rows, columns=CANDIDATE_FEATURE_NAMES, dtype=float)


def train_xgboost(
    dataset_path: str | Path,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    importance_path: str | Path = DEFAULT_IMPORTANCE_PATH,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    model_options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    elite_required = {"shortlisted", "resume_skills", "experience_years", "job_experience_required", "job_role"}
    legacy_required = {"label", "years_experience", "highest_degree", "skills", "current_title", "has_portfolio", "raw_text"}

    if elite_required.issubset(set(frame.columns)):
        # Pipeline for ats_resume_dataset_elite_v3.csv
        X, y, feature_cols, categorical_cols, numeric_cols = preprocess_elite_dataset(frame)

        train_x, test_x, train_y, test_y = train_test_split(
            X, y.astype(int), test_size=test_size, stratify=y, random_state=random_state
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ]
        )

        positive = int((train_y == 1).sum())
        negative = int((train_y == 0).sum())
        scale_pos_weight = negative / positive if positive else 1.0

        options = {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": 1,
        }
        options.update(model_options or {})

        xgb_model = _xgboost_classifier()(**options)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_model)])

        pipeline.fit(train_x, train_y)

        # Extract transformed feature names
        cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        transformed_feature_names = cat_feature_names + numeric_cols

        model = pipeline.named_steps["model"]
        predictions = pipeline.predict(test_x)
        probabilities = pipeline.predict_proba(test_x)[:, 1]

        metrics = {
            "label_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
            "accuracy": float(accuracy_score(test_y, predictions)),
            "precision": float(precision_score(test_y, predictions, zero_division=0)),
            "recall": float(recall_score(test_y, predictions, zero_division=0)),
            "f1": float(f1_score(test_y, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_y, probabilities)),
            "pr_auc": float(average_precision_score(test_y, probabilities)),
            "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
            "classification_report": classification_report(test_y, predictions, zero_division=0),
        }

        output_path, features_path, importance_path = Path(output_path), Path(features_path), Path(importance_path)
        for path in (output_path, features_path, importance_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        model.save_model(str(output_path))
        features_path.write_text(
            json.dumps(
                {
                    "feature_names": transformed_feature_names,
                    "raw_feature_cols": feature_cols,
                    "categorical_cols": categorical_cols,
                    "numeric_cols": numeric_cols,
                    "label_semantics": "Candidate-Job pair suitability: 1=shortlisted, 0=rejected.",
                    "dataset_source": str(dataset_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        importance = pd.DataFrame({"feature_name": transformed_feature_names, "importance": model.feature_importances_})
        importance.sort_values("importance", ascending=False).to_csv(importance_path, index=False)
        return model, metrics

    elif legacy_required.issubset(set(frame.columns)):
        labels = frame["label"]
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError("label must contain only binary 0 and 1 values")
        features = build_training_features(frame)
        train_x, test_x, train_y, test_y = train_test_split(
            features, labels.astype(int), test_size=test_size, stratify=labels, random_state=random_state
        )
        positive = int((train_y == 1).sum())
        negative = int((train_y == 0).sum())
        options = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": 1,
        }
        if positive and negative > positive:
            options["scale_pos_weight"] = negative / positive
        options.update(model_options or {})
        model = _xgboost_classifier()(**options)
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
            "pr_auc": float(average_precision_score(test_y, probabilities)),
            "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
            "classification_report": classification_report(test_y, predictions, zero_division=0),
        }
        output_path, features_path, importance_path = Path(output_path), Path(features_path), Path(importance_path)
        for path in (output_path, features_path, importance_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_path))
        features_path.write_text(
            json.dumps(
                {
                    "feature_names": CANDIDATE_FEATURE_NAMES,
                    "label_semantics": "CV-level source-dataset suitability: 1=suitable, 0=unsuitable.",
                    "pair_label_limitation": "No CV--JD pair labels were available; pair features are not trained by this model.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        importance = pd.DataFrame({"feature_name": CANDIDATE_FEATURE_NAMES, "importance": model.feature_importances_})
        importance.sort_values("importance", ascending=False).to_csv(importance_path, index=False)
        return model, metrics
    else:
        missing_elite = sorted(elite_required - set(frame.columns))
        missing_legacy = sorted(legacy_required - set(frame.columns))
        raise ValueError(
            f"Dataset is missing required columns. For elite dataset, missing: {', '.join(missing_elite)}. "
            f"For legacy dataset, missing: {', '.join(missing_legacy)}"
        )


def main() -> None:
    default_dataset = (
        str(DEFAULT_ELITE_DATASET_PATH) if DEFAULT_ELITE_DATASET_PATH.is_file() else str(DEFAULT_LEGACY_DATASET_PATH)
    )
    parser = argparse.ArgumentParser(description="Train the XGBoost candidate suitability model.")
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--importance-output", default=str(DEFAULT_IMPORTANCE_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    _, metrics = train_xgboost(
        args.dataset, args.output, args.features_output, args.importance_output, test_size=args.test_size
    )
    print("Label distribution:", metrics["label_counts"])
    print(metrics["classification_report"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print(f"Saved model: {args.output}")
    print(f"Saved feature schema: {args.features_output}")
    print(f"Saved feature importance: {args.importance_output}")


if __name__ == "__main__":
    main()

