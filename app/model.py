"""Model loading and inference logic — Step 5 of the FastAPI workflow.

Loads the trained sklearn pipeline (``model_pipeline.joblib``) once at
startup, and exposes a thin ``predict()`` function plus the full
intelligent matching pipeline for the /match endpoints.

The pipeline artifact contains:
    ColumnTransformer (OneHotEncoder for categoricals) → XGBClassifier

Expected input feature vector (12 values, in order):
    experience_years, job_experience_required, experience_diff,
    skill_match_score, experience_match, education_match,
    similarity_score, has_certification, num_resume_skills,
    num_required_skills, education_level (str), job_role (str)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Trained sklearn Pipeline (model_pipeline.joblib) ─────────────────────

_pipeline_model = None

# Feature columns matching train.py exactly
FEATURE_COLS = [
    "experience_years", "job_experience_required", "experience_diff",
    "skill_match_score", "experience_match", "education_match",
    "similarity_score", "has_certification", "num_resume_skills",
    "num_required_skills", "education_level", "job_role",
]
CATEGORICAL_COLS = ["education_level", "job_role"]
NUMERIC_COLS = [col for col in FEATURE_COLS if col not in CATEGORICAL_COLS]


def _resolve_model_path() -> Path:
    """Locate model_pipeline.joblib relative to the project root."""
    # Try several locations: project root, app/, data/models/
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "model_pipeline.joblib",
        project_root / "app" / "model_pipeline.joblib",
        project_root / "data" / "models" / "model_pipeline.joblib",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"model_pipeline.joblib not found in any of: {[str(c) for c in candidates]}"
    )


def get_model():
    """Load the sklearn pipeline once (lazy singleton)."""
    global _pipeline_model
    if _pipeline_model is None:
        model_path = _resolve_model_path()
        logger.info("Loading sklearn pipeline from %s", model_path)
        _pipeline_model = joblib.load(model_path)
        logger.info("Pipeline loaded successfully")
    return _pipeline_model


def predict(features: list[float | str]) -> dict[str, float]:
    """Run prediction through the trained sklearn pipeline.

    Parameters
    ----------
    features:
        A list of 12 values matching ``FEATURE_COLS``.
        The last two values (education_level, job_role) should be strings;
        the OneHotEncoder in the pipeline handles unknown categories.

    Returns
    -------
    dict with ``prediction`` (0 or 1) and ``confidence`` (probability).
    """
    model = get_model()

    if len(features) != len(FEATURE_COLS):
        raise ValueError(
            f"Expected {len(FEATURE_COLS)} features, got {len(features)}. "
            f"Feature order: {FEATURE_COLS}"
        )

    # Build a DataFrame matching the training schema
    row = {}
    for i, col in enumerate(FEATURE_COLS):
        if col in CATEGORICAL_COLS:
            # Categorical columns must be strings for the OneHotEncoder
            row[col] = str(features[i]) if features[i] else "Unknown"
        else:
            row[col] = float(features[i])

    df = pd.DataFrame([row])
    prediction = float(model.predict(df)[0])

    # Extract confidence (probability of positive class)
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        confidence = float(proba[0][1])  # P(shortlisted=1)

    return {"prediction": prediction, "confidence": confidence}


# ── Full Intelligent Matching Pipeline ───────────────────────────────────

_matching_pipeline = None


def get_matching_pipeline():
    """Lazy-load the full CV–Job matching pipeline.

    Imports are deferred to avoid heavy startup costs if only the
    /predict endpoint is used.
    """
    global _matching_pipeline
    if _matching_pipeline is not None:
        return _matching_pipeline

    # Add project root to sys.path so ``src.*`` imports resolve.
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.config import PipelineConfig
    from src.pipeline import create_pipeline

    config = PipelineConfig()
    _matching_pipeline = create_pipeline(config)
    logger.info("Full matching pipeline loaded")
    return _matching_pipeline


def get_pipeline_config():
    """Return a PipelineConfig instance."""
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.config import PipelineConfig
    return PipelineConfig()
