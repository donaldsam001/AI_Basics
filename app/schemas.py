"""Pydantic request/response models for the CV–Job Matching API.

Step 4 — Request/Response schemas adapted for the intelligent
matching pipeline.  The input is structured CV + job data (not raw
float vectors), and the output includes ranking, skill analysis,
and optional XGBoost probability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


# ── Prediction Endpoint (raw float features → model) ────────────────────

class PredictionRequest(BaseModel):
    """Raw feature vector prediction — matches the training model's input shape.

    The last two values (``education_level``, ``job_role``) are categorical
    strings — the pipeline's OneHotEncoder handles them.  All other values
    are numeric floats.
    """
    features: list[float | str] = Field(
        ...,
        description=(
            "Feature vector matching the trained sklearn pipeline's input: "
            "[experience_years, job_experience_required, experience_diff, "
            "skill_match_score, experience_match, education_match, "
            "similarity_score, has_certification, num_resume_skills, "
            "num_required_skills, education_level (str), job_role (str)]"
        ),
    )


class PredictionResponse(BaseModel):
    """Raw model prediction result."""
    prediction: float
    confidence: float | None = None


# ── Match Endpoint (structured CV + Job → full pipeline) ────────────────

class SkillGapAnalysis(BaseModel):
    """Skill gap breakdown for a single candidate."""
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    matched_count: int = 0
    missing_count: int = 0
    skill_match_score: float = 0.0


class CandidateResult(BaseModel):
    """One ranked candidate in the matching response."""
    candidate_name: str
    rank: int | None = None
    match_probability: float | None = None
    retrieval_score: float | None = None
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    matched_preferred_skills: list[str] = []
    semantic_score: float | None = None
    required_skill_score: float | None = None
    preferred_skill_score: float | None = None
    experience_score: float | None = None
    baseline_score: float | None = None
    skill_gap_analysis: SkillGapAnalysis | None = None
    explanation: str | None = None
    explanation_error: str | None = None


class MatchRequest(BaseModel):
    """Incoming matching request — structured CV/job data or CSV paths."""
    job_csv_path: str = Field(
        ...,
        description="Path to a CSV containing job description rows",
    )
    job_index: int = Field(
        0, description="Zero-based row index of the job to match"
    )
    cv_csv_path: str | None = Field(
        None, description="Override CV data path (if not pre-loaded)"
    )
    retrieval_top_k: int | None = Field(
        None, description="Override FAISS retrieval depth"
    )
    ranking_top_k: int | None = Field(
        None, description="Override ranking cut-off"
    )
    explanation_top_k: int | None = Field(
        None, description="Override explanation count"
    )


class MatchResponse(BaseModel):
    """Ranked candidates for a single job."""
    job_title: str
    candidates: list[CandidateResult]


# ── Inline CV + Job Matching (no CSV required) ──────────────────────────

class CVInput(BaseModel):
    """Minimal structured CV input for inline matching."""
    candidate_name: str = "Candidate"
    raw_text: str = ""
    technical_skills: list[str] = []
    programming_languages: list[str] = []
    frameworks: list[str] = []
    years_of_experience: float = 0.0
    education: str = ""
    certifications: str = ""
    current_title: str = ""


class JobInput(BaseModel):
    """Minimal structured job description input."""
    job_title: str = "Software Engineer"
    job_description: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    years_of_experience: float = 0.0
    education_level: str = ""


class InlineMatchRequest(BaseModel):
    """Match CVs against a job description directly (no CSV files needed)."""
    cvs: list[CVInput]
    job: JobInput
    top_k: int = Field(10, description="Number of top candidates to return")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    xgboost_available: bool = False
    pipeline_model_loaded: bool = False
    faiss_loaded: bool = False
    cvs_loaded: bool = False
