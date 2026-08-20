"""FastAPI application — Step 6 of the deployment workflow.

Endpoints
---------
GET  /health          Liveness probe with component availability status.
POST /predict         Raw feature vector → sklearn pipeline prediction.
POST /match           CSV-backed full matching pipeline (FAISS + XGBoost + Qwen).
POST /match/inline    Structured JSON CV/Job → matching (no CSV files needed).

Architecture
------------
Heavy models (sklearn pipeline, embedding model, FAISS, XGBoost, Qwen)
are loaded once at startup via the ``lifespan`` context manager and
reused across all requests.  Component failures degrade gracefully —
the API always returns a useful ranking even if XGBoost or Qwen are
unavailable.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    CandidateResult,
    HealthResponse,
    InlineMatchRequest,
    MatchRequest,
    MatchResponse,
    PredictionRequest,
    PredictionResponse,
    SkillGapAnalysis,
)
from app.model import (
    get_model,
    get_matching_pipeline,
    get_pipeline_config,
    predict,
)

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path for src.* imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── Lifespan: warm models at startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup so the first request is fast."""
    logger.info("🔄 Warming up models...")

    # Step 5a: Load the sklearn pipeline (model_pipeline.joblib)
    try:
        get_model()
        app.state.pipeline_model_loaded = True
        logger.info("✅ sklearn pipeline loaded")
    except Exception as err:
        logger.warning("⚠️  sklearn pipeline unavailable: %s", err)
        app.state.pipeline_model_loaded = False

    # Step 5b: Load the full matching pipeline (embedding + FAISS + XGBoost + Qwen)
    try:
        pipeline = get_matching_pipeline()
        app.state.matching_pipeline = pipeline
        logger.info("✅ Matching pipeline loaded")
    except Exception as err:
        logger.warning("⚠️  Matching pipeline unavailable: %s", err)
        app.state.matching_pipeline = None

    # Pre-load CVs if a default path exists
    app.state.cvs = None
    try:
        config = get_pipeline_config()
        from src.main import load_cvs
        app.state.cvs = load_cvs(config.cv_path)
        logger.info("✅ Loaded %d CVs from %s", len(app.state.cvs), config.cv_path)
    except Exception as err:
        logger.warning("⚠️  CVs not pre-loaded: %s", err)

    # Pre-load FAISS stores
    app.state.vector_store = None
    app.state.metadata_store = None
    try:
        config = get_pipeline_config()
        from src.vector_store import FAISSStore, MetadataStore
        app.state.vector_store = FAISSStore.load(config.faiss_index_path)
        app.state.metadata_store = MetadataStore(config.faiss_metadata_path)
        logger.info("✅ FAISS index loaded: %d vectors", app.state.vector_store.size)
    except Exception as err:
        logger.warning("⚠️  FAISS stores not loaded: %s", err)

    logger.info("🚀 API ready to serve requests")
    yield


# ── Application factory ─────────────────────────────────────────────────

app = FastAPI(
    title="AI_Basics Model API",
    description=(
        "Intelligent CV–Job matching API powered by FAISS retrieval, "
        "XGBoost ranking, and optional Qwen3-4B explanations.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — Raw feature vector prediction\n"
        "- `POST /match` — Full pipeline matching from CSV\n"
        "- `POST /match/inline` — Inline JSON matching\n"
        "- `GET /health` — Component health check"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe with component availability details."""
    pipeline = getattr(app.state, "matching_pipeline", None)
    return HealthResponse(
        status="ok",
        pipeline_model_loaded=getattr(app.state, "pipeline_model_loaded", False),
        xgboost_available=pipeline is not None and pipeline.xgb_scorer is not None,
        faiss_loaded=getattr(app.state, "vector_store", None) is not None,
        cvs_loaded=getattr(app.state, "cvs", None) is not None
        and len(app.state.cvs) > 0,
    )


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    """Raw feature vector → sklearn pipeline prediction.

    Send 12 features matching the training schema:
    ``[experience_years, job_experience_required, experience_diff,
    skill_match_score, experience_match, education_match,
    similarity_score, has_certification, num_resume_skills,
    num_required_skills, education_level, job_role]``
    """
    try:
        result = predict(request.features)
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
        )
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {err}",
        )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/match", response_model=MatchResponse)
def match_candidates(request: MatchRequest):
    """Full pipeline matching from CSV files.

    Loads job descriptions from a CSV, ranks all indexed CVs, and
    returns the top candidates with optional Qwen explanations.
    """
    pipeline = getattr(app.state, "matching_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Matching pipeline is not available. Check /health for details.",
        )

    # Load job
    try:
        from src.main import load_job
        job = load_job(request.job_csv_path, request.job_index)
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(status_code=400, detail=str(err))

    # Load CVs (from request override, pre-loaded state, or error)
    cvs = app.state.cvs
    if request.cv_csv_path:
        try:
            from src.main import load_cvs
            cvs = load_cvs(request.cv_csv_path)
        except (FileNotFoundError, ValueError) as err:
            raise HTTPException(status_code=400, detail=str(err))
    if not cvs:
        raise HTTPException(status_code=400, detail="No CV data available")

    # Apply top-K overrides
    config_snapshot = (
        pipeline.config.retrieval_top_k,
        pipeline.config.ranking_top_k,
        pipeline.config.explanation_top_k,
    )
    if request.retrieval_top_k is not None:
        pipeline.config.retrieval_top_k = request.retrieval_top_k
    if request.ranking_top_k is not None:
        pipeline.config.ranking_top_k = request.ranking_top_k
    if request.explanation_top_k is not None:
        pipeline.config.explanation_top_k = request.explanation_top_k

    try:
        results = pipeline.rank_and_explain(
            cvs, job,
            vector_store=app.state.vector_store,
            metadata_store=app.state.metadata_store,
        )
    except Exception as err:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {err}")
    finally:
        # Restore original top-K values
        pipeline.config.retrieval_top_k = config_snapshot[0]
        pipeline.config.ranking_top_k = config_snapshot[1]
        pipeline.config.explanation_top_k = config_snapshot[2]

    return _build_match_response(job, results)


@app.post("/match/inline", response_model=MatchResponse)
def match_inline(request: InlineMatchRequest):
    """Match CVs against a job description directly using structured JSON.

    No CSV files needed — pass CV and job data inline.
    """
    pipeline = getattr(app.state, "matching_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Matching pipeline is not available. Check /health for details.",
        )

    # Convert Pydantic models to dicts matching the pipeline's expected format
    cvs = [cv.model_dump() for cv in request.cvs]
    job = request.job.model_dump()

    # Override top-K for this request
    original_ranking_top_k = pipeline.config.ranking_top_k
    pipeline.config.ranking_top_k = request.top_k

    try:
        results = pipeline.rank_and_explain(cvs, job)
    except Exception as err:
        logger.exception("Inline matching error")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {err}")
    finally:
        pipeline.config.ranking_top_k = original_ranking_top_k

    return _build_match_response(job, results)


# ── Response builder ─────────────────────────────────────────────────────

def _build_match_response(
    job: dict[str, Any], results: list[dict[str, Any]]
) -> MatchResponse:
    """Convert raw pipeline results into the API response schema."""
    candidates = []
    for result in results:
        # Build skill gap analysis if present
        gap = result.get("skill_gap_analysis")
        skill_gap = None
        if gap and isinstance(gap, dict):
            skill_gap = SkillGapAnalysis(
                matched_skills=gap.get("matched_skills", []),
                missing_skills=gap.get("missing_skills", []),
                matched_count=gap.get("matched_count", 0),
                missing_count=gap.get("missing_count", 0),
                skill_match_score=gap.get("skill_match_score", 0.0),
            )

        candidates.append(
            CandidateResult(
                candidate_name=result.get("candidate_name", "Unknown"),
                rank=result.get("rank"),
                match_probability=result.get(
                    "xgb_probability", result.get("final_score", 0.0)
                ),
                retrieval_score=result.get("faiss_similarity"),
                matched_skills=result.get("matched_skills", []),
                missing_skills=result.get("missing_required_skills", []),
                matched_preferred_skills=result.get("matched_preferred_skills", []),
                semantic_score=result.get("semantic_score"),
                required_skill_score=result.get("required_skill_score"),
                preferred_skill_score=result.get("preferred_skill_score"),
                experience_score=result.get("experience_score"),
                baseline_score=result.get(
                    "deterministic_score", result.get("final_score")
                ),
                skill_gap_analysis=skill_gap,
                explanation=result.get("explanation"),
                explanation_error=result.get("explanation_error"),
            )
        )

    return MatchResponse(
        job_title=job.get("job_title", "Untitled Job"),
        candidates=candidates,
    )
