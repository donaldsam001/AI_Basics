"""FastAPI application for the CV–Job matching pipeline.

Endpoints
---------
POST /match
    Accept a job description, rank all indexed CVs against it, and
    return the top candidates with optional Qwen explanations.

GET /health
    Simple liveness probe.

Architecture
------------
The application loads all heavy models once at startup via the
``lifespan`` context manager and shares them across requests through
``app.state``.  The pipeline handles failures gracefully:

* If XGBoost is unavailable, the baseline deterministic score is used.
* If Qwen is unavailable, explanations are omitted but ranking is intact.
* FAISS retrieval failures return a clear 503.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from src.config import PipelineConfig
from src.main import load_cvs, load_job
from src.pipeline import MatchingPipeline, create_pipeline

logger = logging.getLogger(__name__)


def create_app(config: PipelineConfig | None = None):
    """Factory that creates a configured FastAPI application.

    The heavy import of ``fastapi`` is deferred so the rest of the package
    can be imported without it.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError as error:
        raise RuntimeError(
            "FastAPI is required for the API server. "
            "Install it with: pip install fastapi uvicorn"
        ) from error

    config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Request / response models
    # ------------------------------------------------------------------

    class MatchRequest(BaseModel):
        """Incoming matching request."""
        job_csv_path: str = Field(..., description="Path to a CSV with job description rows")
        job_index: int = Field(0, description="Zero-based row index of the job to match")
        cv_csv_path: str | None = Field(None, description="Override CV data path")
        retrieval_top_k: int | None = Field(None, description="Override FAISS retrieval depth")
        ranking_top_k: int | None = Field(None, description="Override ranking cut-off")
        explanation_top_k: int | None = Field(None, description="Override explanation count")

    class CandidateResponse(BaseModel):
        """One ranked candidate in the response."""
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
        explanation: str | None = None
        explanation_error: str | None = None

    class MatchResponse(BaseModel):
        """Ranked candidates for a single job."""
        job_title: str
        candidates: list[CandidateResponse]

    # ------------------------------------------------------------------
    # Lifespan: load models once
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading pipeline components...")
        pipeline = create_pipeline(config)
        app.state.pipeline = pipeline
        app.state.config = config

        # Preload CVs if the path exists.
        try:
            app.state.cvs = load_cvs(config.cv_path)
            logger.info("Loaded %d CVs from %s", len(app.state.cvs), config.cv_path)
        except Exception as error:
            logger.warning("CVs not preloaded: %s", error)
            app.state.cvs = None

        # Preload FAISS stores if the paths exist.
        app.state.vector_store = None
        app.state.metadata_store = None
        try:
            from src.vector_store import FAISSStore, MetadataStore
            app.state.vector_store = FAISSStore.load(config.faiss_index_path)
            app.state.metadata_store = MetadataStore(config.faiss_metadata_path)
            logger.info("FAISS index loaded: %d vectors", app.state.vector_store.size)
        except Exception as error:
            logger.warning("FAISS stores not loaded: %s", error)

        yield

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    app = FastAPI(
        title="CV–Job Matching API",
        description=(
            "Rank CVs against a job description using FAISS retrieval, "
            "rule-based matching, XGBoost scoring, and optional Qwen3-4B "
            "explanations."
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

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        pipeline: MatchingPipeline = app.state.pipeline
        return {
            "status": "ok",
            "xgboost_available": pipeline.xgb_scorer is not None,
            "qwen_available": pipeline.qwen is not None,
            "faiss_loaded": app.state.vector_store is not None,
            "cvs_loaded": app.state.cvs is not None and len(app.state.cvs) > 0,
        }

    @app.post("/match", response_model=MatchResponse)
    async def match_candidates(request: MatchRequest):
        pipeline: MatchingPipeline = app.state.pipeline
        active_config = app.state.config

        # Override top-K values if the request supplies them.
        if request.retrieval_top_k is not None:
            pipeline.config.retrieval_top_k = request.retrieval_top_k
        if request.ranking_top_k is not None:
            pipeline.config.ranking_top_k = request.ranking_top_k
        if request.explanation_top_k is not None:
            pipeline.config.explanation_top_k = request.explanation_top_k

        # Load job.
        try:
            job = load_job(request.job_csv_path, request.job_index)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error))

        # Load CVs.
        cvs = app.state.cvs
        if request.cv_csv_path:
            try:
                cvs = load_cvs(request.cv_csv_path)
            except (FileNotFoundError, ValueError) as error:
                raise HTTPException(status_code=400, detail=str(error))
        if not cvs:
            raise HTTPException(status_code=400, detail="No CV data available")

        # Run the pipeline.
        try:
            results = pipeline.rank_and_explain(
                cvs, job,
                vector_store=app.state.vector_store,
                metadata_store=app.state.metadata_store,
            )
        except Exception as error:
            logger.exception("Pipeline error")
            raise HTTPException(status_code=503, detail=f"Pipeline error: {error}")

        # Restore original top-K values.
        pipeline.config.retrieval_top_k = active_config.retrieval_top_k
        pipeline.config.ranking_top_k = active_config.ranking_top_k
        pipeline.config.explanation_top_k = active_config.explanation_top_k

        response = pipeline.build_api_response(job, results)
        return response

    return app
