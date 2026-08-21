from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter
from typing import Callable
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    JobRequest,
    MatchRequest,
    MatchResponse,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
)
from src.config import PipelineConfig
from src.database import check_supabase_connection, get_supabase_client
from src.services.matching_service import MatchingService, ModelRegistry, ServiceUnavailableError

logger = logging.getLogger(__name__)

SERVICE_NAME, VERSION = "ai-cv-matching", "1.0.0"

def _error(code: str, message: str) -> dict: return {"error": {"code": code, "message": message}}

def _extract_upload(filename: str, content_type: str | None, payload: bytes) -> str:
    """Extract text from validated in-memory PDF/DOCX data only."""
    suffix = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    allowed = {"pdf": {"application/pdf", "application/x-pdf", "application/octet-stream"}, "docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/octet-stream"}}
    if suffix not in allowed or content_type not in allowed[suffix]: raise ValueError("UNSUPPORTED_MEDIA_TYPE")
    if not payload: raise ValueError("EMPTY_FILE")
    try:
        if suffix == "pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
        else:
            from docx import Document
            text = "\n".join(item.text for item in Document(BytesIO(payload)).paragraphs)
    except Exception as exc:
        logger.info("CV extraction rejected type=%s error=%s", suffix, type(exc).__name__)
        raise ValueError("INVALID_DOCUMENT") from exc
    if not text.strip(): raise ValueError("EMPTY_FILE")
    return text

def _candidate(row: dict) -> dict:
    return {"candidate_id": row.get("candidate_id"), "candidate_name": row.get("candidate_name", "Unknown Candidate"), "rank": row.get("rank"), "faiss_similarity": row.get("faiss_similarity"), "match_probability": row.get("match_probability", row.get("xgb_probability", 0.0)), "matched_skills": row.get("matched_skills", []), "missing_skills": row.get("missing_required_skills", []), "recommendation": row.get("recommendation"), "explanation": row.get("explanation"), "explanation_status": row.get("explanation_status")}

def create_app(config: PipelineConfig | None = None, registry_factory: Callable[[PipelineConfig], ModelRegistry] = ModelRegistry):

    config = config or PipelineConfig()
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        registry = registry_factory(config); registry.load()
        application.state.registry, application.state.service = registry, MatchingService(registry)
        logger.info("Matching service initialized: %s", registry.readiness()); yield
    app = FastAPI(title="AI CV Matching API", version=VERSION, lifespan=lifespan)
    origins = [x.strip() for x in config.cors_origins.split(",") if x.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"])
    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id, started = request.headers.get("X-Request-ID") or str(uuid4()), perf_counter()
        try: response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request error request_id=%s endpoint=%s", request_id, request.url.path)
            return JSONResponse(_error("INTERNAL_ERROR", "Unexpected server error"), 500, headers={"X-Request-ID": request_id})
        response.headers["X-Request-ID"] = request_id
        logger.info("request_id=%s endpoint=%s status=%d latency_ms=%.2f", request_id, request.url.path, response.status_code, (perf_counter()-started)*1000)
        return response
    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, __: RequestValidationError): return JSONResponse(_error("VALIDATION_ERROR", "Invalid request"), 422)
    def service(request: Request) -> MatchingService: return request.app.state.service
    @app.get("/health", response_model=HealthResponse, tags=["Operations"])
    async def health(): return {"status": "ok", "service": SERVICE_NAME, "version": VERSION}
    @app.get("/ready", response_model=ReadyResponse, tags=["Operations"])
    async def ready(request: Request): return request.app.state.registry.readiness()
    @app.post("/api/v1/search", response_model=SearchResponse, tags=["Matching"])
    async def search(request: Request, body: SearchRequest):
        try:
            job = service(request).build_job(body.job.dict()); rows, latency = await run_in_threadpool(service(request).search, job, body.top_k)
            return {"results": rows, "latency": latency}
        except ServiceUnavailableError as exc: raise HTTPException(503, _error("MODEL_NOT_READY", str(exc)))
    @app.post("/api/v1/match", response_model=MatchResponse, tags=["Matching"])
    async def match(request: Request, body: MatchRequest):
        try:
            job = service(request).build_job(body.job.dict()); rows, latency = await run_in_threadpool(service(request).match, job, body.top_k, body.explain)
            return {"job_title": job.get("job_title", "Untitled Job"), "results": [_candidate(x) for x in rows], "latency": latency}
        except ServiceUnavailableError as exc: raise HTTPException(503, _error("MODEL_NOT_READY", str(exc)))
    # @app.post("/api/v1/match/upload", response_model=MatchResponse, tags=["Matching"])
    # async def match_upload(request: Request, cv: UploadFile = File(...), job: str = Form(...), top_k: int = Form(1), explain: bool = Form(False)):
    #     if not 1 <= top_k <= config.api_top_k_max: raise HTTPException(422, _error("VALIDATION_ERROR", f"top_k must be between 1 and {config.api_top_k_max}"))
    #     content = await cv.read(config.max_upload_size + 1)
    #     if len(content) > config.max_upload_size: raise HTTPException(413, _error("FILE_TOO_LARGE", "CV exceeds configured upload size"))
    #     try:
    #         raw_cv, parsed = _extract_upload(cv.filename or "", cv.content_type, content), JobRequest(**json.loads(job))
    #         parsed_job = service(request).build_job(parsed.dict()); result, latency = await run_in_threadpool(service(request).evaluate_cv, raw_cv, parsed_job, explain)
    #         return {"job_title": parsed_job.get("job_title", "Untitled Job"), "results": [_candidate(result)], "latency": latency}
    #     except ValueError as exc:
    #         code = str(exc); statuses = {"UNSUPPORTED_MEDIA_TYPE": 415, "EMPTY_FILE": 400, "INVALID_DOCUMENT": 400}
    #         raise HTTPException(statuses.get(code, 422), _error(code if code in statuses else "VALIDATION_ERROR", "Invalid CV upload" if code in statuses else code))
    #     except ServiceUnavailableError as exc: raise HTTPException(503, _error("MODEL_NOT_READY", str(exc)))
    
    @app.post(
    "/api/v1/match/upload",
    response_model=MatchResponse,
    tags=["Matching"],)
    async def match_upload(
        request: Request,
        cv: UploadFile = File(...),
        job: str = Form(...),
        top_k: int = Form(1),
        explain: bool = Form(False),
    ):
        if not 1 <= top_k <= config.api_top_k_max:
            raise HTTPException(
                status_code=422,
                detail=_error(
                    "VALIDATION_ERROR",
                    f"top_k must be between 1 and {config.api_top_k_max}",
                ),
            )

        content = await cv.read(config.max_upload_size + 1)

        if len(content) > config.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=_error(
                    "FILE_TOO_LARGE",
                    "CV exceeds configured upload size",
                ),
            )

        try:
            raw_cv = _extract_upload(
                cv.filename or "",
                cv.content_type,
                content,
            )

            parsed = JobRequest(**json.loads(job))
            parsed_job = service(request).build_job(
                parsed.model_dump()
            )

            result, latency = await run_in_threadpool(
                service(request).evaluate_cv,
                raw_cv,
                parsed_job,
                explain,
            )

            return {
                "job_title": parsed_job.get("job_title", "Untitled Job"),
                "results": [_candidate(result)],
                "latency": latency,
            }

        except ValueError as exc:
            code = str(exc)

            statuses = {
                "UNSUPPORTED_MEDIA_TYPE": 415,
                "EMPTY_FILE": 400,
                "INVALID_DOCUMENT": 400,
            }

            raise HTTPException(
                status_code=statuses.get(code, 422),
                detail=_error(
                    code if code in statuses else "VALIDATION_ERROR",
                    "Invalid CV upload" if code in statuses else code,
                ),
            )

        except ServiceUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail=_error("MODEL_NOT_READY", str(exc)),
            )
    
    @app.post("/api/v1/explain", response_model=ExplainResponse, tags=["Explanations"])
    async def explain(request: Request, body: ExplainRequest):
        job = service(request).build_job(body.job.dict()); text, status = await run_in_threadpool(service(request).explain_evidence, body.candidate.dict(), job, body.match_probability)
        return {"explanation": text, "explanation_status": status}
    return app

app = create_app()
