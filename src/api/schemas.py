"""Versioned, public Pydantic contracts for the HTTP API."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, validator

class JobRequest(BaseModel):
    title: str = Field("", max_length=300)
    description: str = Field("", max_length=20_000)
    required_skills: list[str] = Field(default_factory=list, max_length=100)
    preferred_skills: list[str] = Field(default_factory=list, max_length=100)
    min_experience_years: float | None = Field(None, ge=0, le=80)
    @validator("description", always=True)
    def require_title_or_description(cls, value: str, values: dict[str, Any]) -> str:
        if not (value or "").strip() and not (values.get("title") or "").strip():
            raise ValueError("job requires a non-empty title or description")
        return value.strip()
    @validator("title")
    def strip_title(cls, value: str) -> str: return value.strip()

class MatchRequest(BaseModel):
    job: JobRequest
    top_k: int = Field(10, ge=1, le=100)
    explain: bool = False
class SearchRequest(BaseModel):
    job: JobRequest
    top_k: int = Field(10, ge=1, le=100)
class CandidateMatch(BaseModel):
    candidate_id: str | None = None
    candidate_name: str
    rank: int | None = None
    faiss_similarity: float | None = None
    match_probability: float
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    explanation: str | None = None
    explanation_status: str | None = None
class Latency(BaseModel):
    total_ms: float
    embedding_ms: float | None = None
    faiss_ms: float | None = None
    xgboost_ms: float | None = None
    qwen_ms: float | None = None
class MatchResponse(BaseModel):
    job_title: str
    results: list[CandidateMatch]
    latency: Latency | None = None
class SearchCandidate(BaseModel):
    candidate_id: str
    candidate_name: str
    faiss_similarity: float
class SearchResponse(BaseModel):
    results: list[SearchCandidate]
    latency: Latency | None = None
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
class ReadyResponse(BaseModel):
    ready: bool
    core_ready: bool
    llm_available: bool
    components: dict[str, bool]
class ExplainCandidate(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_years: float | None = Field(None, ge=0, le=80)
class ExplainRequest(BaseModel):
    candidate: ExplainCandidate
    job: JobRequest
    match_probability: float = Field(..., ge=0, le=1)
class ExplainResponse(BaseModel):
    explanation: str | None
    explanation_status: str
