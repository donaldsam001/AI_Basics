"""Service boundary between FastAPI and the existing matching components.

This module intentionally contains no HTTP types.  It owns model lifecycle and
coordinates preprocessing, FAISS retrieval, canonical XGBoost features, and
the evidence-only Qwen explanation layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from src.config import PipelineConfig
from src.llm.evidence import build_evidence
from src.llm.prompts import build_matching_explanation_prompt
from src.main import load_cvs
from src.models.feature_builder import PAIR_FEATURE_NAMES, build_pair_features
from src.pipeline import create_pipeline
from src.preprocessing import preprocess_cv, preprocess_job
from src.vector_store import FAISSStore, MetadataStore

logger = logging.getLogger(__name__)


class ServiceUnavailableError(RuntimeError):
    """Raised when a core model artifact is unavailable at runtime."""


class ModelRegistry:
    """Load runtime artifacts once and expose them through explicit fields."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.pipeline = None
        self.cvs: list[dict[str, Any]] = []
        self.faiss_store: FAISSStore | None = None
        self.metadata_store: MetadataStore | None = None
        self.errors: dict[str, str] = {}

    def load(self) -> None:
        self.pipeline = create_pipeline(self.config)
        try:
            self.cvs = load_cvs(self.config.cv_path)
        except Exception as exc:  # configuration/data issue, not client input
            self.errors["candidates"] = str(exc)
            logger.warning("Candidate corpus unavailable: %s", exc)
        try:
            self.faiss_store = FAISSStore.load(self.config.faiss_index_path)
            self.metadata_store = MetadataStore(self.config.faiss_metadata_path)
            self.metadata_store.validate_index_size(self.faiss_store.size)
        except Exception as exc:
            self.errors["faiss"] = str(exc)
            self.faiss_store = None
            self.metadata_store = None
            logger.warning("FAISS runtime unavailable: %s", exc)
        scorer = self.pipeline.xgb_scorer
        if scorer is None:
            self.errors["xgboost"] = "XGBoost model or schema could not be loaded"
        elif scorer.feature_names != PAIR_FEATURE_NAMES:
            self.errors["xgboost"] = "XGBoost schema does not match the canonical pair-feature contract"
            self.pipeline.xgb_scorer = None

    @property
    def embedding_model(self):
        return self.pipeline.matcher.embedding_model if self.pipeline else None

    @property
    def xgb_model(self):
        return self.pipeline.xgb_scorer if self.pipeline else None

    @property
    def qwen_model(self):
        return self.pipeline.qwen if self.pipeline else None

    def readiness(self) -> dict[str, Any]:
        # Embedding is lazy to avoid a network/model download during readiness.
        # Its wrapper is nevertheless initialized once at application startup.
        embedding = self.embedding_model is not None
        faiss = self.faiss_store is not None and self.metadata_store is not None
        xgboost = self.xgb_model is not None
        qwen = bool(self.config.qwen_enabled and self.qwen_model is not None and Path(self.config.qwen_model_path).is_file())
        core_ready = embedding and faiss and xgboost and bool(self.cvs)
        return {
            "ready": core_ready,
            "core_ready": core_ready,
            "llm_available": qwen,
            "components": {"embedding_model": embedding, "faiss": faiss, "xgboost": xgboost, "metadata": faiss, "qwen": qwen},
        }


class MatchingService:
    """Thin business API over the repository's existing ML pipeline."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    @staticmethod
    def build_job(payload: dict[str, Any]) -> dict[str, Any]:
        job = preprocess_job(payload.get("description", ""), payload.get("title", ""))
        job["required_skills"] = list(payload.get("required_skills") or [])
        job["preferred_skills"] = list(payload.get("preferred_skills") or [])
        if payload.get("min_experience_years") is not None:
            job["years_of_experience"] = float(payload["min_experience_years"])
        return job

    def _require_core(self) -> None:
        status = self.registry.readiness()
        if not status["core_ready"]:
            raise ServiceUnavailableError("Matching models are not ready")

    @staticmethod
    def _latency(start: float, **parts: float) -> dict[str, float]:
        output = {name: round(value * 1000, 2) for name, value in parts.items()}
        output["total_ms"] = round((perf_counter() - start) * 1000, 2)
        return output

    def search(self, job: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        self._require_core()
        start = perf_counter()
        embedding_start = perf_counter()
        vector = self.registry.pipeline.matcher.encode_job(job)
        embedding_elapsed = perf_counter() - embedding_start
        faiss_start = perf_counter()
        scores, ids = self.registry.faiss_store.search(vector, top_k)
        rows = []
        for score, faiss_id in zip(scores, ids, strict=True):
            metadata = self.registry.metadata_store.get_candidate(int(faiss_id))
            source = self.registry.cvs[metadata["source_index"]]
            rows.append({"candidate_id": source.get("candidate_id") or str(faiss_id), "candidate_name": source.get("candidate_name") or "Unknown Candidate", "faiss_similarity": round(float(score), 6)})
        return rows, self._latency(start, embedding_ms=embedding_elapsed, faiss_ms=perf_counter() - faiss_start)

    def match(self, job: dict[str, Any], top_k: int, explain: bool) -> tuple[list[dict[str, Any]], dict[str, float]]:
        self._require_core()
        start = perf_counter()
        old = self.registry.pipeline.config
        # Do not mutate shared configuration based on a request.
        results = self.registry.pipeline.matcher.rank_retrieved_candidates(
            self.registry.cvs, job, self.registry.faiss_store, self.registry.metadata_store,
            k=top_k, xgb_scorer=self.registry.xgb_model,
        )[:top_k]
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
            result["match_probability"] = result["xgb_probability"]
        qwen_elapsed = 0.0
        if explain:
            qwen_start = perf_counter()
            self._explain_results(results, job)
            qwen_elapsed = perf_counter() - qwen_start
        return results, self._latency(start, qwen_ms=qwen_elapsed)

    def evaluate_cv(self, raw_cv: str, job: dict[str, Any], explain: bool) -> tuple[dict[str, Any], dict[str, float]]:
        if not raw_cv.strip():
            raise ValueError("CV contains no extractable text")
        self._require_core()
        start = perf_counter()
        cv = preprocess_cv(raw_cv)
        cv["raw_text"] = raw_cv
        cv["candidate_id"] = "uploaded-cv"
        result = self.registry.pipeline.matcher.match(cv, job)
        semantic = max(0.0, min(1.0, float(result["semantic_score"]) / 100.0))
        features = build_pair_features(cv, job, semantic_similarity=semantic, faiss_similarity=semantic)
        result["match_probability"] = round(float(self.registry.xgb_model.predict_probabilities([features])[0]), 6)
        result["features"] = features
        if explain:
            self._explain_one(result, cv, job)
        return result, self._latency(start)

    def explain_evidence(self, candidate: dict[str, Any], job: dict[str, Any], probability: float) -> tuple[str | None, str]:
        if not self.registry.config.qwen_enabled or not self.registry.readiness()["llm_available"]:
            return None, "unavailable"
        result = {"candidate_name": candidate.get("name", "Unknown Candidate"), "matched_skills": candidate.get("matched_skills", []), "missing_required_skills": candidate.get("missing_skills", []), "xgb_probability": probability}
        try:
            evidence = build_evidence(candidate, job, result)
            return self.registry.qwen_model.generate(build_matching_explanation_prompt(evidence)), "available"
        except Exception as exc:
            logger.warning("Qwen explanation failed: %s", exc)
            return None, "unavailable"

    def _explain_results(self, results: list[dict[str, Any]], job: dict[str, Any]) -> None:
        for result in results:
            cv = next((item for item in self.registry.cvs if item.get("candidate_id") == result.get("candidate_id")), {})
            self._explain_one(result, cv, job)

    def _explain_one(self, result: dict[str, Any], cv: dict[str, Any], job: dict[str, Any]) -> None:
        result["explanation"], result["explanation_status"] = self.explain_evidence(
            cv, job, float(result.get("match_probability", result.get("xgb_probability", 0.0)))
        )
