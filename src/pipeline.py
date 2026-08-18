"""Complete CV–Job matching pipeline orchestrator.

Integrates every stage of the system in order:

    Embedding → FAISS → Rule-based Matcher → XGBoost → Qwen Explanation

The pipeline is deliberately designed so that:

* XGBoost alone determines candidate ranking.
* Qwen only explains the top-K ranked candidates.
* Qwen failures never break the ranking output.
* The baseline deterministic score is always preserved for comparison.
* All heavy models are injected once and reused across requests.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import PipelineConfig
from src.embeddings import EmbeddingCache, EmbeddingModel
from src.features import build_cv_jd_features
from src.llm.evidence import build_evidence
from src.llm.prompts import build_matching_explanation_prompt
from src.llm.qwen import QwenLLM
from src.matching.matcher import CVJobMatcher
from src.models.xgboost_model import XGBCandidateScorer
from src.preprocessing import build_cv_embedding_text, build_job_embedding_text

logger = logging.getLogger(__name__)


class MatchingPipeline:
    """End-to-end candidate matching with configurable top-K at every stage.

    Parameters
    ----------
    matcher:
        Reusable :class:`CVJobMatcher` instance (embedding model inside).
    xgb_scorer:
        A loaded :class:`XGBCandidateScorer`, or ``None`` to fall back to
        the deterministic baseline.
    qwen:
        A :class:`QwenLLM` instance, or ``None`` to skip explanation
        generation.
    config:
        A :class:`PipelineConfig` with top-K thresholds and paths.
    """

    def __init__(
        self,
        matcher: CVJobMatcher,
        xgb_scorer: XGBCandidateScorer | None = None,
        qwen: QwenLLM | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.matcher = matcher
        self.xgb_scorer = xgb_scorer
        self.qwen = qwen
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def rank_and_explain(
        self,
        cvs: list[dict[str, Any]],
        job: dict[str, Any],
        vector_store: Any | None = None,
        metadata_store: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Run the complete pipeline: retrieve → match → rank → explain.

        Parameters
        ----------
        cvs:
            All loaded and parsed CV dictionaries.
        job:
            A parsed job description dictionary.
        vector_store:
            Optional :class:`FAISSStore`; when supplied together with
            *metadata_store*, FAISS retrieval limits the candidate pool.
        metadata_store:
            Optional :class:`MetadataStore` for FAISS.

        Returns
        -------
        list[dict]
            Ranked candidate results.  The top *explanation_top_k*
            candidates include an ``"explanation"`` key.
        """
        # Phase 1 + 2: Retrieve candidates (FAISS) and generate matcher
        # features with optional XGBoost reranking.
        if vector_store is not None and metadata_store is not None:
            results = self.matcher.rank_retrieved_candidates(
                cvs, job, vector_store, metadata_store,
                k=self.config.retrieval_top_k,
                xgb_scorer=self.xgb_scorer,
            )
        else:
            results = self.matcher.rank_candidates(cvs, job)

        # Phase 3: Trim to ranking_top_k.
        results = results[: self.config.ranking_top_k]

        # Phase 4: Assign explicit ranks after sorting.
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        # Phase 5: Generate explanations for the top explanation_top_k
        # candidates.  This is an optional, non-critical step.
        self._generate_explanations(results, cvs, job)

        return results

    # ------------------------------------------------------------------
    # Explanation generation (optional layer)
    # ------------------------------------------------------------------

    def _generate_explanations(
        self,
        results: list[dict[str, Any]],
        cvs: list[dict[str, Any]],
        job: dict[str, Any],
    ) -> None:
        """Add Qwen explanations to the top-K results in place.

        If *qwen* is ``None`` or a generation attempt fails, the ranking
        result is not affected.  Failed explanations set
        ``explanation=None`` and ``explanation_error=<message>``.
        """
        if self.qwen is None:
            return

        top_k = min(self.config.explanation_top_k, len(results))
        for result in results[:top_k]:
            candidate = self._find_candidate(result, cvs)
            evidence = build_evidence(
                candidate=candidate,
                job=job,
                match_result=result,
                rank=result.get("rank", 0),
            )
            prompt = build_matching_explanation_prompt(evidence)
            try:
                result["explanation"] = self.qwen.generate(prompt)
                result.pop("explanation_error", None)
            except Exception as error:
                logger.warning(
                    "Qwen explanation failed for %s: %s",
                    result.get("candidate_name", "?"), error,
                )
                result["explanation"] = None
                result["explanation_error"] = str(error)

    @staticmethod
    def _find_candidate(
        result: dict[str, Any],
        cvs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Look up the original CV dict by candidate name."""
        name = result.get("candidate_name", "")
        return next(
            (cv for cv in cvs if cv.get("candidate_name") == name),
            cvs[0] if cvs else {},
        )

    # ------------------------------------------------------------------
    # API-friendly response builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_api_response(
        job: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Flatten ranking results into a clean JSON-serialisable response.

        The response structure matches the API contract defined in
        ``src/api/app.py``.
        """
        candidates = []
        for result in results:
            entry: dict[str, Any] = {
                "candidate_name": result.get("candidate_name", "Unknown"),
                "rank": result.get("rank"),
                "match_probability": result.get(
                    "xgb_probability",
                    result.get("final_score", 0.0),
                ),
                "retrieval_score": result.get("faiss_similarity"),
                "matched_skills": result.get("matched_skills", []),
                "missing_skills": result.get("missing_required_skills", []),
                "matched_preferred_skills": result.get("matched_preferred_skills", []),
                "semantic_score": result.get("semantic_score"),
                "required_skill_score": result.get("required_skill_score"),
                "preferred_skill_score": result.get("preferred_skill_score"),
                "experience_score": result.get("experience_score"),
                "baseline_score": result.get("deterministic_score", result.get("final_score")),
            }
            if "explanation" in result:
                entry["explanation"] = result["explanation"]
            if "explanation_error" in result:
                entry["explanation_error"] = result["explanation_error"]
            candidates.append(entry)
        return {
            "job_title": job.get("job_title", "Untitled Job"),
            "candidates": candidates,
        }


# ======================================================================
# Convenience factory
# ======================================================================


def create_pipeline(config: PipelineConfig | None = None) -> MatchingPipeline:
    """Instantiate a pipeline with default or configured components.

    Components that fail to load (e.g. missing model files) log warnings
    and are omitted; the pipeline degrades gracefully.
    """
    config = config or PipelineConfig()

    # Embedding model and matcher.
    embedding_cache = EmbeddingCache(config.embedding_cache_dir)
    matcher = CVJobMatcher(embedding_cache=embedding_cache)

    # XGBoost scorer (optional).
    xgb_scorer = None
    try:
        xgb_scorer = XGBCandidateScorer.load(
            config.xgboost_model_path,
            config.xgboost_features_path,
        )
        logger.info("XGBoost scorer loaded from %s", config.xgboost_model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.warning("XGBoost scorer unavailable: %s", error)

    # Qwen LLM (optional, lazy-loaded on first explanation request).
    qwen: QwenLLM | None = None
    try:
        qwen = QwenLLM(
            model_path=config.qwen_model_path,
            n_ctx=config.qwen_context_size,
            n_threads=config.qwen_threads,
            temperature=config.qwen_temperature,
            max_tokens=config.qwen_max_tokens,
        )
        logger.info("Qwen LLM configured (lazy-loaded) from %s", config.qwen_model_path)
    except Exception as error:
        logger.warning("Qwen LLM unavailable: %s", error)

    return MatchingPipeline(
        matcher=matcher,
        xgb_scorer=xgb_scorer,
        qwen=qwen,
        config=config,
    )
