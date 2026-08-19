"""CV-Job Intelligent Matching System Architecture Implementation.

Pipeline Architecture Flow:
1. PDF/DOCX Document Extraction
2. Resume Parser
3. Semantic Embeddings
4. FAISS Vector Search
5. Top-K Candidate Retrieval
6. Feature Engineering
7. XGBoost Match Prediction
8. Qwen3-4B RAG Explanation
9. Candidate Ranking + Skill Gap Analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import PipelineConfig
from src.embeddings import EmbeddingCache, EmbeddingModel
from src.models.feature_builder import build_pair_features
from src.llm.evidence import build_evidence
from src.llm.prompts import build_matching_explanation_prompt
from src.llm.qwen import QwenLLM
from src.matching.matcher import CVJobMatcher
from src.matching.similarity import calculate_skill_match
from src.models.xgboost_model import XGBCandidateScorer
from src.preprocessing import preprocess_cv, preprocess_job
from src.vector_store import FAISSStore, MetadataStore

logger = logging.getLogger(__name__)


# ======================================================================
# Step 1: PDF/DOCX Document Extraction
# ======================================================================

def extract_document_text(file_path: str | Path) -> str:
    """Step 1: Extract text from PDF, DOCX, or plain text file.

    Parameters
    ----------
    file_path:
        Path to input document (.pdf, .docx, or text file).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if text.strip():
                    return text
        except Exception:
            pass
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext in (".docx", ".doc"):
        import docx
        doc = docx.Document(path)
        return "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())

    else:
        return path.read_text(encoding="utf-8", errors="ignore")


# ======================================================================
# Step 2: Resume Parser
# ======================================================================

def parse_resume(file_path_or_text: str | Path) -> dict[str, Any]:
    """Step 2: Resume Parser - Convert raw resume text into structured fields.

    Extracts candidate name, contact, skills, education, experience years,
    current title, projects, and certifications.
    """
    path = Path(file_path_or_text)
    if path.is_file():
        raw_text = extract_document_text(path)
        file_name = path.name
        candidate_name_fallback = path.stem.replace("_", " ").title()
    else:
        raw_text = str(file_path_or_text)
        file_name = "raw_text_input"
        candidate_name_fallback = "Candidate"

    parsed = preprocess_cv(raw_text)
    parsed["raw_text"] = raw_text
    parsed["file_name"] = file_name

    cand_name = parsed.get("candidate_name")
    if not cand_name or cand_name.casefold() in ("unknown candidate", "experience year", "experience years"):
        parsed["candidate_name"] = candidate_name_fallback

    return parsed


# ======================================================================
# End-to-End System Class (Steps 3 - 9)
# ======================================================================

class CVJobIntelligentMatchingSystem:
    """9-Step Intelligent CV-Job Matching Engine."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        xgb_scorer: XGBCandidateScorer | None = None,
        qwen_llm: QwenLLM | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.matcher = CVJobMatcher(
            embedding_model=self.embedding_model,
            embedding_cache=EmbeddingCache(self.config.embedding_cache_dir),
        )
        self.xgb_scorer = xgb_scorer
        if self.xgb_scorer is None:
            try:
                pipeline_path = Path(self.config.xgboost_model_path).parent / "xgb_pipeline.joblib"
                schema_path = Path(self.config.xgboost_model_path).parent / "xgb_schema.json"
                if pipeline_path.is_file():
                    self.xgb_scorer = XGBCandidateScorer.load_pipeline(pipeline_path, schema_path)
                else:
                    self.xgb_scorer = XGBCandidateScorer.load(
                        self.config.xgboost_model_path,
                        self.config.xgboost_features_path,
                    )
            except Exception as err:
                logger.warning("XGBoost scorer unavailable: %s", err)


        self.qwen_llm = qwen_llm
        if self.qwen_llm is None:
            try:
                self.qwen_llm = QwenLLM(
                    model_path=self.config.qwen_model_path,
                    n_ctx=self.config.qwen_context_size,
                    n_threads=self.config.qwen_threads,
                    temperature=self.config.qwen_temperature,
                    max_tokens=self.config.qwen_max_tokens,
                )
            except Exception as err:
                logger.warning("Qwen LLM unavailable: %s", err)

    # ------------------------------------------------------------------
    # Step 3: Semantic Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Step 3: Generate dense 768-dim normalized embedding vectors."""
        return self.embedding_model.encode(texts)

    # ------------------------------------------------------------------
    # Step 4: FAISS Vector Search Index Construction
    # ------------------------------------------------------------------

    def create_faiss_index(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[FAISSStore, list[dict[str, Any]]]:
        """Step 4: Build FAISS vector store and positional metadata."""
        texts = [cv.get("raw_text", str(cv)) for cv in candidates]
        embeddings = self.compute_embeddings(texts)

        dimension = embeddings.shape[1] if len(embeddings.shape) > 1 else 768
        store = FAISSStore(dimension=dimension)
        store.add(embeddings)

        metadata_records = []
        for faiss_id, cv in enumerate(candidates):
            metadata_records.append({
                "faiss_id": faiss_id,
                "candidate_name": cv.get("candidate_name", f"Candidate_{faiss_id}"),
                "skills": cv.get("technical_skills", []) + cv.get("programming_languages", []),
                "years_experience": cv.get("years_experience", 0),
            })

        return store, metadata_records

    # ------------------------------------------------------------------
    # Steps 5-9: Full Pipeline Execution
    # ------------------------------------------------------------------

    def process_matching(
        self,
        candidate_files_or_dicts: list[str | Path | dict[str, Any]],
        job_description_or_dict: str | Path | dict[str, Any],
        top_k_retrieval: int = 50,
        top_k_explain: int = 5,
    ) -> dict[str, Any]:
        """Run the complete 9-step pipeline.

        Returns structured ranking, XGBoost match predictions, skill gap analysis,
        and Qwen3-4B explanations.
        """
        # Step 1 & 2: Parse Candidates
        cvs: list[dict[str, Any]] = []
        for item in candidate_files_or_dicts:
            if isinstance(item, dict):
                cvs.append(item)
            else:
                cvs.append(parse_resume(item))

        # Step 1 & 2: Parse Job Description
        if isinstance(job_description_or_dict, dict):
            job = job_description_or_dict
        else:
            path = Path(job_description_or_dict)
            if path.is_file():
                raw_job = extract_document_text(path)
                job_title = path.stem.replace("_", " ").title()
            else:
                raw_job = str(job_description_or_dict)
                job_title = "Target Position"
            job = preprocess_job(raw_job, job_title)

        # Step 3: Semantic Embeddings
        cv_texts = [cv.get("raw_text", str(cv)) for cv in cvs]
        job_text = job.get("job_description", str(job))

        cv_embeddings = self.compute_embeddings(cv_texts)
        job_embedding = self.compute_embeddings([job_text])[0]

        # Step 4: FAISS Vector Index
        store = FAISSStore(dimension=cv_embeddings.shape[1])
        store.add(cv_embeddings)

        # Step 5: Top-K Candidate Retrieval
        scores, retrieved_ids = store.search(job_embedding, k=min(top_k_retrieval, len(cvs)))

        # Step 6 & 7: Feature Engineering & XGBoost Match Prediction
        ranked_candidates = []
        for faiss_id, faiss_sim in zip(retrieved_ids, scores):
            if faiss_id < 0 or faiss_id >= len(cvs):
                continue
            cv = cvs[faiss_id]
            cv_emb = cv_embeddings[faiss_id]

            # Matching features
            match_res = self.matcher.match(cv, job, cv_emb, job_embedding)
            match_res["faiss_similarity"] = float(faiss_sim)

            # Step 6: Feature Engineering — using CANONICAL feature builder
            # (same function as training → no schema drift)
            semantic_sim = max(0.0, min(1.0, float(match_res.get("semantic_score", 0.0)) / 100.0))
            pair_features = build_pair_features(
                cv, job,
                semantic_similarity=semantic_sim,
                faiss_similarity=float(faiss_sim),
            )

            # Step 7: XGBoost Prediction
            if self.xgb_scorer is not None:
                prob = self.xgb_scorer.predict_probabilities([pair_features])[0]
                match_res["xgb_probability"] = float(prob)

            ranked_candidates.append((match_res, cv))

        # Step 9: Candidate Ranking & Skill Gap Analysis
        # Sort by XGBoost probability if available, otherwise final deterministic score
        ranked_candidates.sort(
            key=lambda item: item[0].get("xgb_probability", item[0].get("final_score", 0.0)),
            reverse=True,
        )

        final_results = []
        for rank, (res, cv) in enumerate(ranked_candidates, start=1):
            res["rank"] = rank

            # Skill Gap Analysis
            candidate_skills = cv.get("technical_skills", []) + cv.get("programming_languages", [])
            required_skills = job.get("required_skills", [])
            skill_gap = calculate_skill_match(candidate_skills, required_skills)

            res["skill_gap_analysis"] = {
                "matched_skills": skill_gap["matched"],
                "missing_skills": skill_gap["missing"],
                "matched_count": len(skill_gap["matched"]),
                "missing_count": len(skill_gap["missing"]),
                "skill_match_score": skill_gap["score"],
            }

            # Step 8: Qwen3-4B RAG Explanation (Top-K candidates)
            if rank <= top_k_explain and self.qwen_llm is not None:
                evidence = build_evidence(candidate=cv, job=job, match_result=res, rank=rank)
                prompt = build_matching_explanation_prompt(evidence)
                try:
                    res["qwen_explanation"] = self.qwen_llm.generate(prompt)
                except Exception as err:
                    logger.warning("Qwen explanation failed for rank %d: %s", rank, err)
                    res["qwen_explanation"] = None
                    res["qwen_explanation_error"] = str(err)

            final_results.append(res)

        return {
            "job_title": job.get("job_title", "Untitled Job"),
            "total_candidates": len(cvs),
            "retrieved_candidates": len(final_results),
            "rankings": final_results,
        }
