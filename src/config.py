"""Centralised, environment-aware pipeline configuration.

Every setting reads from its environment variable first, then falls back
to a sensible default.  The class can also be instantiated with explicit
keyword overrides for unit tests and programmatic use.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default %d", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default %s", name, value, default)
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass
class PipelineConfig:
    """Complete pipeline configuration with environment-variable fallbacks."""

    # --- Retrieval / ranking / explanation top-K -------------------------
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 100))
    ranking_top_k: int = field(default_factory=lambda: _env_int("RANKING_TOP_K", 20))
    explanation_top_k: int = field(default_factory=lambda: _env_int("EXPLANATION_TOP_K", 5))

    # --- XGBoost ---------------------------------------------------------
    xgboost_model_path: str = field(
        default_factory=lambda: _env_str("XGBOOST_MODEL_PATH", "data/models/cv_job_xgb.json"),
    )
    xgboost_features_path: str = field(
        default_factory=lambda: _env_str("XGBOOST_FEATURES_PATH", "data/models/xgb_features.json"),
    )

    # --- FAISS -----------------------------------------------------------
    faiss_index_path: str = field(
        default_factory=lambda: _env_str("FAISS_INDEX_PATH", "data/faiss/cv.index"),
    )
    faiss_metadata_path: str = field(
        default_factory=lambda: _env_str("FAISS_METADATA_PATH", "data/faiss/cv_metadata.json"),
    )

    # --- Qwen LLM -------------------------------------------------------
    qwen_model_path: str = field(
        default_factory=lambda: _env_str("QWEN_MODEL_PATH", "models/Qwen3-4B-Q4_K_M.gguf"),
    )
    qwen_context_size: int = field(default_factory=lambda: _env_int("QWEN_CONTEXT_SIZE", 8192))
    qwen_threads: int = field(default_factory=lambda: _env_int("QWEN_THREADS", 4))
    qwen_temperature: float = field(default_factory=lambda: _env_float("QWEN_TEMPERATURE", 0.2))
    qwen_max_tokens: int = field(default_factory=lambda: _env_int("QWEN_MAX_TOKENS", 500))

    # --- Embedding -------------------------------------------------------
    embedding_cache_dir: str = field(
        default_factory=lambda: _env_str("EMBEDDING_CACHE_DIR", "data/embeddings"),
    )

    # --- CV data ---------------------------------------------------------
    cv_path: str = field(
        default_factory=lambda: _env_str("CV_DATA_PATH", "data/preprocess/preprocessed_cvs.csv"),
    )

    # --- API -------------------------------------------------------------
    api_host: str = field(default_factory=lambda: _env_str("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env_str("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"))
    max_upload_size: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))
    api_top_k_default: int = field(default_factory=lambda: _env_int("TOP_K_DEFAULT", 10))
    api_top_k_max: int = field(default_factory=lambda: _env_int("TOP_K_MAX", 100))
    qwen_enabled: bool = field(default_factory=lambda: _env_bool("QWEN_ENABLED", False))

    # --- Supabase Database -----------------------------------------------
    supabase_url: str = field(
        default_factory=lambda: _env_str("SUPABASE_URL", ""),
    )
    supabase_key: str = field(
        default_factory=lambda: _env_str(
            "SUPABASE_KEY",
            _env_str("SUPABASE_SERVICE_ROLE_KEY", ""),
        ),
    )

