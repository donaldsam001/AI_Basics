"""Qwen3-4B GGUF wrapper for CPU-friendly explanation generation.

This module loads a quantised Qwen3-4B model via ``llama-cpp-python`` and
exposes a single ``generate`` method that converts a structured prompt into
a human-readable explanation.

The wrapper is intentionally limited to explanation generation.  It does not
perform candidate ranking, modify XGBoost scores, or make hiring decisions.

Configuration
-------------
All parameters can be controlled through environment variables or passed
directly.  Environment variables are only consulted when constructor
arguments are not supplied.

    QWEN_MODEL_PATH   – Path to the GGUF model file.
    QWEN_CONTEXT_SIZE – Context window size  (default: 8192).
    QWEN_THREADS      – CPU threads           (default: 4).
    QWEN_TEMPERATURE  – Sampling temperature  (default: 0.2).
    QWEN_MAX_TOKENS   – Maximum output tokens (default: 500).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "models/Qwen3-4B-Q4_K_M.gguf"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_THREADS = 4
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 500

_SYSTEM_PROMPT = (
    "You are a CV-job matching assistant. "
    "Only explain information provided in the evidence. "
    "Do not invent skills, experience, education, projects, certifications, "
    "technologies, achievements, or responsibilities. "
    "If information is unavailable, state that it is not provided. "
    "Do not make a hiring decision."
)


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


class QwenLLM:
    """Lazily-loaded Qwen3-4B GGUF wrapper using ``llama-cpp-python``.

    The model is loaded on the first call to :meth:`generate` so that the
    class can be instantiated cheaply and the heavy model is only loaded
    when an explanation is actually requested.

    Parameters
    ----------
    model_path:
        Filesystem path to the GGUF model file.  Falls back to the
        ``QWEN_MODEL_PATH`` environment variable, then to
        ``models/Qwen3-4B-Q4_K_M.gguf``.
    n_ctx:
        Context window size.  Falls back to ``QWEN_CONTEXT_SIZE``.
    n_threads:
        CPU thread count.  Falls back to ``QWEN_THREADS``.
    temperature:
        Sampling temperature.  Falls back to ``QWEN_TEMPERATURE``.
    max_tokens:
        Maximum tokens to generate.  Falls back to ``QWEN_MAX_TOKENS``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.n_ctx = n_ctx if n_ctx is not None else _env_int("QWEN_CONTEXT_SIZE", DEFAULT_CONTEXT_SIZE)
        self.n_threads = n_threads if n_threads is not None else _env_int("QWEN_THREADS", DEFAULT_THREADS)
        self.temperature = temperature if temperature is not None else _env_float("QWEN_TEMPERATURE", DEFAULT_TEMPERATURE)
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("QWEN_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        self._llm = None  # Lazy-loaded on first generate() call.

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the GGUF model from disk.

        Raises
        ------
        FileNotFoundError
            If the configured model path does not exist.
        RuntimeError
            If ``llama-cpp-python`` is not installed or the model fails
            to load.
        """
        resolved = Path(self.model_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Qwen model not found: {resolved}")

        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise RuntimeError(
                "llama-cpp-python is required for Qwen explanation generation. "
                "Install it with: pip install llama-cpp-python"
            ) from error

        try:
            self._llm = Llama(
                model_path=str(resolved),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load Qwen model from {resolved}: {error}"
            ) from error

        logger.info("Qwen model loaded from %s (ctx=%d, threads=%d)", resolved, self.n_ctx, self.n_threads)

    @property
    def is_loaded(self) -> bool:
        """Return whether the underlying model is currently loaded."""
        return self._llm is not None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate an explanation from a structured prompt.

        The model is loaded lazily on the first call.  Subsequent calls
        reuse the same model instance.

        Parameters
        ----------
        prompt:
            A fully formatted explanation prompt (see :mod:`src.llm.prompts`).

        Returns
        -------
        str
            The generated explanation text.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        RuntimeError
            If the model cannot be loaded or generation fails.
        """
        if self._llm is None:
            self._load_model()

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response["choices"][0]["message"]["content"]
        return content.strip() if content else ""
