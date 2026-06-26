"""
pipeline/client.py
==================
Manages the optional Google Gemini LLM client singleton.

`init_llm_client()` is called once from `app.main`'s lifespan hook.
Other modules access the singleton via `get_llm_client()` at call-time
(not at import-time) so the module can be imported before the client is
initialised without issue.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger("queuestorm.pipeline.client")

# ---------------------------------------------------------------------------
# Load .env early so os.getenv picks up the values.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=False)
except Exception:
    pass

_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "stub").strip().lower() or "stub"
_GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
_GEMINI_MODEL: str = (
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
)

_llm_client: Any = None  # set by init_llm_client()


# ---------------------------------------------------------------------------
# Accessors — always call these at runtime, never read the module globals
# from other modules directly (the client is set lazily).
# ---------------------------------------------------------------------------


def get_llm_client() -> Any:
    """Return the current LLM client singleton (may be None if not ready)."""
    return _llm_client


def get_gemini_model() -> str:
    """Return the configured Gemini model name."""
    return _GEMINI_MODEL


# ---------------------------------------------------------------------------
# Initialisation (called once from lifespan)
# ---------------------------------------------------------------------------


def init_llm_client() -> Dict[str, Any]:
    """Initialise the optional Gemini client.

    Safe to call multiple times. Returns a small status dict.

    Returns:
        dict with keys:
          - provider: "stub" | "gemini"
          - model:   model name in use, or "n/a" for stub
          - ready:   True if a real Gemini call would be attempted
    """
    global _llm_client

    if _LLM_PROVIDER != "gemini":
        logger.info("LLM provider is 'stub' (default). Set LLM_PROVIDER=gemini to enable.")
        return {"provider": "stub", "model": "n/a", "ready": False}

    if not _GEMINI_API_KEY:
        logger.warning(
            "LLM_PROVIDER=gemini but GEMINI_API_KEY is empty — falling back to stub extractor."
        )
        return {"provider": "stub", "model": "n/a", "ready": False}

    try:
        from google import genai  # type: ignore

        _llm_client = genai.Client(api_key=_GEMINI_API_KEY)
        logger.info("[redacted] client initialised with model=%s", _GEMINI_MODEL)
        return {"provider": "gemini", "model": _GEMINI_MODEL, "ready": True}
    except Exception as exc:
        logger.warning(
            "Failed to initialise [redacted] client (%s) — falling back to stub.", exc
        )
        _llm_client = None
        return {"provider": "stub", "model": "n/a", "ready": False}


def get_llm_status() -> Dict[str, Any]:
    """Return the current LLM provider status (read-only)."""
    return {
        "provider": _LLM_PROVIDER if _llm_client is not None else "stub",
        "model": _GEMINI_MODEL if _llm_client is not None else "n/a",
        "ready": _llm_client is not None,
    }
