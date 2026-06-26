"""
pipeline/__init__.py
====================
Public API for the pipeline package.

Exposes the symbols that app.main needs:
    from app.pipeline import init_llm_client, run_pipeline
"""

from app.pipeline.client import get_llm_status, init_llm_client  # noqa: F401
from app.pipeline.runner import run_pipeline  # noqa: F401

__all__ = ["init_llm_client", "get_llm_status", "run_pipeline"]
