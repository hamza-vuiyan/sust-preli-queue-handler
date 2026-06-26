"""
logic.py — compatibility shim
==============================
All pipeline logic has been moved to the `app.pipeline` package for
better organisation and debuggability:

    app/pipeline/
        client.py      — Gemini LLM client lifecycle
        extractor.py   — Step A: fact extraction (Gemini + stub)
        evidence.py    — Step B: deterministic rule engine
        classifier.py  — Step C-1: classify, route, severity
        drafter.py     — Step C-2: response templates
        runner.py      — run_pipeline() orchestrator

This file re-exports the public API so any code that still does
`from app.logic import ...` keeps working without modification.
"""

# Re-exports — keep in sync with app/pipeline/__init__.py
from app.pipeline.classifier import classify_case, compute_severity, route_department  # noqa: F401
from app.pipeline.client import get_llm_status, init_llm_client  # noqa: F401
from app.pipeline.drafter import draft_response  # noqa: F401
from app.pipeline.evidence import evaluate_evidence  # noqa: F401
from app.pipeline.extractor import call_llm_extractor  # noqa: F401
from app.pipeline.runner import run_pipeline  # noqa: F401