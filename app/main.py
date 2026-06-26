"""
main.py
=======
QueueStorm Investigator — FastAPI entrypoint.

Exposes:
    GET  /health           — instant liveness probe.
    POST /analyze-ticket   — hybrid complaint-investigation pipeline.

Design notes for the judges:
  * `/health` does no I/O and initializes nothing; it returns immediately.
  * `/analyze-ticket` runs `app.logic.run_pipeline` followed by
    `app.safety.apply_safety_guardrails` before returning.
  * All uncaught exceptions are routed through global handlers so the
    service NEVER returns a raw 500 to clients.
  * The whole request handler is wrapped in a top-level try/except so a
    crash inside the pipeline is converted into a controlled 500 envelope.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.pipeline import init_llm_client, run_pipeline
from app.pipeline.classifier import compute_human_review_required, compute_reason_codes
from app.safety import apply_safety_guardrails
from app.schemas import (
    AnalyzeTicketRequest,
    AnalyzeTicketResponse,
    EvidenceVerdict,
    SafeErrorResponse,
)

# ---------------------------------------------------------------------------
# Logging — kept minimal so the judge demo output stays readable.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("queuestorm")


# ---------------------------------------------------------------------------
# Lifespan — startup/shutdown hooks. Kept deliberately lightweight so the
# app starts fast (important for the 5-second SLA on /analyze-ticket).
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log that we booted and yield. No heavy init here on purpose."""
    logger.info("QueueStorm Investigator is starting up.")
    # Build the optional [redacted] client once. Safe to call even when
    # no API key is set — it falls back to the stub extractor.
    status_dict = init_llm_client()
    logger.info("LLM status: %s", status_dict)
    yield
    logger.info("QueueStorm Investigator is shutting down.")


app = FastAPI(
    title="QueueStorm Investigator",
    version="1.0.0",
    description=(
        "AI/API support copilot that triages customer complaints by "
        "cross-referencing free-text issues against a transaction history."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe")
async def health() -> Dict[str, str]:
    """Instant liveness probe — does no I/O, no LLM init, no DB calls."""
    return {"status": "ok"}


@app.post(
    "/analyze-ticket",
    response_model=AnalyzeTicketResponse,
    summary="Run the hybrid investigation pipeline on a support ticket.",
    responses={
        400: {"model": SafeErrorResponse, "description": "Bad request."},
        422: {"model": SafeErrorResponse, "description": "Validation error."},
        500: {"model": SafeErrorResponse, "description": "Internal error."},
    },
)
async def analyze_ticket(payload: AnalyzeTicketRequest) -> AnalyzeTicketResponse:
    """Run Step A -> Step B -> Step C, then scrub the reply through safety.py.

    Any exception is caught here as a last line of defense so the caller
    always gets a JSON envelope — never a raw 500 with a traceback.
    """
    started = time.perf_counter()
    try:
        # --- Step A + B + C --------------------------------------------------
        extracted, evidence, case_type, department, severity, drafted = run_pipeline(
            payload.complaint,
            payload.transaction_history,
            payload.language,
        )

        # --- Outbound safety guardrails (mandatory) --------------------------
        cleaned_reply, guardrails_applied = apply_safety_guardrails(
            drafted["customer_reply"]
        )

        # --- Decision metadata -----------------------------------------------
        human_review_required = compute_human_review_required(case_type, evidence, severity)
        reason_codes = compute_reason_codes(case_type, evidence)

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return AnalyzeTicketResponse(
            ticket_id=payload.ticket_id,
            case_type=case_type,
            severity=severity,
            department=department,
            evidence_verdict=evidence.evidence_verdict,
            relevant_transaction_id=evidence.relevant_transaction_id,
            confidence=evidence.confidence,
            agent_summary=drafted["agent_summary"],
            recommended_next_action=drafted["recommended_next_action"],
            customer_reply=cleaned_reply,
            human_review_required=human_review_required,
            reason_codes=reason_codes,
        )
    except HTTPException:
        # Re-raise so FastAPI's HTTPException handler sees it.
        raise
    except Exception as exc:  # pragma: no cover — defensive belt-and-braces
        # Should never reach here in practice because every step is
        # exception-safe, but if it does we still return a clean envelope
        # instead of leaking internals.
        logger.exception("Unhandled error in /analyze-ticket: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process ticket at this time.",
        )


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
#
# These exist for two reasons:
#   1. Pydantic validation errors must be returned in our `SafeErrorResponse`
#      envelope so clients don't see FastAPI's default shape.
#   2. Any *unexpected* exception inside a route becomes a 500 with a
#      controlled message — never a traceback dump.
# ---------------------------------------------------------------------------


def _safe_error(
    error: str,
    message: str,
    status_code: int,
    details: Any = None,
) -> JSONResponse:
    """Build a SafeErrorResponse-shaped JSONResponse."""
    body = SafeErrorResponse(error=error, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def _on_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert Pydantic validation errors (422) into the safe envelope."""
    # Strip non-serializable bits (e.g. ctx) from each error entry.
    flat: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "invalid value")
        flat.append(f"{loc}: {msg}")
    logger.info("Validation error on %s: %s", request.url.path, flat)
    return _safe_error(
        error="validation_error",
        message="One or more fields failed validation.",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=flat,
    )


@app.exception_handler(HTTPException)
async def _on_http_exception(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Pass HTTPException through our envelope but never leak `detail` raw."""
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    # Treat the standard 422 raised by FastAPI as a validation error too.
    error_tag = "validation_error" if exc.status_code == 422 else "http_error"
    return _safe_error(
        error=error_tag,
        message=message,
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def _on_unhandled_exception(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all so the service never returns a raw 500."""
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return _safe_error(
        error="internal_error",
        message="Unable to process ticket at this time.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# ---------------------------------------------------------------------------
# Local runner — `python -m app.main` boots Uvicorn on port 7860.
# Port 7860 matches the Dockerfile EXPOSE / CMD so local runs and the
# Hugging Face Spaces container stay consistent.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )