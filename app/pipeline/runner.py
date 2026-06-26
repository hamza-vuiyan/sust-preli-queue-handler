"""
pipeline/runner.py
==================
Pipeline orchestrator — wires Step A → B → C together.

`run_pipeline` is the only function called by `app.main`. Everything
else in this package is an implementation detail.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.schemas import CaseType, Department, Severity, Transaction
from app.pipeline.classifier import classify_case, compute_severity, route_department
from app.pipeline.drafter import draft_response
from app.pipeline.evidence import EvidenceVerdictResult, evaluate_evidence
from app.pipeline.extractor import call_llm_extractor

logger = logging.getLogger("queuestorm.pipeline.runner")


def run_pipeline(
    complaint_text: str,
    transaction_history: List[Transaction],
) -> Tuple[
    Dict[str, Any],       # extracted facts (Step A output)
    EvidenceVerdictResult,# evidence verdict (Step B output)
    CaseType,             # case classification (Step C)
    Department,           # department routing (Step C)
    Severity,             # severity score (Step C)
    Dict[str, str],       # drafted reply (Step C)
]:
    """Execute the full pipeline and return all intermediate results.

    Step A: call_llm_extractor  — extract facts from complaint text
    Step B: evaluate_evidence   — score transactions against facts
    Step C: classify + route + compute_severity + draft_response
    """
    extracted  = call_llm_extractor(complaint_text)
    evidence   = evaluate_evidence(extracted, transaction_history)
    case_type  = classify_case(extracted)
    department = route_department(case_type)
    severity   = compute_severity(case_type, evidence)
    drafted    = draft_response(extracted, evidence, case_type, severity)

    logger.debug(
        "Pipeline complete | case=%s dept=%s severity=%s verdict=%s",
        case_type, department, severity, evidence.evidence_verdict,
    )

    return extracted, evidence, case_type, department, severity, drafted
