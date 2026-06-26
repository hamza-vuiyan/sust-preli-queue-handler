"""
pipeline/classifier.py
======================
Step C (part 1) — Case classification, department routing, and severity scoring.

All logic is pure Python lookups and comparisons — no I/O or LLM involved.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.schemas import CaseType, Department, EvidenceVerdict, Severity
from app.pipeline.evidence import EvidenceVerdictResult

logger = logging.getLogger("queuestorm.pipeline.classifier")

# ---------------------------------------------------------------------------
# Static routing tables — per the API spec.
# ---------------------------------------------------------------------------

_CORE_ISSUE_TO_CASE: Dict[str, CaseType] = {
    "wrong_transfer":                 CaseType.WRONG_TRANSFER,
    "payment_failed":                 CaseType.PAYMENT_FAILED,
    "refund_request":                 CaseType.REFUND_REQUEST,
    "duplicate_payment":              CaseType.DUPLICATE_PAYMENT,
    "merchant_settlement_delay":      CaseType.MERCHANT_SETTLEMENT_DELAY,
    "agent_cash_in_issue":            CaseType.AGENT_CASH_IN_ISSUE,
    "phishing_or_social_engineering": CaseType.PHISHING_OR_SOCIAL_ENGINEERING,
    "other":                          CaseType.OTHER,
}

_CASE_TO_DEPARTMENT: Dict[CaseType, Department] = {
    CaseType.PHISHING_OR_SOCIAL_ENGINEERING: Department.FRAUD_RISK,
    CaseType.WRONG_TRANSFER:                 Department.DISPUTE_RESOLUTION,
    CaseType.DUPLICATE_PAYMENT:              Department.DISPUTE_RESOLUTION,
    CaseType.PAYMENT_FAILED:                 Department.PAYMENTS_OPS,
    CaseType.MERCHANT_SETTLEMENT_DELAY:      Department.MERCHANT_OPERATIONS,
    CaseType.AGENT_CASH_IN_ISSUE:            Department.AGENT_OPERATIONS,
    CaseType.REFUND_REQUEST:                 Department.CUSTOMER_SUPPORT,
    CaseType.OTHER:                          Department.CUSTOMER_SUPPORT,
}


# ===========================================================================
# Public API
# ===========================================================================


def classify_case(extracted: Dict[str, Any]) -> CaseType:
    """Map the extracted core_issue string to the public CaseType enum."""
    return _CORE_ISSUE_TO_CASE.get(
        extracted.get("core_issue", "other"),
        CaseType.OTHER,
    )


def route_department(case_type: CaseType) -> Department:
    """Return the department that should own this case."""
    return _CASE_TO_DEPARTMENT.get(case_type, Department.CUSTOMER_SUPPORT)


def compute_severity(case_type: CaseType, evidence: EvidenceVerdictResult) -> Severity:
    """Derive a severity score from the case type and evidence verdict.

    Rules (in priority order):
      1. Phishing                           → always CRITICAL
      2. wrong_transfer + INCONSISTENT      → MEDIUM
      3. INCONSISTENT evidence              → escalate base severity one tier
      4. wrong_transfer, payment_failed,
         duplicate_payment                  → base is HIGH
      5. agent_cash_in_issue + CONSISTENT   → base is HIGH
      6. OTHER + INSUFFICIENT_DATA          → LOW
      7. Default                            → MEDIUM
    """
    if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
        return Severity.CRITICAL

    if case_type == CaseType.WRONG_TRANSFER and evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        return Severity.MEDIUM

    base = Severity.MEDIUM

    if case_type in (
        CaseType.WRONG_TRANSFER,
        CaseType.PAYMENT_FAILED,
        CaseType.DUPLICATE_PAYMENT,
    ):
        base = Severity.HIGH

    if case_type == CaseType.AGENT_CASH_IN_ISSUE and evidence.evidence_verdict == EvidenceVerdict.CONSISTENT:
        base = Severity.HIGH

    if evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        return _escalate(base)

    if (
        case_type == CaseType.OTHER
        and evidence.evidence_verdict == EvidenceVerdict.INSUFFICIENT_DATA
    ):
        return Severity.LOW

    return base


# ===========================================================================
# Private helpers
# ===========================================================================


def _escalate(level: Severity) -> Severity:
    """Bump severity up by one tier, capped at CRITICAL."""
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    try:
        idx = order.index(level)
    except ValueError:
        return level
    return order[min(idx + 1, len(order) - 1)]


# ===========================================================================
# Human-review flag
# ===========================================================================

# Case types that always require a human to review regardless of verdict.
_ALWAYS_HUMAN_REVIEW: frozenset = frozenset({
    CaseType.PHISHING_OR_SOCIAL_ENGINEERING,
    CaseType.WRONG_TRANSFER,
    CaseType.DUPLICATE_PAYMENT,
})


def compute_human_review_required(
    case_type: CaseType,
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> bool:
    """Return True when the case should be routed to a human agent.

    Rules (any one is sufficient):
      - Case type is always-review (phishing, wrong_transfer, duplicate_payment)
      - Evidence is INCONSISTENT (potential fraud, needs investigation)
      - Severity is HIGH or CRITICAL (except for payment_failed which is automated)
    """
    if case_type in _ALWAYS_HUMAN_REVIEW:
        return True
    if evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        return True
    if severity in (Severity.HIGH, Severity.CRITICAL) and case_type != CaseType.PAYMENT_FAILED:
        return True
    return False


# ===========================================================================
# Reason codes
# ===========================================================================


def compute_reason_codes(
    case_type: CaseType,
    evidence: EvidenceVerdictResult,
) -> List[str]:
    """Build a human-readable list of codes explaining the pipeline decision.

    The first code is always the case_type value. Additional codes reflect
    the evidence verdict and confidence tier.
    """
    codes: List[str] = [case_type.value]

    if evidence.evidence_verdict == EvidenceVerdict.CONSISTENT:
        codes.append("transaction_match")
    elif evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        codes.append("inconsistent_history")
    else:
        codes.append("insufficient_data")

    if evidence.confidence >= 0.9:
        codes.append("high_confidence")
    elif evidence.confidence >= 0.7:
        codes.append("medium_confidence")

    if evidence.relevant_transaction_id:
        codes.append("transaction_identified")

    return codes
