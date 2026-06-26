"""
schemas.py
==========
Strict Pydantic v2 schemas for the QueueStorm Investigator API.

Why this lives in its own module:
- Centralizes the *exact* contract the API promises to clients.
- Keeps enum strings identical to the spec so the validator never has to
  guess whether "WrongTransfer" should map to "wrong_transfer".
- Used by both `app.main` (validation) and `app.logic` (typing).

NOTE: enum string values are part of the public API. Do NOT rename them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums — values must match the spec verbatim.
# ---------------------------------------------------------------------------


class EvidenceVerdict(str, Enum):
    """How well the transaction history supports the customer's claim."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    INSUFFICIENT_DATA = "insufficient_data"


class CaseType(str, Enum):
    """Categorization of the support ticket."""

    WRONG_TRANSFER = "wrong_transfer"
    PAYMENT_FAILED = "payment_failed"
    REFUND_REQUEST = "refund_request"
    DUPLICATE_PAYMENT = "duplicate_payment"
    MERCHANT_SETTLEMENT_DELAY = "merchant_settlement_delay"
    AGENT_CASH_IN_ISSUE = "agent_cash_in_issue"
    PHISHING_OR_SOCIAL_ENGINEERING = "phishing_or_social_engineering"
    OTHER = "other"


class Severity(str, Enum):
    """Operational urgency assigned by the pipeline."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Department(str, Enum):
    """Team the ticket should be routed to."""

    CUSTOMER_SUPPORT = "customer_support"
    DISPUTE_RESOLUTION = "dispute_resolution"
    PAYMENTS_OPS = "payments_ops"
    MERCHANT_OPERATIONS = "merchant_operations"
    AGENT_OPERATIONS = "agent_operations"
    FRAUD_RISK = "fraud_risk"


# Status enum is internal-only (transaction statuses are not part of the
# public response), but we expose it as an Enum so the schema stays tight.
class TransactionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    REVERSED = "reversed"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    """One element of the customer-supplied transaction_history array."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., min_length=1, description="Unique transaction id.")
    amount: float = Field(..., ge=0, description="Transaction amount in `currency`.")
    currency: str = Field(..., min_length=3, max_length=3, description="ISO 4217 currency code.")
    counterparty: str = Field(..., min_length=1, description="Receiver identifier.")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of the transaction.")
    status: TransactionStatus = Field(..., description="Reported status of the transaction.")
    channel: Optional[str] = Field(default=None, description="Optional channel, e.g. 'mobile_app'.")
    type: Optional[str] = Field(default=None, description="Optional type, e.g. 'transfer'.")


class AnalyzeTicketRequest(BaseModel):
    """Request payload for POST /analyze-ticket."""

    model_config = ConfigDict(extra="forbid")

    complaint_text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Free-text customer complaint.",
    )
    customer_id: str = Field(..., min_length=1, description="Internal customer identifier.")
    transaction_history: List[Transaction] = Field(
        ...,
        max_length=500,
        description="Recent transactions to cross-reference against the complaint.",
    )


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class EvidenceVerdictResult(BaseModel):
    """Result of the deterministic rule engine (Step B)."""

    model_config = ConfigDict(extra="forbid")

    evidence_verdict: EvidenceVerdict
    relevant_transaction_id: Optional[str] = Field(
        default=None,
        description="Single best-matching transaction id, or None if ambiguous/missing.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rule-engine confidence score in [0, 1].",
    )


class AnalyzeTicketResponse(BaseModel):
    """Response payload for POST /analyze-ticket."""

    model_config = ConfigDict(extra="forbid")

    case_type: CaseType
    severity: Severity
    department: Department
    evidence: EvidenceVerdictResult
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    guardrails_applied: List[str] = Field(
        default_factory=list,
        description="Audit trail of safety rules the reply was scrubbed through.",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time spent inside the pipeline (excluding network).",
    )


# ---------------------------------------------------------------------------
# Error envelope used by the global exception handlers in `app.main`.
# ---------------------------------------------------------------------------


class SafeErrorResponse(BaseModel):
    """Controlled error envelope — never leaks stack traces or internals."""

    model_config = ConfigDict(extra="forbid")

    error: str
    message: str
    details: Optional[List[str]] = None