"""
schemas.py
==========
Strict Pydantic v2 schemas for the QueueStorm Investigator API.

Why this lives in its own module:
- Centralizes the *exact* contract the API promises to clients.
- Keeps enum strings identical to the spec so the validator never has to
  guess whether "WrongTransfer" should map to "wrong_transfer".
- Used by both `app.main` (validation) and `app.pipeline` (typing).

NOTE: enum string values are part of the public API. Do NOT rename them.

Field-name flexibility
----------------------
The request model accepts both the canonical names AND common real-world
aliases (e.g. `complaint` == `complaint_text`). Extra / unknown fields
from the caller are silently ignored so clients can send richer payloads
(ticket_id, language, channel, campaign_context, …) without breaking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    REVERSED = "reversed"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

# Aliases that real-world clients send for TransactionStatus values.
_STATUS_ALIASES: Dict[str, str] = {
    "success":    "completed",
    "succeed":    "completed",
    "successful": "completed",
    "done":       "completed",
    "ok":         "completed",
    "error":      "failed",
    "fail":       "failed",
    "failure":    "failed",
    "declined":   "failed",
    "cancelled":  "reversed",
    "canceled":   "reversed",
    "refunded":   "reversed",
    "in_progress": "pending",
    "processing": "pending",
}


class Transaction(BaseModel):
    """One element of the customer-supplied transaction_history array."""

    # extra="ignore" — callers may include extra metadata fields; we just
    # don't need them in the rule engine.
    model_config = ConfigDict(extra="ignore")

    transaction_id: str = Field(..., min_length=1, description="Unique transaction id.")
    amount: float = Field(..., ge=0, description="Transaction amount in `currency`.")
    currency: str = Field(
        default="BDT",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code. Defaults to 'BDT' if omitted.",
    )
    counterparty: str = Field(..., min_length=1, description="Receiver identifier.")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of the transaction.")
    status: TransactionStatus = Field(..., description="Reported status of the transaction.")
    channel: Optional[str] = Field(default=None, description="Optional channel, e.g. 'mobile_app'.")
    type: Optional[str] = Field(default=None, description="Optional type, e.g. 'transfer'.")

    @field_validator("status", mode="before")
    @classmethod
    def normalise_status(cls, v: Any) -> Any:
        """Map common real-world status strings to canonical enum values.

        e.g. "completed" → "success", "cancelled" → "reversed".
        Unknown values are passed through and will fail normal enum validation
        with a clear error message.
        """
        if isinstance(v, str):
            return _STATUS_ALIASES.get(v.lower().strip(), v)
        return v


class AnalyzeTicketRequest(BaseModel):
    """Request payload for POST /analyze-ticket.

    Accepts both canonical field names AND common aliases:
      - `complaint`       OR  `complaint_text`
      - `customer_id`     is optional (defaults to "unknown")
      - `ticket_id`       is optional, echoed back in the response
      - `language`        is optional

    Unknown extra fields (channel, campaign_context, …)
    are silently ignored so richer upstream payloads work without changes.
    """

    # extra="ignore" — accept richer payloads from real-world clients.
    model_config = ConfigDict(extra="ignore")

    ticket_id: Optional[str] = Field(
        default=None,
        description="Optional ticket identifier, echoed back in the response.",
    )
    complaint: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Free-text customer complaint.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language of the complaint, e.g. en, bn, mixed.",
    )
    customer_id: str = Field(
        default="unknown",
        description="Internal customer identifier. Optional — defaults to 'unknown'.",
    )
    transaction_history: List[Transaction] = Field(
        ...,
        max_length=500,
        description="Recent transactions to cross-reference against the complaint.",
    )

    @model_validator(mode="before")
    @classmethod
    def remap_aliases(cls, data: Any) -> Any:
        """Handle field-name aliases before Pydantic validates individual fields.

        Mappings applied (only when the canonical name is absent):
          complaint_text   → complaint
          complaint        (canonical, used as-is)
        """
        if not isinstance(data, dict):
            return data

        # Map `complaint_text` → `complaint` when only the alias is present.
        if "complaint" not in data and "complaint_text" in data:
            data = dict(data)          # don't mutate the original
            data["complaint"] = data.pop("complaint_text")

        return data


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
    """Response payload for POST /analyze-ticket.

    Fields are flat (no nested evidence object) to match the organizer spec.
    Evidence verdict, relevant transaction, and confidence are top-level.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: Optional[str] = Field(
        default=None,
        description="Echoed from the request ticket_id, if provided.",
    )
    relevant_transaction_id: Optional[str] = Field(
        default=None,
        description="Best-matching transaction id, or None if not found.",
    )
    evidence_verdict: EvidenceVerdict
    case_type: CaseType
    severity: Severity
    department: Department
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    human_review_required: bool = Field(
        default=False,
        description="True when the case must be reviewed by a human agent.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rule-engine confidence score in [0, 1].",
    )
    reason_codes: List[str] = Field(
        default_factory=list,
        description="Machine-readable codes explaining the pipeline decision.",
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