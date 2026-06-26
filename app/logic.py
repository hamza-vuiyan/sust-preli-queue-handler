"""
logic.py
========
The hybrid investigation pipeline for QueueStorm Investigator.

This module is the *brain* of the API. `app.main` calls the public functions
in this order:

    extracted = call_llm_extractor(complaint_text)            # Step A
    evidence  = evaluate_evidence(extracted, history)         # Step B
    case_type = classify_case(extracted)                      # Step C (classify)
    dept      = route_department(case_type)                   # Step C (route)
    severity  = compute_severity(case_type, evidence)         # Step C (severity)
    reply     = draft_response(extracted, evidence, ...)      # Step C (draft)

Every public function is fully type-hinted, never raises on bad input
(returns safe defaults instead), and is small enough to read in one pass
during the hackathon demo.

The "LLM" in Step A is intentionally a deterministic stub today — the
real swap-in point is `call_llm_extractor`. Everything after it is pure
Python so judges can audit the rules.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.schemas import (
    CaseType,
    Department,
    EvidenceVerdict,
    EvidenceVerdictResult,
    Severity,
    Transaction,
    TransactionStatus,
)

logger = logging.getLogger("queuestorm.logic")

# ---------------------------------------------------------------------------
# Optional [redacted] plug-point.
# ---------------------------------------------------------------------------
#
# The default mode is `stub` — a deterministic regex/keyword extractor that
# needs no API key and works offline. Setting LLM_PROVIDER=gemini in `.env`
# (and providing GEMINI_API_KEY) switches Step A to a real Gemini call with
# a strict JSON schema. If the key is missing or any error happens at
# runtime, we silently fall back to the stub so /analyze-ticket never
# returns a 500 because of an LLM misconfiguration.
#
# `init_llm_client()` is called once from `app.main`'s lifespan hook. After
# that, `call_llm_extractor` simply reads the module-level `_llm_client`.
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=False)
except Exception:  # pragma: no cover — dotenv is optional, env vars also work
    pass

_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "stub").strip().lower() or "stub"
_GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"

_llm_client: Any = None  # set by `init_llm_client()`


def init_llm_client() -> Dict[str, Any]:
    """Initialize the optional [redacted] client.

    Safe to call multiple times. Returns a small status dict that the
    `/health`-adjacent admin surface (or test scripts) can read.

    Returns:
        dict with keys:
          - provider: "stub" | "gemini"
          - model:   the model name in use (or "n/a" for stub)
          - ready:   True if a real call would be attempted.
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
        # Lazy import so the stub path doesn't require google-genai at all.
        from google import genai  # type: ignore

        _llm_client = genai.Client(api_key=_GEMINI_API_KEY)
        logger.info("[redacted] client initialised with model=%s", _GEMINI_MODEL)
        return {"provider": "gemini", "model": _GEMINI_MODEL, "ready": True}
    except Exception as exc:  # pragma: no cover — defensive
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

# ---------------------------------------------------------------------------
# Constants used across the pipeline.
# ---------------------------------------------------------------------------

# How much an amount can drift from the claimed figure and still count as a
# match (covers FX rounding and merchant fees).
AMOUNT_TOLERANCE: float = 0.01

# Window (in seconds) within which a transaction timestamp must fall to be
# considered "the one the user is talking about" when no other signal exists.
DEFAULT_TIME_WINDOW_SECONDS: int = 24 * 60 * 60  # 24h

# Minimum prior successful transfers to the same counterparty needed to
# upgrade a "wrong_transfer" claim to "inconsistent" (contradiction rule).
WRONG_TRANSFER_CONTRADICTION_THRESHOLD: int = 2

# Confidence scores returned to the API. These are deliberately simple so
# judges can see the reasoning by inspecting the function below.
CONFIDENCE_PERFECT_MATCH: float = 0.95
CONFIDENCE_PARTIAL_MATCH: float = 0.6
CONFIDENCE_AMBIGUOUS: float = 0.4
CONFIDENCE_INSUFFICIENT: float = 0.2
CONFIDENCE_CONTRADICTION: float = 0.85


# ===========================================================================
# Step A — "LLM" fact extraction (deterministic stub).
# ===========================================================================
#
# In production this would call an LLM with a strict JSON schema and return
# the parsed object. For the hackathon we ship a regex/keyword extractor
# that is good enough to demo end-to-end and trivial to swap out.
#
# To replace with a real LLM later:
#   1. Call the model with a JSON-only prompt and `response_format={"type":"json_object"}`.
#   2. Parse the response into the same dict shape this function returns.
#   3. Keep the keys (`claimed_amount`, `claimed_counterparty`, `core_issue`)
#      identical — Step B depends on them.


_AMOUNT_PATTERN = re.compile(
    r"(?:bdt|tk|taka|inr|rs|usd|\$|€|£)?\s*([0-9]+(?:[,][0-9]{3})*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

# Common counterparty hints — words that often precede a phone/account id.
_COUNTERPARTY_HINTS = (
    "to ",
    "sent to ",
    "transferred to ",
    "paid to ",
    "received from ",
    "from ",
    "number ",
    "account ",
)

_PHISHING_KEYWORDS = (
    "otp",
    "pin",
    "password",
    "phish",
    "social engineering",
    "fake call",
    "pretended to be",
    "asked for my",
    "verify your",
)
_WRONG_TRANSFER_KEYWORDS = ("wrong number", "wrong transfer", "wrong person", "mistakenly", "by mistake")
_DUPLICATE_KEYWORDS = ("twice", "two times", "duplicate", "charged twice", "double charged")
_PAYMENT_FAILED_KEYWORDS = ("payment failed", "didn't go through", "did not go through", "not received", "money debited")
_REFUND_KEYWORDS = ("refund", "return my money", "reimburse", "money back")
_MERCHANT_DELAY_KEYWORDS = ("merchant", "shop", "store", "seller hasn't", "settlement", "delivery")
_AGENT_CASHIN_KEYWORDS = ("agent", "cash in", "cash-in", "deposit", "bkash agent", "nagad agent")


def call_llm_extractor(complaint_text: str) -> Dict[str, Any]:
    """Extract the three structured facts we need from free-text complaints.

    When a [redacted] client is configured (`LLM_PROVIDER=gemini` and
    `GEMINI_API_KEY` is set), this function calls Gemini with a strict JSON
    prompt and parses the response. If Gemini is not configured, or any
    network/parse error happens, it falls back to the deterministic
    `_extract_*` helpers below — the API never returns a 500 because of an
    LLM misconfiguration.

    Returns:
        dict with keys:
          - claimed_amount (Optional[float])
          - claimed_counterparty (Optional[str])
          - core_issue (str) — one of the canonical issue labels below.
          - raw_excerpt (str) — first ~120 chars, useful for the agent_summary.
    """
    if not complaint_text:
        return {
            "claimed_amount": None,
            "claimed_counterparty": None,
            "core_issue": "other",
            "raw_excerpt": "",
        }

    # --- [redacted] path (opt-in) ------------------------------------------
    if _llm_client is not None:
        try:
            return _call_gemini_extractor(complaint_text)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "[redacted] extraction failed (%s) — falling back to stub extractor.", exc
            )
            # Fall through to the stub.

    # --- Stub path (default, zero-config) ----------------------------------
    text = complaint_text.strip()
    lower = text.lower()

    claimed_amount = _extract_amount(lower)
    claimed_counterparty = _extract_counterparty(text)
    core_issue = _classify_core_issue(lower)

    return {
        "claimed_amount": claimed_amount,
        "claimed_counterparty": claimed_counterparty,
        "core_issue": core_issue,
        "raw_excerpt": text[:120],
    }


# Canonical issue labels — passed to Gemini in the JSON-schema prompt so it
# cannot invent a new category the rule engine doesn't know how to route.
_CANONICAL_CORE_ISSUES = (
    "phishing_or_social_engineering",
    "wrong_transfer",
    "duplicate_payment",
    "merchant_settlement_delay",
    "agent_cash_in_issue",
    "payment_failed",
    "refund_request",
    "other",
)

_GEMINI_SYSTEM_INSTRUCTION = (
    "You are a complaint-analysis assistant for a fintech support pipeline. "
    "Read the customer's complaint and extract structured facts. "
    "Reply with STRICT JSON ONLY (no markdown, no commentary) matching this schema: "
    '{"claimed_amount": number | null, '
    '"claimed_counterparty": string | null, '
    '"core_issue": string}. '
    f"`core_issue` MUST be one of: {list(_CANONICAL_CORE_ISSUES)}. "
    "If the complaint mentions a phishing/social-engineering attempt "
    "(asked for OTP/PIN/password, fake support call, impersonation), set "
    "`core_issue` to `phishing_or_social_engineering` and leave amount/counterparty null."
)


def _call_gemini_extractor(complaint_text: str) -> Dict[str, Any]:
    """Call [redacted] and parse the structured JSON response.

    Raises on any failure — the caller is responsible for falling back to
    the stub extractor.
    """
    # The `google-genai` SDK exposes both sync and async clients. We use
    # the sync `.models.generate_content` here because FastAPI runs this
    # function inside an `async def` route; the call is small and Gemini
    # responds in <2s for flash models, well inside the 5s SLA.
    from google.genai import types  # type: ignore

    response = _llm_client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=complaint_text,
        config=types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=512,
        ),
    )

    # The SDK returns the JSON text in `response.text` when JSON mode is on.
    raw_text = (getattr(response, "text", "") or "").strip()
    if not raw_text:
        raise RuntimeError("empty response from [redacted]")

    parsed = json.loads(raw_text)

    # Defensive normalization — Gemini might return slightly different
    # casing or extra keys. We coerce into the shape the rule engine needs.
    core_issue_raw = str(parsed.get("core_issue", "other")).strip().lower()
    core_issue = core_issue_raw if core_issue_raw in _CANONICAL_CORE_ISSUES else "other"

    amount_raw = parsed.get("claimed_amount")
    claimed_amount: Optional[float]
    if amount_raw is None or amount_raw == "":
        claimed_amount = None
    else:
        try:
            claimed_amount = float(amount_raw)
        except (TypeError, ValueError):
            claimed_amount = None

    counterparty_raw = parsed.get("claimed_counterparty")
    claimed_counterparty: Optional[str]
    if isinstance(counterparty_raw, str) and counterparty_raw.strip():
        claimed_counterparty = counterparty_raw.strip()
    else:
        claimed_counterparty = None

    return {
        "claimed_amount": claimed_amount,
        "claimed_counterparty": claimed_counterparty,
        "core_issue": core_issue,
        "raw_excerpt": complaint_text.strip()[:120],
    }


def _extract_amount(lower_text: str) -> Optional[float]:
    """Return the first plausible numeric amount mentioned in the text."""
    for match in _AMOUNT_PATTERN.finditer(lower_text):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        # Reject year-like 4-digit numbers that aren't preceded by currency.
        if value.is_integer() and 1900 <= value <= 2100 and len(raw) == 4:
            continue
        if value > 0:
            return value
    return None


def _extract_counterparty(text: str) -> Optional[str]:
    """Heuristic counterparty extraction.

    Strategy (in priority order):
      1. Any phone-number-shaped token (digits with optional dashes/spaces)
         wins outright — that's almost always the counterparty.
      2. Any token that looks like a merchant id (letters + digits + `_`).
      3. Fall back to the first non-stop-word after a hint verb.
    """
    # Priority 1: phone-number-shaped token anywhere in the text.
    phone_match = re.search(r"\b\d[\d\-\s]{4,}\d\b", text)
    if phone_match:
        token = phone_match.group(0).strip()
        # Clean up multiple spaces but keep dashes — phone numbers use them.
        token = re.sub(r"\s+", " ", token)
        return token

    # Priority 2: merchant / account id style token (e.g. merchant_supermart_22).
    # Scan ALL candidates (not just the first regex match) and return the
    # first one that looks like an id — has an underscore or a digit. This
    # is more robust than picking whichever token comes first in the text.
    for token_match in re.finditer(r"\b[a-zA-Z][\w-]{3,}\b", text):
        token = token_match.group(0)
        if ("_" in token or any(ch.isdigit() for ch in token)) and token.lower() not in _STOP_WORDS:
            return token

    # Priority 3: walk the hint verbs and pick the first useful token.
    lowered = text.lower()
    for hint in _COUNTERPARTY_HINTS:
        idx = lowered.find(hint)
        if idx == -1:
            continue
        after = text[idx + len(hint):].strip()
        for stop in (".", ",", ";", "?", "!", "\n"):
            cut = after.find(stop)
            if cut != -1:
                after = after[:cut]
        for token in after.split():
            token = token.strip("\"'()[]{}")
            if token and token.lower() not in _STOP_WORDS:
                return token
    return None


# Stop-words that the hint-follower must skip so it doesn't pick
# "the"/"wrong" after "to the wrong number".
_STOP_WORDS = frozenset({
    "the", "a", "an", "my", "your", "our", "this", "that",
    "wrong", "right", "correct",
})


def _classify_core_issue(lower_text: str) -> str:
    """Map free text to one of the canonical `core_issue` labels."""
    # Order matters — phishing is checked first because it can co-occur
    # with a wrong_transfer claim and should take priority for routing.
    if any(k in lower_text for k in _PHISHING_KEYWORDS):
        return "phishing_or_social_engineering"
    if any(k in lower_text for k in _WRONG_TRANSFER_KEYWORDS):
        return "wrong_transfer"
    if any(k in lower_text for k in _DUPLICATE_KEYWORDS):
        return "duplicate_payment"
    if any(k in lower_text for k in _MERCHANT_DELAY_KEYWORDS):
        return "merchant_settlement_delay"
    if any(k in lower_text for k in _AGENT_CASHIN_KEYWORDS):
        return "agent_cash_in_issue"
    if any(k in lower_text for k in _PAYMENT_FAILED_KEYWORDS):
        return "payment_failed"
    if any(k in lower_text for k in _REFUND_KEYWORDS):
        return "refund_request"
    return "other"


# ===========================================================================
# Step B — Deterministic rule engine.
# ===========================================================================


def evaluate_evidence(
    extracted: Dict[str, Any],
    transaction_history: List[Transaction],
) -> EvidenceVerdictResult:
    """Score every transaction against the extracted facts.

    Rules (in evaluation order):
      0. No history at all -> INSUFFICIENT_DATA.
      1. Phishing case -> INSUFFICIENT_DATA (no money to verify).
      2. Zero numeric matches on amount -> INSUFFICIENT_DATA.
      3. Exactly one strong match -> CONSISTENT (with contradiction override
         for wrong_transfer claims — see Rule 5).
      4. Multiple strong matches, all same counterparty ->
         INSUFFICIENT_DATA (ambiguous), UNLESS this is a wrong_transfer
         claim and there are >=2 PRIOR successful transfers to the same
         counterparty — in which case INCONSISTENT (contradiction).
      5. Wrong-transfer claim AND prior success(es) to same counterparty ->
         INCONSISTENT (contradiction).
      6. Otherwise -> INSUFFICIENT_DATA.
    """
    # Rule 0: empty history.
    if not transaction_history:
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=None,
            confidence=CONFIDENCE_INSUFFICIENT,
        )

    # Rule 1: phishing — no money has moved; we cannot corroborate.
    if extracted.get("core_issue") == "phishing_or_social_engineering":
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=None,
            confidence=CONFIDENCE_INSUFFICIENT,
        )

    claimed_amount: Optional[float] = extracted.get("claimed_amount")
    claimed_counterparty: Optional[str] = (extracted.get("claimed_counterparty") or "").lower() or None

    # Rule 2: no amount in the complaint — we cannot anchor a match.
    if claimed_amount is None:
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=None,
            confidence=CONFIDENCE_INSUFFICIENT,
        )

    # Score every transaction. A "strong" match requires BOTH amount AND
    # counterparty to agree (with tolerance). A "weak" match agrees only on
    # amount — useful as a fallback when the counterparty string didn't
    # extract cleanly.
    strong_matches: List[Transaction] = []
    weak_matches: List[Transaction] = []

    for tx in transaction_history:
        amount_ok = abs(tx.amount - claimed_amount) <= AMOUNT_TOLERANCE
        party_ok = (
            claimed_counterparty is not None
            and claimed_counterparty in tx.counterparty.lower()
        ) or (
            claimed_counterparty is not None
            and tx.counterparty.lower() in claimed_counterparty
        )

        if amount_ok and party_ok:
            strong_matches.append(tx)
        elif amount_ok:
            weak_matches.append(tx)

    # Rule 4 + 5: multiple strong matches — contradiction beats ambiguity.
    if len(strong_matches) >= 2:
        counterparties = {tx.counterparty.lower() for tx in strong_matches}
        # All matches share the same counterparty.
        if len(counterparties) == 1:
            # Wrong-transfer + repeated prior successes to the same person
            # is a contradiction, not just "ambiguous". We treat the most
            # recent match as the alleged one and count how many PRIOR
            # successful transfers went to the same counterparty.
            if extracted.get("core_issue") == "wrong_transfer" and claimed_counterparty:
                most_recent = _pick_most_recent(strong_matches)
                prior_successes = _count_prior_successes(
                    most_recent, claimed_counterparty, transaction_history
                )
                if prior_successes >= WRONG_TRANSFER_CONTRADICTION_THRESHOLD:
                    return EvidenceVerdictResult(
                        evidence_verdict=EvidenceVerdict.INCONSISTENT,
                        relevant_transaction_id=most_recent.transaction_id,
                        confidence=CONFIDENCE_CONTRADICTION,
                    )
            return EvidenceVerdictResult(
                evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
                relevant_transaction_id=None,
                confidence=CONFIDENCE_AMBIGUOUS,
            )
        # Same amount, different merchants — pick the most recent and flag
        # the rest for the human reviewer.
        strongest = _pick_most_recent(strong_matches)
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=strongest.transaction_id,
            confidence=CONFIDENCE_AMBIGUOUS,
        )

    # Rule 3: exactly one strong match -> CONSISTENT.
    if len(strong_matches) == 1:
        tx = strong_matches[0]
        # Rule 5: contradiction check for wrong_transfer.
        if extracted.get("core_issue") == "wrong_transfer" and claimed_counterparty:
            prior_successes = _count_prior_successes(
                tx, claimed_counterparty, transaction_history
            )
            if prior_successes >= WRONG_TRANSFER_CONTRADICTION_THRESHOLD:
                return EvidenceVerdictResult(
                    evidence_verdict=EvidenceVerdict.INCONSISTENT,
                    relevant_transaction_id=tx.transaction_id,
                    confidence=CONFIDENCE_CONTRADICTION,
                )
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.CONSISTENT,
            relevant_transaction_id=tx.transaction_id,
            confidence=CONFIDENCE_PERFECT_MATCH,
        )

    # Fallback: weak matches only.
    if weak_matches:
        tx = _pick_most_recent(weak_matches)
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=tx.transaction_id,
            confidence=CONFIDENCE_PARTIAL_MATCH,
        )

    # Rule 6: nothing matched at all.
    return EvidenceVerdictResult(
        evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
        relevant_transaction_id=None,
        confidence=CONFIDENCE_INSUFFICIENT,
    )


def _pick_most_recent(transactions: List[Transaction]) -> Transaction:
    """Return the transaction with the latest `timestamp`."""
    return max(transactions, key=lambda t: t.timestamp)


def _count_prior_successes(
    match: Transaction,
    claimed_counterparty_lower: Optional[str],
    history: List[Transaction],
) -> int:
    """Count earlier successful transactions to the same counterparty."""
    if not claimed_counterparty_lower:
        return 0
    count = 0
    for tx in history:
        if tx.transaction_id == match.transaction_id:
            continue
        if tx.timestamp >= match.timestamp:
            continue
        if tx.status != TransactionStatus.SUCCESS:
            continue
        if claimed_counterparty_lower in tx.counterparty.lower():
            count += 1
    return count


# ===========================================================================
# Step C — Classification, routing, severity.
# ===========================================================================


_CORE_ISSUE_TO_CASE: Dict[str, CaseType] = {
    "wrong_transfer": CaseType.WRONG_TRANSFER,
    "payment_failed": CaseType.PAYMENT_FAILED,
    "refund_request": CaseType.REFUND_REQUEST,
    "duplicate_payment": CaseType.DUPLICATE_PAYMENT,
    "merchant_settlement_delay": CaseType.MERCHANT_SETTLEMENT_DELAY,
    "agent_cash_in_issue": CaseType.AGENT_CASH_IN_ISSUE,
    "phishing_or_social_engineering": CaseType.PHISHING_OR_SOCIAL_ENGINEERING,
    "other": CaseType.OTHER,
}


def classify_case(extracted: Dict[str, Any]) -> CaseType:
    """Map the extracted `core_issue` to the public `CaseType` enum."""
    return _CORE_ISSUE_TO_CASE.get(
        extracted.get("core_issue", "other"),
        CaseType.OTHER,
    )


# Static routing table — per the spec.
_CASE_TO_DEPARTMENT: Dict[CaseType, Department] = {
    CaseType.PHISHING_OR_SOCIAL_ENGINEERING: Department.FRAUD_RISK,
    CaseType.WRONG_TRANSFER: Department.DISPUTE_RESOLUTION,
    CaseType.DUPLICATE_PAYMENT: Department.DISPUTE_RESOLUTION,
    CaseType.PAYMENT_FAILED: Department.PAYMENTS_OPS,
    CaseType.MERCHANT_SETTLEMENT_DELAY: Department.MERCHANT_OPERATIONS,
    CaseType.AGENT_CASH_IN_ISSUE: Department.AGENT_OPERATIONS,
    CaseType.REFUND_REQUEST: Department.CUSTOMER_SUPPORT,
    CaseType.OTHER: Department.CUSTOMER_SUPPORT,
}


def route_department(case_type: CaseType) -> Department:
    """Return the department that should own this case."""
    return _CASE_TO_DEPARTMENT.get(case_type, Department.CUSTOMER_SUPPORT)


def compute_severity(case_type: CaseType, evidence: EvidenceVerdictResult) -> Severity:
    """Derive a severity score from the case type and rule-engine verdict.

    Rules:
      - Phishing is always CRITICAL (account-takeover risk).
      - INCONSISTENT evidence escalates one tier (capped at CRITICAL).
      - CONSISTENT with payment_failed is HIGH (money in limbo).
      - Default MEDIUM; demote to LOW only for OTHER + INSUFFICIENT_DATA.
    """
    if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
        return Severity.CRITICAL

    base = Severity.MEDIUM

    if case_type == CaseType.WRONG_TRANSFER:
        base = Severity.HIGH

    if case_type == CaseType.PAYMENT_FAILED:
        base = Severity.HIGH

    if evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        return _escalate(base)

    if (
        case_type == CaseType.PAYMENT_FAILED
        and evidence.evidence_verdict == EvidenceVerdict.CONSISTENT
    ):
        return Severity.HIGH

    if (
        case_type == CaseType.OTHER
        and evidence.evidence_verdict == EvidenceVerdict.INSUFFICIENT_DATA
    ):
        return Severity.LOW

    return base


def _escalate(level: Severity) -> Severity:
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    try:
        idx = order.index(level)
    except ValueError:
        return level
    return order[min(idx + 1, len(order) - 1)]


# ===========================================================================
# Step C — Drafting the outbound reply.
# ===========================================================================
#
# Templates are intentionally simple. They are *un-guarded* on purpose —
# `app.safety` scrubs anything sensitive before the reply hits the wire.
# If you add a new branch here, add a corresponding unit case in
# `app.safety` self-test so the guardrails stay in sync.


def draft_response(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    case_type: CaseType,
    severity: Severity,
) -> Dict[str, str]:
    """Generate the agent-facing summary and the outbound customer reply.

    Returns:
        dict with keys:
          - agent_summary (str): One paragraph for the human agent.
          - recommended_next_action (str): What the agent should do next.
          - customer_reply (str): The message the system *wants* to send.
                                    Still un-guarded here — safety.py
                                    is the LAST thing to touch it.
    """
    handler = _DRAFTERS.get(case_type, _draft_other)
    return handler(extracted, evidence, severity)


def _draft_wrong_transfer(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    amount = extracted.get("claimed_amount")
    counterparty = extracted.get("claimed_counterparty") or "the recipient"
    tx_ref = evidence.relevant_transaction_id or "N/A"

    if evidence.evidence_verdict == EvidenceVerdict.CONSISTENT:
        agent_summary = (
            f"Customer reports a wrong transfer of {amount} to {counterparty}. "
            f"Transaction {tx_ref} matches the claim on both amount and "
            f"counterparty. Recommend opening a dispute for review."
        )
        next_action = (
            f"Open dispute case for {tx_ref}; freeze any pending settlement "
            f"to the counterparty and request callback within 2 hours."
        )
        customer_reply = (
            f"Thanks for reaching out — we found a transfer matching your "
            f"description (reference {tx_ref}). We will refund you the eligible "
            f"amount after our dispute team reviews the case. Please do not "
            f"share your PIN, OTP, or password with anyone while we work on this."
        )
    elif evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        agent_summary = (
            f"Customer claims a wrong transfer to {counterparty}, but history "
            f"shows multiple prior successful transfers to the same "
            f"counterparty. Treat the claim with caution."
        )
        next_action = (
            "Verify the customer's identity with two security questions "
            "before processing any reversal."
        )
        customer_reply = (
            "Thanks for getting in touch. We've flagged your case for manual "
            "review and will get back to you within one business day."
        )
    else:
        agent_summary = (
            f"Customer claims a wrong transfer of {amount} to {counterparty}, "
            f"but we could not confirm a unique matching transaction."
        )
        next_action = "Ask the customer for the exact transaction id or timestamp."
        customer_reply = (
            "Thanks for reaching out. To help us locate the transfer quickly, "
            "please reply with the exact transaction reference from your app "
            "or the time and date of the transfer."
        )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_phishing(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    agent_summary = (
        "Customer reports a possible phishing / social-engineering attempt. "
        "Treat as CRITICAL — escalate to fraud_risk immediately."
    )
    next_action = (
        "Lock outgoing transfers on the account for 24h, raise a fraud-risk "
        "ticket, and call the customer on their registered number to confirm."
    )
    customer_reply = (
        "Thank you for flagging this. We will never ask for your PIN, OTP, "
        "or password over the phone or by message. Our team will contact you "
        "through our official channels to help secure your account."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_payment_failed(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    tx_ref = evidence.relevant_transaction_id or "N/A"
    if evidence.evidence_verdict == EvidenceVerdict.CONSISTENT:
        agent_summary = (
            f"Customer reports a payment that they believe failed. "
            f"Transaction {tx_ref} matches the claim. Recommend reconciliation."
        )
        next_action = f"Trigger a reconciliation sweep for {tx_ref} and confirm receipt."
    else:
        agent_summary = (
            "Customer reports a payment failure. No matching successful "
            "transaction found in the supplied history."
        )
        next_action = "Pull the gateway-level status for the most recent transaction."
    customer_reply = (
        "Thanks for letting us know. We're checking the transaction now and "
        "will update you as soon as we have confirmation."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_refund_request(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    agent_summary = (
        "Customer is requesting a refund. Forward to customer_support for "
        "manual review."
    )
    next_action = "Verify the original transaction and the merchant's refund policy."
    customer_reply = (
        "Thanks for reaching out. We've recorded your refund request and our "
        "team will review the case and follow up with the outcome."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_duplicate_payment(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    agent_summary = (
        "Customer reports being charged twice for the same transaction. "
        "Verify duplicate id before processing a reversal."
    )
    next_action = "Compare gateway and ledger records for the disputed window."
    customer_reply = (
        "Thanks for getting in touch. We're checking for any duplicate "
        "charges and will update you shortly."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_merchant_delay(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    agent_summary = (
        "Customer is reporting a merchant settlement delay. Route to "
        "merchant_operations for partner follow-up."
    )
    next_action = "Contact the merchant partner and request settlement status."
    customer_reply = (
        "Thanks for your patience. We're following up with the merchant and "
        "will share an update as soon as we hear back."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_agent_cashin(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    agent_summary = (
        "Customer reports an issue with an agent cash-in. Route to "
        "agent_operations for on-the-ground follow-up."
    )
    next_action = (
        "Verify the agent's cash-in ledger and contact the agent within 1h."
    )
    customer_reply = (
        "Thanks for letting us know. We've passed this to our agent "
        "operations team and they will reach out shortly."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


def _draft_other(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    agent_summary = (
        "Customer complaint did not match a known case type. Manual review "
        "required."
    )
    next_action = "Assign to customer_support tier-2 for triage."
    customer_reply = (
        "Thanks for reaching out. We've received your message and a "
        "specialist will follow up shortly."
    )
    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": customer_reply,
    }


_DRAFTERS = {
    CaseType.WRONG_TRANSFER: _draft_wrong_transfer,
    CaseType.PAYMENT_FAILED: _draft_payment_failed,
    CaseType.REFUND_REQUEST: _draft_refund_request,
    CaseType.DUPLICATE_PAYMENT: _draft_duplicate_payment,
    CaseType.MERCHANT_SETTLEMENT_DELAY: _draft_merchant_delay,
    CaseType.AGENT_CASH_IN_ISSUE: _draft_agent_cashin,
    CaseType.PHISHING_OR_SOCIAL_ENGINEERING: _draft_phishing,
}


# ---------------------------------------------------------------------------
# Convenience: one-shot pipeline used by `app.main`.
# ---------------------------------------------------------------------------


def run_pipeline(
    complaint_text: str,
    transaction_history: List[Transaction],
) -> Tuple[
    Dict[str, Any],
    EvidenceVerdictResult,
    CaseType,
    Department,
    Severity,
    Dict[str, str],
]:
    """Execute the full Step A -> Step C pipeline and return everything."""
    extracted = call_llm_extractor(complaint_text)
    evidence = evaluate_evidence(extracted, transaction_history)
    case_type = classify_case(extracted)
    department = route_department(case_type)
    severity = compute_severity(case_type, evidence)
    drafted = draft_response(extracted, evidence, case_type, severity)
    return extracted, evidence, case_type, department, severity, drafted