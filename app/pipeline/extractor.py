"""
pipeline/extractor.py
=====================
Step A — Fact extraction from free-text complaints.

Supports two modes selected at runtime:
  - Gemini  : calls the configured model with a strict JSON schema prompt.
  - Stub    : deterministic regex/keyword extractor (default, zero-config).

The public entry point is `call_llm_extractor`. Everything below it is
private to this module. To swap in a different LLM, replace
`_call_gemini_extractor` — the rest of the pipeline doesn't care.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from app.pipeline.client import get_gemini_model, get_llm_client

logger = logging.getLogger("queuestorm.pipeline.extractor")

# ---------------------------------------------------------------------------
# Canonical issue labels — must stay in sync with CaseType in schemas.py.
# Passed to Gemini in the prompt so it cannot invent a new category.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Regex patterns and keyword lists for the stub extractor.
# ---------------------------------------------------------------------------
_AMOUNT_PATTERN = re.compile(
    r"(?:bdt|tk|taka|inr|rs|usd|\$|€|£)?\s*([0-9]+(?:[,][0-9]{3})*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

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
    "otp", "pin", "password", "phish", "social engineering",
    "fake call", "pretended to be", "asked for my", "verify your",
)
_WRONG_TRANSFER_KEYWORDS   = ("wrong number", "wrong transfer", "wrong person", "mistakenly", "by mistake")
_DUPLICATE_KEYWORDS        = ("twice", "two times", "duplicate", "charged twice", "double charged")
_PAYMENT_FAILED_KEYWORDS   = ("payment failed", "didn't go through", "did not go through", "not received", "money debited")
_REFUND_KEYWORDS           = ("refund", "return my money", "reimburse", "money back")
_MERCHANT_DELAY_KEYWORDS   = ("merchant", "shop", "store", "seller hasn't", "settlement", "delivery")
_AGENT_CASHIN_KEYWORDS     = ("agent", "cash in", "cash-in", "deposit", "bkash agent", "nagad agent")

# Words the hint-follower must skip to avoid picking "the wrong number" → "wrong".
_STOP_WORDS = frozenset({
    "the", "a", "an", "my", "your", "our", "this", "that",
    "wrong", "right", "correct",
})


# ===========================================================================
# Public API
# ===========================================================================


def call_llm_extractor(complaint_text: str) -> Dict[str, Any]:
    """Extract structured facts from a free-text complaint.

    Tries Gemini first (if the client is ready), then falls back to the
    deterministic stub extractor. Never raises — returns safe defaults on
    empty / None input.

    Returns:
        dict with keys:
          - claimed_amount       (Optional[float])
          - claimed_counterparty (Optional[str])
          - core_issue           (str)  — one of _CANONICAL_CORE_ISSUES
          - raw_excerpt          (str)  — first ~120 chars for agent_summary
    """
    if not complaint_text:
        return {
            "claimed_amount": None,
            "claimed_counterparty": None,
            "core_issue": "other",
            "raw_excerpt": "",
        }

    client = get_llm_client()
    if client is not None:
        try:
            return _call_gemini_extractor(complaint_text, client)
        except Exception as exc:
            logger.warning(
                "[redacted] extraction failed (%s) — falling back to stub extractor.", exc
            )

    return _stub_extractor(complaint_text)


# ===========================================================================
# Gemini path
# ===========================================================================

# Generic phrases Gemini sometimes returns as claimed_counterparty that are
# NOT real identifiers. We null these out so the evidence engine uses
# amount-only matching rather than failing a counterparty comparison.
_GENERIC_COUNTERPARTY_PHRASES = frozenset({
    "wrong number", "wrong person", "unknown", "unknown number",
    "n/a", "na", "none", "null", "not specified", "not mentioned",
    "unspecified", "unclear", "unknown recipient", "wrong recipient",
})


def _normalise_counterparty(raw: object) -> Optional[str]:
    """Return a clean counterparty string, or None if it's generic/empty."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.lower() in _GENERIC_COUNTERPARTY_PHRASES:
        return None
    return cleaned


def _call_gemini_extractor(complaint_text: str, client: Any) -> Dict[str, Any]:
    """Call the configured Gemini model and parse the strict JSON response.

    Raises on any failure — the caller (`call_llm_extractor`) falls back
    to the stub.
    """
    from google.genai import types  # type: ignore

    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=complaint_text,
        config=types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=512,
        ),
    )

    raw_text = (getattr(response, "text", "") or "").strip()
    if not raw_text:
        raise RuntimeError("empty response from [redacted]")

    parsed = json.loads(raw_text)

    # Defensive normalisation — coerce into the shape the rule engine needs.
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

    # Use _normalise_counterparty to reject generic phrases like "wrong number"
    # that Gemini sometimes returns when the complaint doesn't name a real recipient.
    claimed_counterparty = _normalise_counterparty(parsed.get("claimed_counterparty"))

    return {
        "claimed_amount": claimed_amount,
        "claimed_counterparty": claimed_counterparty,
        "core_issue": core_issue,
        "raw_excerpt": complaint_text.strip()[:120],
    }



# ===========================================================================
# Stub path (default, zero-config)
# ===========================================================================


def _stub_extractor(complaint_text: str) -> Dict[str, Any]:
    """Deterministic regex/keyword extractor — no API key required."""
    text = complaint_text.strip()
    lower = text.lower()
    return {
        "claimed_amount": _extract_amount(lower),
        "claimed_counterparty": _extract_counterparty(text),
        "core_issue": _classify_core_issue(lower),
        "raw_excerpt": text[:120],
    }


def _extract_amount(lower_text: str) -> Optional[float]:
    """Return the first plausible numeric amount mentioned in the text."""
    for match in _AMOUNT_PATTERN.finditer(lower_text):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        # Reject year-like 4-digit numbers not preceded by a currency symbol.
        if value.is_integer() and 1900 <= value <= 2100 and len(raw) == 4:
            continue
        if value > 0:
            return value
    return None


def _extract_counterparty(text: str) -> Optional[str]:
    """Heuristic counterparty extraction (phone → merchant id → hint verb).

    Priority order:
      1. Phone-number-shaped token (digits with optional dashes/spaces).
      2. Merchant/account id token (letters + digits or underscore).
      3. First non-stop-word after a hint verb.
    """
    # Priority 1: phone number anywhere in the text.
    phone_match = re.search(r"\b\d[\d\-\s]{4,}\d\b", text)
    if phone_match:
        token = phone_match.group(0).strip()
        return re.sub(r"\s+", " ", token)

    # Priority 2: merchant / account id style token.
    for token_match in re.finditer(r"\b[a-zA-Z][\w-]{3,}\b", text):
        token = token_match.group(0)
        if ("_" in token or any(ch.isdigit() for ch in token)) and token.lower() not in _STOP_WORDS:
            return token

    # Priority 3: hint verb follower.
    lowered = text.lower()
    for hint in _COUNTERPARTY_HINTS:
        idx = lowered.find(hint)
        if idx == -1:
            continue
        after = text[idx + len(hint):].strip()
        for stop_char in (".", ",", ";", "?", "!", "\n"):
            cut = after.find(stop_char)
            if cut != -1:
                after = after[:cut]
        for token in after.split():
            token = token.strip("\"'()[]{}") 
            if token and token.lower() not in _STOP_WORDS:
                return token
    return None


def _classify_core_issue(lower_text: str) -> str:
    """Map free text to one of the canonical core_issue labels.

    Order matters — phishing is checked first because it can co-occur with
    other issue types and should always take routing priority.
    """
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
