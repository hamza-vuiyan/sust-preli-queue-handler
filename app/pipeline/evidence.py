"""
pipeline/evidence.py
====================
Step B — Deterministic evidence rule engine.

Scores the customer-supplied transaction history against the facts
extracted in Step A and returns one of three verdicts:

    CONSISTENT        — the history supports the claim
    INCONSISTENT      — the history contradicts the claim
    INSUFFICIENT_DATA — not enough data to reach a verdict

All logic is pure Python — no LLM or I/O involved.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.schemas import (
    EvidenceVerdict,
    EvidenceVerdictResult,
    Transaction,
    TransactionStatus,
)

logger = logging.getLogger("queuestorm.pipeline.evidence")

# ---------------------------------------------------------------------------
# Tuning constants — change here, nowhere else.
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

# Confidence scores returned to the API.
CONFIDENCE_PERFECT_MATCH: float = 0.95    # amount + counterparty both match
CONFIDENCE_AMOUNT_ONLY_MATCH: float = 0.80 # amount matches, counterparty not specified
CONFIDENCE_PARTIAL_MATCH: float = 0.6     # amount matches, counterparty mismatch
CONFIDENCE_AMBIGUOUS: float = 0.4
CONFIDENCE_INSUFFICIENT: float = 0.2
CONFIDENCE_CONTRADICTION: float = 0.85


# ===========================================================================
# Public API
# ===========================================================================


def evaluate_evidence(
    extracted: Dict[str, Any],
    transaction_history: List[Transaction],
) -> EvidenceVerdictResult:
    """Score every transaction against the extracted facts.

    Rules (in evaluation order):
      0. No history at all              → INSUFFICIENT_DATA
      1. Phishing case                  → INSUFFICIENT_DATA (no money to verify)
      2. No claimed amount              → INSUFFICIENT_DATA
      3. Exactly one strong match       → CONSISTENT
         (wrong_transfer + prior successes overrides to INCONSISTENT)
      4. Multiple strong matches, same  → INSUFFICIENT_DATA (ambiguous)
         counterparty                    (wrong_transfer override still applies)
      5. Multiple strong matches, diff  → INSUFFICIENT_DATA (ambiguous)
         counterparties
      6. Weak matches only              → INSUFFICIENT_DATA (partial confidence)
      7. No matches at all              → INSUFFICIENT_DATA (low confidence)
    """
    # Rule 0
    if not transaction_history:
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=None,
            confidence=CONFIDENCE_INSUFFICIENT,
        )

    # Rule 1
    if extracted.get("core_issue") == "phishing_or_social_engineering":
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=None,
            confidence=CONFIDENCE_INSUFFICIENT,
        )

    claimed_amount: Optional[float] = extracted.get("claimed_amount")
    claimed_counterparty: Optional[str] = (
        extracted.get("claimed_counterparty") or ""
    ).lower() or None

    # Rule 2
    if claimed_amount is None:
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=None,
            confidence=CONFIDENCE_INSUFFICIENT,
        )

    # When no counterparty was extractable from the complaint (Gemini returns
    # null, stub extracts a generic word like "number"), amount-matching
    # transactions are the best signal we have — treat them as strong matches.
    # The confidence is capped below CONFIDENCE_PERFECT_MATCH to reflect that
    # we couldn't verify the recipient.
    counterparty_unspecified = claimed_counterparty is None

    strong_matches: List[Transaction] = []
    weak_matches: List[Transaction] = []   # only populated when counterparty IS specified

    for tx in transaction_history:
        amount_ok = abs(tx.amount - claimed_amount) <= AMOUNT_TOLERANCE

        if counterparty_unspecified:
            # No counterparty to compare — amount match alone is the signal.
            if amount_ok:
                strong_matches.append(tx)
        else:
            party_ok = (
                claimed_counterparty in tx.counterparty.lower()
                or tx.counterparty.lower() in claimed_counterparty
            )
            if amount_ok and party_ok:
                strong_matches.append(tx)
            elif amount_ok:
                weak_matches.append(tx)

    # Rules 4 + 5: multiple strong matches.
    if len(strong_matches) >= 2:
        counterparties = {tx.counterparty.lower() for tx in strong_matches}
        if len(counterparties) == 1:
            most_recent = _pick_most_recent(strong_matches)
            # If the user is claiming a duplicate payment, finding multiple identical
            # transactions to the same counterparty is exactly what we expect.
            if extracted.get("core_issue") == "duplicate_payment":
                return EvidenceVerdictResult(
                    evidence_verdict=EvidenceVerdict.CONSISTENT,
                    relevant_transaction_id=most_recent.transaction_id,
                    confidence=CONFIDENCE_PERFECT_MATCH,
                )

            # All matches share the same counterparty — check contradiction rule.
            if extracted.get("core_issue") == "wrong_transfer" and claimed_counterparty:
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
        # Different counterparties — ambiguous, pick most recent.
        strongest = _pick_most_recent(strong_matches)
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=strongest.transaction_id,
            confidence=CONFIDENCE_AMBIGUOUS,
        )

    # Rule 3: exactly one strong match → CONSISTENT.
    if len(strong_matches) == 1:
        tx = strong_matches[0]
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
        # Use a lower confidence when the match is amount-only (no counterparty
        # was extracted from the complaint) to reflect that we couldn't verify
        # the recipient independently.
        confidence = (
            CONFIDENCE_AMOUNT_ONLY_MATCH
            if counterparty_unspecified
            else CONFIDENCE_PERFECT_MATCH
        )
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.CONSISTENT,
            relevant_transaction_id=tx.transaction_id,
            confidence=confidence,
        )

    # Rule 6: weak matches only.
    if weak_matches:
        tx = _pick_most_recent(weak_matches)
        return EvidenceVerdictResult(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
            relevant_transaction_id=tx.transaction_id,
            confidence=CONFIDENCE_PARTIAL_MATCH,
        )

    # Rule 7: nothing matched at all.
    return EvidenceVerdictResult(
        evidence_verdict=EvidenceVerdict.INSUFFICIENT_DATA,
        relevant_transaction_id=None,
        confidence=CONFIDENCE_INSUFFICIENT,
    )


# ===========================================================================
# Private helpers
# ===========================================================================


def _pick_most_recent(transactions: List[Transaction]) -> Transaction:
    """Return the transaction with the latest timestamp."""
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
        if tx.status != TransactionStatus.COMPLETED:
            continue
        if claimed_counterparty_lower in tx.counterparty.lower():
            count += 1
    return count
