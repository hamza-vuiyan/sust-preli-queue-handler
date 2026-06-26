"""
pipeline/drafter.py
===================
Step C (part 2) — Template-based response drafting.

Each case type has its own `_draft_*` function that generates:
  - agent_summary          : one paragraph for the human agent
  - recommended_next_action: what the agent should do next
  - customer_reply         : outbound message (UN-guarded — safety.py is last)

To add a new case type: write a `_draft_<name>` function and register it
in `_DRAFTERS`. The `draft_response` dispatcher picks it up automatically.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.schemas import CaseType, EvidenceVerdict, Severity
from app.pipeline.evidence import EvidenceVerdictResult
from app.pipeline.client import get_llm_client, get_gemini_model

logger = logging.getLogger("queuestorm.pipeline.drafter")


# ===========================================================================
# Public API
# ===========================================================================


def draft_response(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    case_type: CaseType,
    severity: Severity,
    language: Optional[str] = None,
) -> Dict[str, str]:
    """Dispatch to the correct drafter and return the three reply fields."""
    handler = _DRAFTERS.get(case_type, _draft_other)
    result = handler(extracted, evidence, severity)
    
    if language in ("bn", "mixed"):
        result["customer_reply"] = _translate_to_bangla(result["customer_reply"])
        
    return result

def _translate_to_bangla(english_text: str) -> str:
    client = get_llm_client()
    if not client:
        return english_text
        
    from google.genai import types  # type: ignore
    try:
        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=f"Translate the following customer support reply to formal Bangla. Maintain the exact same meaning, especially regarding safety (do not promise refunds if not promised, keep warnings about OTP/PIN). Only output the translated text, no markdown or extra commentary.\n\n{english_text}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
            ),
        )
        translated = (getattr(response, "text", "") or "").strip()
        if translated:
            return translated
    except Exception as e:
        logger.warning("Translation to Bangla failed: %s", e)
    
    return english_text


# ===========================================================================
# Per-case drafter functions
# ===========================================================================


def _draft_wrong_transfer(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    amount      = extracted.get("claimed_amount")
    counterparty = extracted.get("claimed_counterparty") or "the recipient"
    tx_ref      = evidence.relevant_transaction_id or "N/A"

    if evidence.evidence_verdict == EvidenceVerdict.CONSISTENT:
        return {
            "agent_summary": (
                f"Customer reports a wrong transfer of {amount} to {counterparty}. "
                f"Transaction {tx_ref} matches the claim on both amount and "
                f"counterparty. Recommend opening a dispute for review."
            ),
            "recommended_next_action": (
                f"Open dispute case for {tx_ref}; freeze any pending settlement "
                f"to the counterparty and request callback within 2 hours."
            ),
            "customer_reply": (
                f"Thanks for reaching out — we found a transfer matching your "
                f"description (reference {tx_ref}). We will refund you the eligible "
                f"amount after our dispute team reviews the case. Please do not "
                f"share your PIN, OTP, or password with anyone while we work on this."
            ),
        }

    if evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        return {
            "agent_summary": (
                f"Customer claims a wrong transfer to {counterparty}, but history "
                f"shows multiple prior successful transfers to the same counterparty. "
                f"Treat the claim with caution."
            ),
            "recommended_next_action": (
                "Verify the customer's identity with two security questions "
                "before processing any reversal."
            ),
            "customer_reply": (
                "Thanks for getting in touch. We've flagged your case for manual "
                "review and will get back to you within one business day."
            ),
        }

    # INSUFFICIENT_DATA
    return {
        "agent_summary": (
            f"Customer claims a wrong transfer of {amount} to {counterparty}, "
            f"but we could not confirm a unique matching transaction."
        ),
        "recommended_next_action": (
            "Ask the customer for the exact transaction id or timestamp."
        ),
        "customer_reply": (
            "Thanks for reaching out. To help us locate the transfer quickly, "
            "please reply with the exact transaction reference from your app "
            "or the time and date of the transfer."
        ),
    }


def _draft_phishing(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer reports a possible phishing / social-engineering attempt. "
            "Treat as CRITICAL — escalate to fraud_risk immediately."
        ),
        "recommended_next_action": (
            "Lock outgoing transfers on the account for 24h, raise a fraud-risk "
            "ticket, and call the customer on their registered number to confirm."
        ),
        "customer_reply": (
            "Thank you for flagging this. We will never ask for your PIN, OTP, "
            "or password over the phone or by message. Our team will contact you "
            "through our official channels to help secure your account."
        ),
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

    return {
        "agent_summary": agent_summary,
        "recommended_next_action": next_action,
        "customer_reply": (
            "Thanks for letting us know. We're checking the transaction now and "
            "will update you as soon as we have confirmation."
        ),
    }


def _draft_refund_request(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer is requesting a refund. Forward to customer_support for manual review."
        ),
        "recommended_next_action": (
            "Verify the original transaction and the merchant's refund policy."
        ),
        "customer_reply": (
            "Thanks for reaching out. We've recorded your refund request and our "
            "team will review the case and follow up with the outcome."
        ),
    }


def _draft_duplicate_payment(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer reports being charged twice for the same transaction. "
            "Verify duplicate id before processing a reversal."
        ),
        "recommended_next_action": (
            "Compare gateway and ledger records for the disputed window."
        ),
        "customer_reply": (
            "Thanks for getting in touch. We're checking for any duplicate "
            "charges and will update you shortly."
        ),
    }


def _draft_merchant_delay(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer is reporting a merchant settlement delay. Route to "
            "merchant_operations for partner follow-up."
        ),
        "recommended_next_action": (
            "Contact the merchant partner and request settlement status."
        ),
        "customer_reply": (
            "Thanks for your patience. We're following up with the merchant and "
            "will share an update as soon as we hear back."
        ),
    }


def _draft_agent_cashin(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer reports an issue with an agent cash-in. Route to "
            "agent_operations for on-the-ground follow-up."
        ),
        "recommended_next_action": (
            "Verify the agent's cash-in ledger and contact the agent within 1h."
        ),
        "customer_reply": (
            "Thanks for letting us know. We've passed this to our agent "
            "operations team and they will reach out shortly."
        ),
    }


def _draft_other(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer complaint did not match a known case type. Manual review required."
        ),
        "recommended_next_action": "Assign to customer_support tier-2 for triage.",
        "customer_reply": (
            "Thanks for reaching out. We've received your message and a "
            "specialist will follow up shortly."
        ),
    }


# ===========================================================================
# Dispatch table — maps CaseType to its drafter function.
# To add a new case type: add an entry here + write its _draft_* function.
# ===========================================================================
_DRAFTERS = {
    CaseType.WRONG_TRANSFER:                 _draft_wrong_transfer,
    CaseType.PAYMENT_FAILED:                 _draft_payment_failed,
    CaseType.REFUND_REQUEST:                 _draft_refund_request,
    CaseType.DUPLICATE_PAYMENT:              _draft_duplicate_payment,
    CaseType.MERCHANT_SETTLEMENT_DELAY:      _draft_merchant_delay,
    CaseType.AGENT_CASH_IN_ISSUE:            _draft_agent_cashin,
    CaseType.PHISHING_OR_SOCIAL_ENGINEERING: _draft_phishing,
}
