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
                f"Customer reports sending {amount} to {counterparty}. "
                f"Transaction {tx_ref} matches the claim. Dispute initiated for review."
            ),
            "recommended_next_action": (
                f"Verify {tx_ref} details with the customer and initiate "
                f"the wrong-transfer dispute workflow per policy."
            ),
            "customer_reply": (
                f"We have noted your concern about transaction {tx_ref}. "
                f"Please do not share your PIN or OTP with anyone. "
                f"Our dispute team will review the case and contact you through official support channels."
            ),
        }

    if evidence.evidence_verdict == EvidenceVerdict.INCONSISTENT:
        return {
            "agent_summary": (
                f"Customer claims a wrong transfer to {counterparty}, but history "
                f"shows multiple prior transfers to the same counterparty. "
                f"Evidence is inconsistent — flag for human review."
            ),
            "recommended_next_action": (
                "Flag for human review. Verify with the customer whether this was genuinely "
                "a wrong transfer given the established transaction pattern with this recipient."
            ),
            "customer_reply": (
                f"We have received your request regarding the recent transaction. "
                f"Please do not share your PIN or OTP with anyone. "
                f"Our dispute team will review the case carefully and contact you through official support channels."
            ),
        }

    # INSUFFICIENT_DATA
    return {
        "agent_summary": (
            f"Customer claims a wrong transfer of {amount} to {counterparty}, "
            f"but we could not confirm a unique matching transaction in the provided history."
        ),
        "recommended_next_action": (
            "Reply to customer asking for the exact transaction ID or timestamp."
        ),
        "customer_reply": (
            "Thank you for reaching out. To help us locate the transfer quickly, "
            "please share the transaction ID or the exact time and date. "
            "Please do not share your PIN or OTP with anyone."
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
            "Escalate to fraud_risk team immediately. Confirm to customer that the company never asks for OTP. "
            "Log the reported number for fraud pattern analysis."
        ),
        "customer_reply": (
            "Thank you for reaching out before sharing any information. "
            "We never ask for your PIN, OTP, or password under any circumstances. "
            "Please do not share these with anyone, even if they claim to be from us. "
            "Our fraud team has been notified of this incident."
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
            f"Customer reports a payment that they believe failed, but balance was deducted. "
            f"Transaction {tx_ref} matches the claim. Recommend reconciliation."
        )
        next_action = (
            f"Investigate {tx_ref} ledger status. If balance was deducted on a failed payment, "
            f"initiate the automatic reversal flow within standard SLA."
        )
        customer_reply = (
            f"We have noted that transaction {tx_ref} may have caused an unexpected balance deduction. "
            f"Our payments team will review the case and any eligible amount will be returned "
            f"through official channels. Please do not share your PIN or OTP with anyone."
        )
    else:
        agent_summary = (
            "Customer reports a payment failure. No matching failed "
            "transaction found in the supplied history."
        )
        next_action = "Pull the gateway-level status for the most recent transaction."
        customer_reply = (
            "Thank you for reaching out. We are checking the payment status and "
            "will update you as soon as we have confirmation. "
            "Please do not share your PIN or OTP with anyone."
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
    tx_ref = evidence.relevant_transaction_id or "the transaction"
    return {
        "agent_summary": (
            f"Customer requests refund of payment ({tx_ref}) due to change of mind or dissatisfaction. Not a service failure."
        ),
        "recommended_next_action": (
            "Inform the customer that refund eligibility depends on the merchant's own policy. "
            "Provide guidance on contacting the merchant directly for a refund."
        ),
        "customer_reply": (
            "Thank you for reaching out. Refunds for completed payments depend on the merchant's own policy. "
            "We recommend contacting the merchant directly. "
            "If you need help reaching them, please reply and we will guide you. "
            "Please do not share your PIN or OTP with anyone."
        ),
    }


def _draft_duplicate_payment(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    tx_ref = evidence.relevant_transaction_id or "the transaction"
    return {
        "agent_summary": (
            f"Customer reports a possible duplicate payment. Transaction {tx_ref} is suspected duplicate. "
            "Verify with payments_ops before processing any reversal."
        ),
        "recommended_next_action": (
            f"Verify the duplicate with payments_ops. If the biller confirms only one payment was received, "
            f"initiate reversal of {tx_ref}."
        ),
        "customer_reply": (
            f"We have noted the possible duplicate payment for transaction {tx_ref}. "
            f"Our payments team will verify with the biller and any eligible amount will be returned "
            f"through official channels. Please do not share your PIN or OTP with anyone."
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
    tx_ref = evidence.relevant_transaction_id or "the transaction"
    return {
        "agent_summary": (
            f"Customer reports agent cash-in ({tx_ref}) not reflected in balance. "
            "Route to agent_operations for on-the-ground follow-up."
        ),
        "recommended_next_action": (
            f"Investigate {tx_ref} pending status with agent operations. "
            "Confirm settlement state and resolve within the standard cash-in SLA."
        ),
        "customer_reply": (
            f"We have noted your concern about the cash-in transaction. "
            f"Our agent operations team will investigate and contact you through official channels. "
            f"Please do not share your PIN or OTP with anyone."
        ),
    }


def _draft_other(
    extracted: Dict[str, Any],
    evidence: EvidenceVerdictResult,
    severity: Severity,
) -> Dict[str, str]:
    return {
        "agent_summary": (
            "Customer complaint did not match a specific case type. Insufficient detail — manual review required."
        ),
        "recommended_next_action": (
            "Reply to customer asking for specific details: which transaction, what amount, what went wrong, and approximate time."
        ),
        "customer_reply": (
            "Thank you for reaching out. To help you faster, please share the transaction ID, "
            "the amount involved, and a short description of what went wrong. "
            "Please do not share your PIN or OTP with anyone."
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
