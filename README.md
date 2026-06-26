# QueueStorm Investigator

An AI-powered support ticketing analysis engine built for the SUST CSE Carnival 2026. This service processes free-text customer complaints, compares them against transaction history, verifies evidence, and drafts safe, actionable replies for human agents.

## Required Deliverables Status
- ✅ **API Schema Correctness**: Full support for `complaint`, `ticket_id`, and `language` fields with rigorous validation.
- ✅ **Evidence Engine**: Deterministic logic handles wrong transfers, failed payments, duplicate payments, and unverified phishing attempts.
- ✅ **Safety Logic**: The output pipeline scrubs unsafe refund promises and prevents credential harvesting. 
- ✅ **Language Support**: Seamless translation to Bangla for `customer_reply` based on user payload language.

## Architecture & Tech Stack
- **FastAPI**: High-performance asynchronous API framework.
- **Pydantic**: Strict schema validation.
- **Google GenAI SDK (Gemini)**: Extracts structured data from raw complaints and generates localized responses.

## Setup Instructions

1. **Clone and create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Copy `.env.example` to `.env` and add your Gemini API Key.
   ```bash
   cp .env.example .env
   # Edit .env and set GEMINI_API_KEY
   ```

4. **Run the API:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## AI Approach and Safety Logic
- **Extraction**: The `call_llm_extractor` function uses Gemini (via `google-genai`) to parse free-text complaints into structured variables (`claimed_amount`, `claimed_counterparty`, `core_issue`). 
- **Deterministic Evidence Rule Engine**: We strictly decouple the LLM from the decision-making process. The `evidence.py` engine compares the LLM's extracted facts against the explicit `transaction_history`.
- **Safety Guardrails**: A dedicated `safety.py` pass intercepts the drafted `customer_reply` before it is dispatched to the user. It explicitly censors requests for PIN/OTP/passwords and replaces unauthorized refund promises with safe language ("Any eligible amount will be returned through official channels").

## MODELS

**Model Used:** `gemini-2.5-flash`
- **Where it runs:** `app/pipeline/extractor.py` and `app/pipeline/drafter.py`
- **Why it was chosen:** Flash is extremely fast, cost-effective, and highly capable at structured JSON extraction. It also provides excellent native multilingual support, which is critical for handling Bangla complaints (`SAMPLE-07`). It ensures we meet the sub-5 second p95 latency requirement without blowing out API limits.

## Known Limitations
- The system heavily relies on the deterministic stub if the Gemini API is unreachable.
- The `wrong_transfer` core issue relies on finding the specific transaction amount to trigger the contradiction threshold rule.
