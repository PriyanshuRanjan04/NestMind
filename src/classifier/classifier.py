"""
Intent Classifier — one LLM call per query.

Design decisions
----------------
1. **Structured output via JSON mode + response_format**
   We use `response_format={"type": "json_object"}` with an explicit JSON
   schema in the system prompt. This gives us reliable structured output
   without requiring function-calling, which keeps the token budget lower.

2. **System prompt as single source of truth**
   The entire taxonomy, entity vocabulary, context rules, and examples live
   in the system prompt so the classifier is self-contained. The user prompt
   is just the query + optional conversation history.

3. **Fallback on failure**
   If the LLM call fails or returns unparseable JSON, we fall back to
   agent=general_query with low confidence rather than crashing the request.
   This is the safe degradation path.

4. **Context window management**
   We keep only the last N user turns in the history to avoid blowing the
   context window on long sessions. N=6 is generous enough to handle all
   conversation fixture tests while keeping token cost predictable.

5. **No retry loop here**
   Retries are handled by the caller (pipeline.py) using CLASSIFIER_MAX_RETRIES.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI, APIError

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.schemas import (
    AgentName,
    ClassifierEntities,
    ClassifierResult,
    SafetyFlag,
    SafetyVerdict,
)

logger = logging.getLogger(__name__)

# Max prior user turns to include in context.
# 6 covers the deepest conversation fixture (4 turns) with headroom.
_MAX_HISTORY_TURNS = 6

# ---------------------------------------------------------------------------
# System prompt — classifier contract
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the intent classifier for NestMind, an AI co-investor platform for novice investors.

Your job is to analyze the user's query in the context of their conversation history and return a single structured JSON object. You make ONE decision per call. You never explain your reasoning — only return the JSON.

---

## AGENT TAXONOMY
Route to exactly one of these agents (use the exact string):

- portfolio_health       → user wants assessment of their own portfolio (concentration, performance, diversification, health check)
- market_research        → factual/recent info about a specific instrument, sector, index, or market event
- investment_strategy    → advice on what to buy/sell/rebalance/hedge, allocation guidance
- financial_planning     → long-term goals: retirement, education fund, house, FIRE, savings rate
- financial_calculator   → deterministic computation: DCA, mortgage, future value, FX conversion, tax
- risk_assessment        → risk metrics, beta, drawdown, stress test, what-if scenarios
- product_recommendation → recommend specific funds/ETFs matching user profile
- predictive_analysis    → forecasts, trend extrapolation, forward-looking analysis
- customer_support       → platform issues, login, account, transaction history, how-to
- general_query          → definitions, education, greetings, conversational, gibberish

---

## ENTITY EXTRACTION RULES
Extract only entities explicitly present or clearly resolvable from conversation context.

- tickers: array of strings, uppercase with exchange suffix where relevant (AAPL, ASML.AS, HSBA.L, 7203.T). Resolve common names: "apple"→"AAPL", "nvidia"→"NVDA", "microsoft"→"MSFT", "tesla"→"TSLA", "google"→"GOOGL", "amazon"→"AMZN", "amd"→"AMD", "meta"→"META", "barclays"→"BARC.L", "hsbc"→"HSBA.L", "toyota"→"7203.T", "asml"→"ASML.AS", "gold"→"GOLD". Tolerate typos: "microsfot"→"MSFT", "nvdia"→"NVDA", "appel"→"AAPL".
- amount: number in the unit of currency field
- currency: ISO 4217 string (USD, EUR, GBP, JPY, SGD)
- rate: decimal (0.08 for 8%, 0.065 for 6.5%)
- period_years: integer number of years
- frequency: one of exactly: daily, weekly, monthly, yearly
- horizon: one of exactly: 6_months, 1_year, 5_years
- time_period: one of exactly: today, this_week, this_month, this_year
- topics: array of strings (lowercase), for concepts, themes, or instruments without tickers
- sectors: array of strings (lowercase)
- index: one of exactly: S&P 500, FTSE 100, NIKKEI 225, MSCI World
- action: one of exactly: buy, sell, hold, hedge, rebalance
- goal: one of exactly: retirement, education, house, FIRE, emergency_fund

Only include a field if you have a value for it. Never include null or empty arrays.

---

## CONVERSATION CONTEXT RULES

CARRY context when:
- Current turn uses pronouns ("it", "them", "that") or vague references ("what about AMD?" after NVDA discussion → carry intent, switch ticker)
- Current turn is a follow-up action on a previously mentioned entity ("should I sell some?" after NVDA discussion → carry ticker NVDA)
- Current turn is a comparison ("compare them" → carry both tickers from prior turns)

DO NOT CARRY context when:
- Current turn introduces a completely new topic unrelated to prior turns
- Current turn is a greeting, thanks, or conversational closer ("thx", "thanks", "ok")
- Current turn has its own complete set of entities and clear intent

---

## MULTI-INTENT RULE
If the query contains two intents, pick the PRIMARY one:
- "how is my portfolio doing and what should I sell?" → portfolio_health (primary), extract action: sell
- "tell me about the markets and recommend a fund" → market_research (primary)

---

## SAFETY VERDICT
Return an informational safety verdict. This does NOT block the query. It is metadata only.

Categories: insider_trading, market_manipulation, money_laundering, guaranteed_returns, reckless_advice, sanctions_evasion, fraud, clean

---

## OUTPUT FORMAT
Return ONLY this JSON object. No explanation, no markdown, no preamble.

{
  "intent": "<short description of what the user wants>",
  "agent": "<exact agent string from taxonomy>",
  "entities": {
    // only fields with values
  },
  "safety_verdict": {
    "flag": "clean" | "<category>",
    "note": "<one sentence if flagged, empty string if clean>"
  },
  "confidence": <float 0.0–1.0>,
  "context_used": <true|false>
}

---

## EXAMPLES

Query: "hi"
{"intent":"greeting","agent":"general_query","entities":{},"safety_verdict":{"flag":"clean","note":""},"confidence":0.99,"context_used":false}

Query: "what's the price of AAPL right now?"
{"intent":"current price lookup for Apple","agent":"market_research","entities":{"tickers":["AAPL"],"time_period":"today"},"safety_verdict":{"flag":"clean","note":""},"confidence":0.98,"context_used":false}

Query: "if i invest 2500 monthly for 20 years at 8%, what will i have?"
{"intent":"future value calculation via DCA","agent":"financial_calculator","entities":{"amount":2500,"frequency":"monthly","period_years":20,"rate":0.08},"safety_verdict":{"flag":"clean","note":""},"confidence":0.97,"context_used":false}

Prior turns: ["tell me about NVDA"] | Query: "should I sell some?"
{"intent":"sell decision for NVDA carried from prior turn","agent":"investment_strategy","entities":{"tickers":["NVDA"],"action":"sell"},"safety_verdict":{"flag":"clean","note":""},"confidence":0.91,"context_used":true}

Query: "abcdefg"
{"intent":"unrecognizable input","agent":"general_query","entities":{},"safety_verdict":{"flag":"clean","note":""},"confidence":0.55,"context_used":false}
"""


def _build_user_message(
    query: str,
    history: Optional[list[str]] = None,
) -> str:
    """Build the user message with optional conversation history."""
    if not history:
        return f'Query: "{query}"'

    # Trim to last N turns
    trimmed = history[-_MAX_HISTORY_TURNS:]
    prior_block = "\n".join(f'- "{t}"' for t in trimmed)
    return f'Prior user turns:\n{prior_block}\n\nCurrent query: "{query}"'


async def classify(
    query: str,
    history: Optional[list[str]] = None,
    *,
    client: Optional[AsyncOpenAI] = None,
) -> ClassifierResult:
    """
    Classify *query* and return a ClassifierResult.

    Parameters
    ----------
    query   : The user's current message.
    history : Prior user turns in this session (oldest first).
    client  : Optional pre-built AsyncOpenAI client (for testing / injection).

    Raises
    ------
    Does NOT raise — on any failure returns a safe fallback result.
    """
    if client is None:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    user_message = _build_user_message(query, history)

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,   # deterministic classification
            max_tokens=512,    # classifier output is compact
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return _parse_result(data)

    except asyncio.TimeoutError as exc:
        logger.warning(
            "Classifier timeout — falling back to general_query | query=%r",
            query,
        )
        return _fallback_result("timeout")
    except json.JSONDecodeError as exc:
        logger.warning(
            "Classifier returned malformed JSON — falling back | error=%s | query=%r",
            exc,
            query,
        )
        return _fallback_result("parse_error")
    except APIError as exc:
        # RateLimitError is a subclass of APIError — check status code
        reason = "rate_limit" if getattr(exc, "status_code", None) == 429 else "api_error"
        logger.warning(
            "Classifier API error — falling back | reason=%s | status=%s | query=%r",
            reason,
            getattr(exc, "status_code", "?"),
            query,
        )
        return _fallback_result(reason)
    except Exception as exc:
        logger.warning(
            "Classifier unexpected error — falling back | error=%s | query=%r",
            exc,
            query,
        )
        return _fallback_result("api_error")


def _parse_result(data: dict) -> ClassifierResult:
    """Parse raw LLM JSON into a validated ClassifierResult."""
    try:
        agent_str = data.get("agent", "general_query")
        # Validate against enum — fall back on unknown values
        try:
            agent = AgentName(agent_str)
        except ValueError:
            logger.warning("Unknown agent '%s' — defaulting to general_query", agent_str)
            agent = AgentName.general_query

        raw_entities = data.get("entities", {}) or {}
        entities = ClassifierEntities(
            tickers=raw_entities.get("tickers") or None,
            amount=raw_entities.get("amount"),
            currency=raw_entities.get("currency"),
            rate=raw_entities.get("rate"),
            period_years=raw_entities.get("period_years"),
            frequency=raw_entities.get("frequency"),
            horizon=raw_entities.get("horizon"),
            time_period=raw_entities.get("time_period"),
            topics=raw_entities.get("topics") or None,
            sectors=raw_entities.get("sectors") or None,
            index=raw_entities.get("index"),
            action=raw_entities.get("action"),
            goal=raw_entities.get("goal"),
        )

        sv_raw = data.get("safety_verdict", {}) or {}
        try:
            flag = SafetyFlag(sv_raw.get("flag", "clean"))
        except ValueError:
            flag = SafetyFlag.clean

        safety_verdict = SafetyVerdict(
            flag=flag,
            note=sv_raw.get("note", ""),
        )

        confidence = float(data.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        return ClassifierResult(
            intent=str(data.get("intent", "unknown intent")),
            agent=agent,
            entities=entities,
            safety_verdict=safety_verdict,
            confidence=confidence,
            context_used=bool(data.get("context_used", False)),
        )

    except Exception as exc:
        logger.warning("Failed to parse classifier result: %s — data: %s", exc, data)
        return _fallback_result("parse_error")


def _fallback_result(reason: str = "api_error") -> ClassifierResult:
    """
    Safe fallback when classification fails for any reason.

    Returns the exact schema mandated by the spec:
      intent        = "classification_unavailable"
      agent         = general_query
      entities      = {}
      safety_verdict = {flag: clean, note: ""}
      confidence    = 0.0
      context_used  = false
      fallback      = true
      fallback_reason = reason  (timeout | api_error | parse_error | rate_limit)
    """
    return ClassifierResult(
        intent="classification_unavailable",
        agent=AgentName.general_query,
        entities=ClassifierEntities(),
        safety_verdict=SafetyVerdict(flag=SafetyFlag.clean, note=""),
        confidence=0.0,
        context_used=False,
        fallback=True,
        fallback_reason=reason,
    )
