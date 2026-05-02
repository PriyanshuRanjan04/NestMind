"""
Portfolio Health Check Agent — refactored to use structured JSON output.

Flow:
  1. Fetch live prices + FX rates via yfinance (sync, run in executor)
  2. Build the structured input payload matching the agent's input schema
  3. One LLM call (JSON mode) with the detailed system prompt → structured report
  4. Yield the JSON string as SSE tokens
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import yfinance as yf

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.schemas import ClassifierResult, UserProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (verbatim from product spec)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Portfolio Health Agent for NestMind, an AI co-investor platform. Your job is to give a novice investor a clear, honest, plain-language health check of their portfolio.

You will receive:
1. The user's portfolio data (positions, risk profile, base currency, preferred benchmark)
2. Real-time market data for each holding (current price, returns, sector)
3. Benchmark performance data

---

## YOUR JOB

Produce a structured JSON health report. Every field below is required.

{
  "summary": "<2-3 sentence plain English overview of the portfolio's overall health. No jargon without explanation. Surface the ONE or TWO things that matter most.>",

  "concentration_risk": {
    "top_position_pct": <float>,
    "top_3_positions_pct": <float>,
    "flag": "low" | "moderate" | "high",
    "flag_reason": "<one sentence explaining why>"
  },

  "performance": {
    "total_return_pct": <float>,
    "best_performer": {"ticker": "<>", "return_pct": <float>},
    "worst_performer": {"ticker": "<>", "return_pct": <float>}
  },

  "benchmark_comparison": {
    "benchmark": "<exact benchmark name>",
    "portfolio_return_pct": <float>,
    "benchmark_return_pct": <float>,
    "alpha_pct": <float>,
    "verdict": "outperforming" | "underperforming" | "in line"
  },

  "sector_exposure": [
    {"sector": "<sector name>", "pct": <float>}
  ],

  "observations": [
    {
      "severity": "warning" | "info" | "positive",
      "text": "<plain language observation. Max 2 sentences. Actionable where possible.>"
    }
  ],

  "disclaimer": "This is not financial or investment advice. NestMind provides information for educational purposes only. Past performance is not indicative of future results. Please consult a qualified financial adviser before making investment decisions."
}

---

## CONCENTRATION FLAGS
- top_position_pct > 40% → "high"
- top_position_pct 20–40% → "moderate"
- top_position_pct < 20% → "low"

---

## OBSERVATIONS RULES
- Maximum 5 observations, minimum 2
- Always surface the MOST IMPORTANT thing first
- Write for a novice: explain what the metric means if not obvious
- Be specific: use actual tickers, percentages, numbers
- Be actionable: suggest what to consider
- Severity: "warning" → could hurt them | "info" → neutral | "positive" → going well

---

## EMPTY PORTFOLIO RULE (CRITICAL)
If positions array is empty:
- Set all numeric fields to 0, flag to "low", sector_exposure to []
- summary must be encouraging BUILD-oriented message
- Observations must be BUILD-focused using risk_profile and age
- Still include disclaimer

---

## TONE & LANGUAGE RULES
- Plain English. Define financial terms inline.
- Never say "I cannot" or "I don't have access to"
- Never guarantee returns
- Use qualifiers: "you may want to consider", "it could be worth reviewing"
- Speak as "you" / "your portfolio"
- Retiree profiles: weight toward income, yield, capital preservation
- Aggressive profiles: acknowledge higher risk tolerance but still flag concentration
- Moderate profiles: balanced tone

---

## MULTI-CURRENCY RULE
If positions are in multiple currencies, note in an observation that FX movements affect returns.

---

Return ONLY the JSON object. No explanation, no markdown, no preamble."""

# ---------------------------------------------------------------------------
# Sector fallback map (avoids slow yfinance .info calls for common tickers)
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMD": "Technology",
    "ASML.AS": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "7203.T": "Consumer Discretionary",
    "JNJ": "Healthcare",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "HSBA.L": "Financials",
    "VTI": "Broad Market ETF", "VOO": "Broad Market ETF", "SPY": "Broad Market ETF",
    "QQQ": "Technology ETF", "VXUS": "International ETF",
    "VYM": "Dividend ETF", "SCHD": "Dividend ETF",
    "BND": "Fixed Income", "TLT": "Fixed Income", "AGG": "Fixed Income",
    "ISF.L": "Broad Market ETF", "IWDA.L": "Broad Market ETF", "1306.T": "Broad Market ETF",
    "GOLD": "Commodities",
}

_FX_TICKERS = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPYUSD=X", "SGD": "SGDUSD=X"}

_BENCHMARK_TICKER_MAP = {
    "S&P 500": "SPY", "QQQ": "QQQ", "MSCI World": "IWDA.L",
    "FTSE 100": "ISF.L", "NIKKEI 225": "1306.T",
}

_COUNTRY_BENCHMARK = {"US": "SPY", "GB": "ISF.L", "SG": "IWDA.L", "JP": "1306.T"}


def _get_sector(ticker: str) -> str:
    if ticker in _SECTOR_MAP:
        return _SECTOR_MAP[ticker]
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or info.get("category") or "Other"
    except Exception:
        return "Other"


def _fetch_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


def _fetch_fx_rate(currency: str, base: str = "USD") -> float:
    if currency == base:
        return 1.0
    ticker = _FX_TICKERS.get(currency)
    if not ticker:
        return 1.0
    price = _fetch_price(ticker)
    return price if price else 1.0


def _fetch_period_return(ticker: str, start_date: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(start=start_date)
        if len(hist) < 2:
            return None
        start = float(hist["Close"].iloc[0])
        end = float(hist["Close"].iloc[-1])
        return (end - start) / start * 100 if start else None
    except Exception:
        return None


def _resolve_benchmark(user: UserProfile) -> tuple[str, str]:
    """Return (ticker, display_name)."""
    pref = (user.preferences.preferred_benchmark or "").strip()
    if pref in _BENCHMARK_TICKER_MAP:
        return _BENCHMARK_TICKER_MAP[pref], pref
    if pref:
        return pref, pref
    ticker = _COUNTRY_BENCHMARK.get(user.country.upper(), "SPY")
    reverse = {v: k for k, v in _BENCHMARK_TICKER_MAP.items()}
    return ticker, reverse.get(ticker, ticker)


def _build_payload(user: UserProfile) -> dict:
    """
    Fetch all market data and build the structured agent input payload.
    Run this in a thread-pool executor to avoid blocking the event loop.
    """
    positions = user.positions
    base_ccy = user.base_currency or "USD"

    # FX rates
    currencies = list({p.currency for p in positions})
    fx_rates: dict[str, float] = {
        ccy: _fetch_fx_rate(ccy, base_ccy) for ccy in currencies
    }
    fx_rates[base_ccy] = 1.0

    # Per-position enrichment
    earliest_date = min((p.purchased_at for p in positions), default="2020-01-01")
    enriched = []
    total_cost_base = 0.0
    total_value_base = 0.0

    for pos in positions:
        fx = fx_rates.get(pos.currency, 1.0)
        current_price = _fetch_price(pos.ticker)
        cost_base = pos.quantity * pos.avg_cost * fx
        value_base = (pos.quantity * current_price * fx) if current_price else cost_base
        return_pct = ((value_base - cost_base) / cost_base * 100) if cost_base else 0.0
        sector = _get_sector(pos.ticker)

        total_cost_base += cost_base
        total_value_base += value_base

        enriched.append({
            "ticker": pos.ticker,
            "quantity": pos.quantity,
            "avg_cost": pos.avg_cost,
            "currency": pos.currency,
            "current_price": round(current_price, 4) if current_price else None,
            "current_value_base": round(value_base, 2),
            "return_pct": round(return_pct, 2),
            "sector": sector,
            "purchased_at": pos.purchased_at,
        })

    total_return_pct = (
        (total_value_base - total_cost_base) / total_cost_base * 100
        if total_cost_base else 0.0
    )

    # Benchmark
    bm_ticker, bm_name = _resolve_benchmark(user)
    bm_return = _fetch_period_return(bm_ticker, earliest_date) or 0.0

    return {
        "user": {
            "user_id": user.user_id,
            "name": user.name,
            "age": user.age,
            "risk_profile": user.risk_profile,
            "base_currency": base_ccy,
            "preferred_benchmark": bm_name,
        },
        "positions": enriched,
        "portfolio_summary": {
            "total_value_base": round(total_value_base, 2),
            "total_cost_base": round(total_cost_base, 2),
            "total_return_pct": round(total_return_pct, 2),
        },
        "benchmark_data": {
            "name": bm_name,
            "return_pct": round(bm_return, 2),
        },
        "fx_rates": {k: round(v, 6) for k, v in fx_rates.items()},
    }


async def _call_llm(payload: dict) -> dict:
    """
    Call OpenAI with the new system prompt.
    Returns the parsed structured health report dict.
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception as exc:
        logger.error("Portfolio health LLM call failed: %s", exc)
        return _fallback_report(payload)


def _fallback_report(payload: dict) -> dict:
    """Minimal valid report when LLM call fails."""
    ps = payload.get("portfolio_summary", {})
    positions = payload.get("positions", [])
    bm = payload.get("benchmark_data", {})
    name = payload.get("user", {}).get("name", "there")

    if not positions:
        return {
            "summary": (
                f"Welcome, {name}! Your account is ready — you have no positions yet. "
                "Consider starting with a low-cost diversified index fund."
            ),
            "concentration_risk": {"top_position_pct": 0, "top_3_positions_pct": 0, "flag": "low", "flag_reason": "No positions held."},
            "performance": {"total_return_pct": 0, "best_performer": {"ticker": "N/A", "return_pct": 0}, "worst_performer": {"ticker": "N/A", "return_pct": 0}},
            "benchmark_comparison": {"benchmark": bm.get("name", "N/A"), "portfolio_return_pct": 0, "benchmark_return_pct": 0, "alpha_pct": 0, "verdict": "in line"},
            "sector_exposure": [],
            "observations": [{"severity": "info", "text": "You have no investments yet. Start by defining your goal and time horizon, then consider a broad-market index fund as your foundation."}],
            "disclaimer": "This is not financial or investment advice. NestMind provides information for educational purposes only. Past performance is not indicative of future results. Please consult a qualified financial adviser before making investment decisions.",
        }

    sorted_pos = sorted(positions, key=lambda x: x["current_value_base"], reverse=True)
    total = ps.get("total_value_base", 1) or 1
    top1 = sorted_pos[0]["current_value_base"] / total * 100 if sorted_pos else 0
    flag = "high" if top1 > 40 else ("moderate" if top1 > 20 else "low")
    ret = ps.get("total_return_pct", 0)
    bm_ret = bm.get("return_pct", 0)

    return {
        "summary": f"Your portfolio has a total return of {ret:+.1f}%. Health check data is temporarily limited — please try again for a full analysis.",
        "concentration_risk": {"top_position_pct": round(top1, 1), "top_3_positions_pct": 0, "flag": flag, "flag_reason": "Computed from position values."},
        "performance": {"total_return_pct": round(ret, 2), "best_performer": {"ticker": "N/A", "return_pct": 0}, "worst_performer": {"ticker": "N/A", "return_pct": 0}},
        "benchmark_comparison": {"benchmark": bm.get("name", "N/A"), "portfolio_return_pct": round(ret, 2), "benchmark_return_pct": round(bm_ret, 2), "alpha_pct": round(ret - bm_ret, 2), "verdict": "outperforming" if ret > bm_ret else "underperforming"},
        "sector_exposure": [],
        "observations": [{"severity": "info", "text": "Full analysis temporarily unavailable. Your portfolio metrics have been computed but narrative observations could not be generated."}],
        "disclaimer": "This is not financial or investment advice. NestMind provides information for educational purposes only. Past performance is not indicative of future results. Please consult a qualified financial adviser before making investment decisions.",
    }


async def run(
    user: UserProfile,
    classifier_result: ClassifierResult,
) -> AsyncIterator[str]:
    """
    Main entry point. Yields the structured health report as a JSON string.
    """
    loop = asyncio.get_event_loop()

    # Step 1: Fetch market data (blocking I/O → thread pool)
    payload = await loop.run_in_executor(None, _build_payload, user)

    # Step 2: LLM call → structured health report
    report = await _call_llm(payload)

    # Store for pipeline to access
    run._last_report = report

    # Step 3: Yield the JSON string (pipeline wraps in SSE token events)
    yield json.dumps(report, indent=2)


run._last_report: dict = {}
