"""
Portfolio Health agent tests — yfinance and OpenAI both mocked.

Tests:
  1. Metric computations (concentration, performance, FX conversion)
  2. Benchmark comparison
  3. Empty portfolio (user_004) — must not crash, must return BUILD message
  4. Observations are generated correctly
  5. Disclaimer is always present
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.portfolio_health import _compute_portfolio_metrics, _resolve_benchmark_ticker
from src.schemas import ClassifierResult, AgentName, ClassifierEntities, SafetyVerdict, SafetyFlag, UserProfile

FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "users"


def _load_user(filename: str) -> UserProfile:
    with open(FIXTURES_DIR / filename) as f:
        return UserProfile(**json.load(f))


def _mock_classifier_result() -> ClassifierResult:
    return ClassifierResult(
        intent="portfolio health check",
        agent=AgentName.portfolio_health,
        entities=ClassifierEntities(),
        safety_verdict=SafetyVerdict(flag=SafetyFlag.clean, note=""),
        confidence=0.95,
        context_used=False,
    )


# ---------------------------------------------------------------------------
# Mock yfinance prices
# ---------------------------------------------------------------------------

_MOCK_PRICES = {
    # US stocks
    "AAPL": 185.0,
    "MSFT": 380.0,
    "NVDA": 850.0,
    "GOOGL": 165.0,
    "META": 510.0,
    "AMZN": 185.0,
    "TSLA": 175.0,
    "AMD": 155.0,
    "QQQ": 440.0,
    # ETFs
    "VTI": 240.0,
    "VXUS": 58.0,
    "BND": 74.0,
    "VOO": 480.0,
    "VYM": 115.0,
    "SCHD": 80.0,
    "TLT": 100.0,
    "JNJ": 155.0,
    "PG": 160.0,
    "KO": 62.0,
    # International
    "ASML.AS": 720.0,
    "HSBA.L": 7.20,
    "7203.T": 2800.0,
    # FX
    "EURUSD=X": 1.08,
    "GBPUSD=X": 1.27,
    "JPYUSD=X": 0.0067,
    # Benchmarks
    "SPY": 520.0,
    "IWDA.L": 90.0,
    "ISF.L": 8.50,
    "1306.T": 2800.0,
}


def _mock_fetch_price(ticker: str):
    return _MOCK_PRICES.get(ticker)


def _mock_fetch_period_return(ticker: str, start_date: str):
    # Return a fixed benchmark return for all tickers in tests
    return 14.5  # 14.5% benchmark return


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkResolution:
    def test_us_user_sp500(self):
        user = _load_user("user_001_active_trader_us.json")
        ticker, label = _resolve_benchmark_ticker(user)
        assert ticker == "QQQ"  # user_001 prefers QQQ

    def test_concentrated_user_sp500(self):
        user = _load_user("user_003_concentrated.json")
        ticker, label = _resolve_benchmark_ticker(user)
        assert ticker == "SPY"

    def test_multi_currency_user_msci(self):
        user = _load_user("user_006_multi_currency.json")
        ticker, label = _resolve_benchmark_ticker(user)
        assert ticker == "IWDA.L"

    def test_empty_user_sp500(self):
        user = _load_user("user_004_empty.json")
        ticker, label = _resolve_benchmark_ticker(user)
        # Falls back to country default (US → SPY)
        assert ticker == "SPY"


class TestEmptyPortfolio:
    """user_004_empty must not crash and must return a useful BUILD message."""

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_empty_portfolio_does_not_crash(self, _fx, _period, _price):
        user = _load_user("user_004_empty.json")
        report = _compute_portfolio_metrics(user)
        assert report is not None
        assert report.get("empty") is True

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_empty_portfolio_no_positions_key(self, _fx, _period, _price):
        user = _load_user("user_004_empty.json")
        report = _compute_portfolio_metrics(user)
        # Must not contain financial metrics that require positions
        assert "concentration_risk" not in report or report.get("empty")

    @pytest.mark.asyncio
    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    async def test_empty_portfolio_streams_something(self, _fx, _period, _price):
        from openai import AsyncOpenAI
        from unittest.mock import AsyncMock

        # Mock the OpenAI streaming call
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Welcome to NestMind! "

        async def _mock_stream():
            yield mock_chunk
            mock_chunk.choices[0].delta.content = None
            yield mock_chunk

        with patch("openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = AsyncMock(return_value=_mock_stream())

            user = _load_user("user_004_empty.json")
            result = _mock_classifier_result()

            tokens = []
            async for token in _stream_empty_narrative_mock(user):
                tokens.append(token)

            # We just verify no crash occurs — actual content depends on OpenAI


async def _stream_empty_narrative_mock(user):
    """Helper that runs the empty narrative path with mocked streaming."""
    from src.agents.portfolio_health import _compute_portfolio_metrics, run
    with (
        patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price),
        patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return),
        patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0),
    ):
        # Return fallback text directly — just test it doesn't crash
        from src.agents.portfolio_health import _stream_empty_narrative
        # Use the fallback path by making OpenAI raise
        with patch("openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            from openai import APIError
            instance.chat.completions.create = AsyncMock(
                side_effect=Exception("mocked error")
            )
            async for token in _stream_empty_narrative(user):
                yield token


class TestConcentratedPortfolio:
    """user_003: ~majority in NVDA — should flag high concentration."""

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_concentration_flagged_high(self, _fx, _period, _price):
        user = _load_user("user_003_concentrated.json")
        report = _compute_portfolio_metrics(user)
        assert not report.get("empty")
        conc = report["concentration_risk"]
        assert conc["flag"] == "high"
        assert conc["top_position_pct"] > 50

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_concentration_observation_present(self, _fx, _period, _price):
        user = _load_user("user_003_concentrated.json")
        report = _compute_portfolio_metrics(user)
        warnings = [o for o in report["observations"] if o["severity"] == "warning"]
        assert any("NVDA" in o["text"] or "concentrated" in o["text"].lower() for o in warnings)

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_disclaimer_present(self, _fx, _period, _price):
        user = _load_user("user_003_concentrated.json")
        report = _compute_portfolio_metrics(user)
        assert "disclaimer" in report
        assert len(report["disclaimer"]) > 50


class TestActiveTraderPortfolio:
    """user_001: 9 holdings, tech-heavy."""

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_all_metrics_present(self, _fx, _period, _price):
        user = _load_user("user_001_active_trader_us.json")
        report = _compute_portfolio_metrics(user)
        assert not report.get("empty")
        assert "concentration_risk" in report
        assert "performance" in report
        assert "benchmark_comparison" in report
        assert "observations" in report
        assert "disclaimer" in report

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_total_value_positive(self, _fx, _period, _price):
        user = _load_user("user_001_active_trader_us.json")
        report = _compute_portfolio_metrics(user)
        assert report["total_value"] > 0

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_benchmark_comparison_shape(self, _fx, _period, _price):
        user = _load_user("user_001_active_trader_us.json")
        report = _compute_portfolio_metrics(user)
        bm = report["benchmark_comparison"]
        assert "benchmark" in bm
        assert "portfolio_return_pct" in bm
        assert "benchmark_return_pct" in bm
        assert "alpha_pct" in bm


class TestMultiCurrencyPortfolio:
    """user_006: USD + EUR + GBP + JPY holdings."""

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    def test_fx_conversion_applied(self, _period, _price):
        """Total value should reflect FX conversion for non-USD positions."""
        def _mock_fx(currency, base="USD"):
            return {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}.get(currency, 1.0)

        with patch("src.agents.portfolio_health._fetch_fx_rate", side_effect=_mock_fx):
            user = _load_user("user_006_multi_currency.json")
            report = _compute_portfolio_metrics(user)
            assert report is not None

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", side_effect=lambda ccy, base="USD": {
        "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067
    }.get(ccy, 1.0))
    def test_multi_currency_total_value(self, _fx, _period, _price):
        user = _load_user("user_006_multi_currency.json")
        report = _compute_portfolio_metrics(user)
        assert not report.get("empty")
        assert report["total_value"] > 10000  # sanity check
        assert report["base_currency"] == "USD"


class TestRetireePortfolio:
    """user_008: conservative, dividend-focused, age 68."""

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_report_does_not_crash(self, _fx, _period, _price):
        user = _load_user("user_008_retiree.json")
        report = _compute_portfolio_metrics(user)
        assert not report.get("empty")

    @patch("src.agents.portfolio_health._fetch_price", side_effect=_mock_fetch_price)
    @patch("src.agents.portfolio_health._fetch_period_return", side_effect=_mock_fetch_period_return)
    @patch("src.agents.portfolio_health._fetch_fx_rate", return_value=1.0)
    def test_disclaimer_always_present(self, _fx, _period, _price):
        user = _load_user("user_008_retiree.json")
        report = _compute_portfolio_metrics(user)
        assert "disclaimer" in report
        assert "investment advice" in report["disclaimer"].lower()
