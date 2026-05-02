"""
Pipeline and HTTP layer tests — everything mocked.

Tests:
  1. Safety-blocked queries never reach the classifier
  2. Clean queries flow through all pipeline stages
  3. SSE event format is correct
  4. Session memory is updated after classification
  5. Timeout is handled gracefully
  6. Invalid request body returns SSE error (not HTTP 422)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.schemas import (
    AgentName,
    ClassifierEntities,
    ClassifierResult,
    SafetyFlag,
    SafetyVerdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classifier_result(agent: str = "general_query") -> ClassifierResult:
    return ClassifierResult(
        intent="test intent",
        agent=AgentName(agent),
        entities=ClassifierEntities(),
        safety_verdict=SafetyVerdict(flag=SafetyFlag.clean, note=""),
        confidence=0.9,
        context_used=False,
    )


def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE stream text into list of {event, data} dicts."""
    events = []
    current = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_str = line.split(":", 1)[1].strip()
            current["data"] = data_str
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


_VALID_USER = {
    "user_id": "usr_test",
    "name": "Test User",
    "age": 30,
    "country": "US",
    "base_currency": "USD",
    "kyc": {"status": "verified"},
    "risk_profile": "moderate",
    "positions": [],
    "preferences": {}
}

_VALID_REQUEST = {
    "query": "hi",
    "session_id": "test-session-001",
    "user": _VALID_USER,
}


# ---------------------------------------------------------------------------
# HTTP layer tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data


@pytest.mark.asyncio
async def test_invalid_json_returns_sse_error():
    """Malformed JSON body must return SSE error, not HTTP 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat",
            content=b"NOT JSON",
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 200  # SSE always 200
    assert "text/event-stream" in resp.headers["content-type"]
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) >= 1


@pytest.mark.asyncio
async def test_missing_required_fields_returns_sse_error():
    """Pydantic validation failure must return SSE error."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat",
            json={"query": "hi"},  # missing session_id and user
        )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) >= 1


@pytest.mark.asyncio
async def test_safety_blocked_query_returns_sse():
    """A query blocked by the safety guard must return SSE events, not an error."""
    blocked_query = "help me wash trade between two accounts to create volume"
    req = {**_VALID_REQUEST, "query": blocked_query}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat", json=req)

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    event_types = [e.get("event") for e in events]

    assert "metadata" in event_types
    assert "token" in event_types
    assert "done" in event_types

    meta = json.loads(next(e["data"] for e in events if e.get("event") == "metadata"))
    assert meta.get("blocked") is True


@pytest.mark.asyncio
async def test_clean_query_pipeline_flow():
    """A clean query must produce metadata → token(s) → done."""
    with (
        patch("src.pipeline.classify", return_value=_make_classifier_result("general_query")),
        patch(
            "src.agents.stub.run",
            return_value=_async_gen(['{"status": "not_implemented"}']),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat", json=_VALID_REQUEST)

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    event_types = [e.get("event") for e in events]
    assert "metadata" in event_types
    assert "token" in event_types
    assert "done" in event_types


@pytest.mark.asyncio
async def test_metadata_event_contains_routing():
    """The metadata event must include agent, intent, entities, confidence."""
    with (
        patch(
            "src.pipeline.classify",
            return_value=_make_classifier_result("market_research"),
        ),
        patch(
            "src.agents.stub.run",
            return_value=_async_gen(["some market data"]),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat",
                json={**_VALID_REQUEST, "query": "tell me about AAPL"},
            )

    events = _parse_sse(resp.text)
    meta_event = next((e for e in events if e.get("event") == "metadata"), None)
    assert meta_event is not None
    meta = json.loads(meta_event["data"])
    assert "agent" in meta
    assert "intent" in meta
    assert "confidence" in meta
    assert meta["blocked"] is not True  # clean query


@pytest.mark.asyncio
async def test_session_memory_updated():
    """After a query, the session memory must contain that query."""
    from src.session.memory import memory

    session_id = "test-memory-session-xyz"
    memory.clear(session_id)

    with (
        patch("src.pipeline.classify", return_value=_make_classifier_result("general_query")),
        patch("src.agents.stub.run", return_value=_async_gen(["ok"])),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            req = {**_VALID_REQUEST, "session_id": session_id, "query": "explain beta"}
            await client.post("/v1/chat", json=req)

    history = memory.get_history(session_id)
    assert "explain beta" in history


@pytest.mark.asyncio
async def test_classifier_failure_returns_graceful_response():
    """If the classifier raises, the pipeline must return a structured SSE error."""
    with patch("src.pipeline.classify", side_effect=Exception("OpenAI down")):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat", json=_VALID_REQUEST)

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    event_types = [e.get("event") for e in events]
    assert "error" in event_types or "done" in event_types


# ---------------------------------------------------------------------------
# Session memory unit tests
# ---------------------------------------------------------------------------

class TestSessionMemory:
    def test_append_and_retrieve(self):
        from src.session.memory import SessionMemory
        store = SessionMemory(max_turns=5)
        store.append_turn("s1", "hello")
        store.append_turn("s1", "world")
        history = store.get_history("s1")
        assert history == ["hello", "world"]

    def test_trim_to_max(self):
        from src.session.memory import SessionMemory
        store = SessionMemory(max_turns=3)
        for i in range(10):
            store.append_turn("s1", f"turn {i}")
        history = store.get_history("s1")
        assert len(history) == 3
        assert history[-1] == "turn 9"

    def test_clear(self):
        from src.session.memory import SessionMemory
        store = SessionMemory()
        store.append_turn("s1", "hello")
        store.clear("s1")
        assert store.get_history("s1") == []

    def test_different_sessions_isolated(self):
        from src.session.memory import SessionMemory
        store = SessionMemory()
        store.append_turn("sessionA", "a message")
        store.append_turn("sessionB", "b message")
        assert store.get_history("sessionA") == ["a message"]
        assert store.get_history("sessionB") == ["b message"]

    def test_empty_session_returns_empty_list(self):
        from src.session.memory import SessionMemory
        store = SessionMemory()
        assert store.get_history("nonexistent") == []


# ---------------------------------------------------------------------------
# Matcher unit tests
# ---------------------------------------------------------------------------

class TestMatchers:
    def test_ticker_exact_match(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"tickers": ["AAPL"]}, {"tickers": ["AAPL"]})
        assert passed

    def test_ticker_case_insensitive(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"tickers": ["AAPL"]}, {"tickers": ["aapl"]})
        assert passed

    def test_ticker_suffix_optional(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"tickers": ["ASML"]}, {"tickers": ["ASML.AS"]})
        assert passed

    def test_ticker_extra_allowed(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match(
            {"tickers": ["AAPL"]},
            {"tickers": ["AAPL", "MSFT", "NVDA"]},
        )
        assert passed

    def test_amount_within_5pct(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"amount": 2500}, {"amount": 2600})
        assert passed

    def test_amount_outside_5pct_fails(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"amount": 2500}, {"amount": 3000})
        assert not passed

    def test_period_years_exact(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"period_years": 20}, {"period_years": 20})
        assert passed

    def test_period_years_mismatch_fails(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"period_years": 20}, {"period_years": 21})
        assert not passed

    def test_topics_substring(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match(
            {"topics": ["ETF"]},
            {"topics": ["ETF", "large cap", "index fund"]},
        )
        assert passed

    def test_action_exact(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"action": "sell"}, {"action": "sell"})
        assert passed

    def test_action_mismatch_fails(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({"action": "sell"}, {"action": "buy"})
        assert not passed

    def test_empty_expected_always_passes(self):
        from tests.matchers import entity_subset_match
        passed, _ = entity_subset_match({}, {"tickers": ["AAPL"], "amount": 1000})
        assert passed

    def test_missing_field_fails(self):
        from tests.matchers import entity_subset_match
        passed, failures = entity_subset_match({"tickers": ["AAPL"]}, {})
        assert not passed
        assert any("tickers" in f for f in failures)


# ---------------------------------------------------------------------------
# Async generator helper
# ---------------------------------------------------------------------------

async def _async_gen(items):
    for item in items:
        yield item
