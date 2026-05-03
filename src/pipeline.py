"""
Pipeline orchestrator — the spine of the system.

Request flow:
  1. Safety guard (synchronous, no LLM, must complete < 10 ms)
  2. Intent classifier (one LLM call, structured output)
  3. Session memory update (append current turn)
  4. Agent dispatch (portfolio_health fully implemented; others → stub)
  5. SSE stream back to client

The pipeline wraps the entire flow in an asyncio.timeout to enforce the
configured PIPELINE_TIMEOUT_S budget. Any timeout is converted to a
structured SSE error event, not a stack trace.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from src import config
from src.agents import portfolio_health, stub
from src.classifier.classifier import classify
from src.safety.guard import check as safety_check
from src.schemas import (
    AgentName,
    ChatRequest,
    ClassifierEntities,
    ClassifierResult,
    SafetyFlag,
    SafetyVerdict,
    SSEEventType,
    UserProfile,
)
from src.session.memory import memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(event_type: SSEEventType, data: str) -> str:
    """Format a Server-Sent Events frame."""
    return f"event: {event_type.value}\ndata: {data}\n\n"


def _sse_token(text: str) -> str:
    return _sse_event(SSEEventType.token, text)


def _sse_metadata(payload: dict) -> str:
    return _sse_event(SSEEventType.metadata, json.dumps(payload))


def _sse_done() -> str:
    return _sse_event(SSEEventType.done, "")


def _sse_error(message: str, code: str = "PIPELINE_ERROR") -> str:
    return _sse_event(
        SSEEventType.error,
        json.dumps({"code": code, "message": message}),
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

async def _dispatch(
    user: UserProfile,
    classifier_result: ClassifierResult,
) -> AsyncIterator[str]:
    """Route to the correct agent and yield token strings."""
    agent = classifier_result.agent

    if agent == AgentName.portfolio_health:
        async for token in portfolio_health.run(user, classifier_result):
            yield token
    else:
        async for token in stub.run(user, classifier_result, agent.value):
            yield token


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(request: ChatRequest) -> AsyncIterator[str]:
    """
    Full request pipeline. Yields SSE-formatted strings.

    Always yields at least one 'done' or 'error' event — callers can rely on
    this for stream termination.
    """
    query = request.query.strip()
    session_id = request.session_id
    user = request.user

    # ------------------------------------------------------------------
    # Step 1 — Safety guard (synchronous, pre-LLM)
    # ------------------------------------------------------------------
    guard_result = safety_check(query)

    if guard_result.blocked:
        logger.info(
            "Query blocked by safety guard: category=%s session=%s",
            guard_result.category,
            session_id,
        )
        yield _sse_metadata({
            "blocked": True,
            "category": guard_result.category.value,
        })
        yield _sse_token(guard_result.message)
        yield _sse_done()
        return

    # ------------------------------------------------------------------
    # Step 2 — Retrieve session history
    # ------------------------------------------------------------------
    history = memory.get_history(session_id)

    # ------------------------------------------------------------------
    # Step 3 — Intent classification (one LLM call)
    # ------------------------------------------------------------------
    # Two-layer protection:
    #   Inner layer: classify()'s own try/except catches OpenAI SDK errors.
    #   Outer layer: catches the case where classify itself raises
    #     (e.g. when patched in tests via side_effect, import failure, etc.).
    # Both layers are intentional — do not remove either.
    try:
        async with asyncio.timeout(config.PIPELINE_TIMEOUT_S):
            classifier_result: ClassifierResult = await classify(
                query, history
            )
    except asyncio.TimeoutError:
        logger.warning(
            "Classifier timed out | session=%s | query=%r",
            session_id,
            query,
        )
        yield _sse_error(
            "Request timed out during classification. Please try again.",
            code="TIMEOUT",
        )
        yield _sse_done()
        return
    except Exception as exc:
        # classify() raised unexpectedly (e.g. mock side_effect in tests,
        # or an unforeseen initialisation error). Build a valid fallback
        # result and continue — the pipeline will stream the graceful
        # user message via the fallback path below (step 4b).
        logger.error(
            "classify() raised unexpectedly | session=%s | query=%r | error=%s",
            session_id,
            query,
            exc,
        )
        classifier_result = ClassifierResult(
            intent="classification_unavailable",
            agent=AgentName.general_query,
            entities=ClassifierEntities(),
            safety_verdict=SafetyVerdict(flag=SafetyFlag.clean, note=""),
            confidence=0.0,
            context_used=False,
            fallback=True,
            fallback_reason=f"classify_raised: {exc}",
        )

    # ------------------------------------------------------------------
    # Step 4 — Emit metadata SSE event (routing decision)
    # ------------------------------------------------------------------
    metadata_payload: dict = {
        "intent": classifier_result.intent,
        "agent": classifier_result.agent.value,
        "entities": classifier_result.entities.to_dict(),
        "confidence": classifier_result.confidence,
        "context_used": classifier_result.context_used,
        "blocked": False,
        "safety_verdict": {
            "flag": classifier_result.safety_verdict.flag.value,
            "note": classifier_result.safety_verdict.note,
        },
    }
    # Include fallback fields when the classifier degraded gracefully
    if classifier_result.fallback:
        metadata_payload["fallback"] = True
        metadata_payload["fallback_reason"] = classifier_result.fallback_reason
        logger.warning(
            "Classifier fallback | reason=%s | session=%s | query=%r",
            classifier_result.fallback_reason,
            session_id,
            query,
        )
    yield _sse_metadata(metadata_payload)

    # ------------------------------------------------------------------
    # Step 4b — If classification fell back, stream the spec-mandated
    # user message and stop — do NOT dispatch to an agent.
    # The safety guard is the ONLY thing that can stop a query otherwise.
    # ------------------------------------------------------------------
    if classifier_result.fallback:
        yield _sse_token(
            "I'm having trouble processing your request right now. "
            "Please try again in a moment."
        )
        yield _sse_done()
        return

    # ------------------------------------------------------------------
    # Step 5 — Append turn to session memory AFTER classification
    #   (so the classifier sees history up to but not including this turn)
    # ------------------------------------------------------------------
    memory.append_turn(session_id, query)

    # ------------------------------------------------------------------
    # Step 6 — Agent dispatch with timeout
    # ------------------------------------------------------------------
    try:
        async with asyncio.timeout(config.PIPELINE_TIMEOUT_S):
            async for token in _dispatch(user, classifier_result):
                yield _sse_token(token)
    except asyncio.TimeoutError:
        yield _sse_error(
            "Agent response timed out. Please try again.",
            code="AGENT_TIMEOUT",
        )
    except Exception as exc:
        logger.exception("Agent error for agent=%s: %s", classifier_result.agent, exc)
        yield _sse_error(
            f"The {classifier_result.agent.value} agent encountered an error. "
            "Please try again or contact support."
        )

    yield _sse_done()
