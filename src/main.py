"""
FastAPI application — HTTP layer.

Endpoints
---------
POST /v1/chat   — full pipeline, SSE streaming only
GET  /health    — liveness probe
GET  /          — API info

Design decisions
----------------
- SSE is the ONLY response mode. No JSON fallback.
- Errors are structured SSE error events, not HTTP error codes.
  This simplifies client-side handling — the client only needs to parse
  SSE frames, not switch between JSON and SSE based on status codes.
- The pipeline timeout is enforced inside pipeline.py, not at the HTTP layer.
  The HTTP layer just proxies the async generator.
- User profile is passed in the request body (no auth in this demo build).
  A production system would decode a JWT and look up the profile.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from src.pipeline import run_pipeline, _sse_error, _sse_done
from src.schemas import ChatRequest, SSEEventType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NestMind API starting up")
    yield
    logger.info("NestMind API shutting down")


app = FastAPI(
    title="NestMind AI",
    description=(
        "NestMind is an AI co-investor platform. "
        "Every response streams via Server-Sent Events."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "NestMind AI",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main chat endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat", tags=["chat"])
async def chat(request: Request):
    """
    Accept a user query and stream the response via SSE.

    Body: ChatRequest (JSON)

    SSE event types:
      - metadata  : routing decision (agent, intent, entities)
      - token     : streaming response text chunk
      - done      : end of stream
      - error     : structured error
    """
    # Parse and validate request body
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        # Capture exc in a local variable to avoid Python closure gotcha
        # with async generators (exc is unbound after the except block exits).
        _error_msg = f"Invalid request: {exc}"

        async def _error_stream():
            yield _sse_error(_error_msg, code="INVALID_REQUEST")
            yield _sse_done()

        return StreamingResponse(
            _error_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    logger.info(
        "Chat request: session=%s user=%s query=%r",
        chat_request.session_id,
        chat_request.user.user_id,
        chat_request.query[:80],
    )

    return StreamingResponse(
        run_pipeline(chat_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
