"""
Stub agent — returned for all agents not yet fully implemented.

Contract (from assignment):
  - Must not crash
  - Returns: classified intent, extracted entities, which agent would handle,
    and a short message indicating this is not implemented in this build.
"""
from __future__ import annotations

from typing import AsyncIterator
import json

from src.schemas import ClassifierResult, UserProfile


async def run(
    user: UserProfile,
    classifier_result: ClassifierResult,
    agent_name: str,
) -> AsyncIterator[str]:
    """
    Yield a single structured stub response.

    The output is JSON-formatted so consumers can parse it programmatically.
    """
    payload = {
        "status": "not_implemented",
        "agent": agent_name,
        "intent": classifier_result.intent,
        "entities": classifier_result.entities.to_dict(),
        "message": (
            f"The '{agent_name}' agent is not yet implemented in this build. "
            "Your query has been correctly classified and routed — "
            "this agent will be available in a future release."
        ),
    }
    yield json.dumps(payload, indent=2)
