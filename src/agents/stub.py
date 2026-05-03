"""
Stub agent — returned for all agents not yet fully implemented.

Contract:
  - Must not crash under any input.
  - Returns a structured JSON object containing:
      status              "not_implemented"
      classified_intent   the intent string from the classifier
      target_agent        the agent string that would handle this
      extracted_entities  the entities object from the classifier
      message             human-readable "not yet available" notice
      user_facing_message per-agent friendly one-sentence UI string

Output format: raw JSON string (no markdown, no explanation).
The pipeline yields it as a single SSE token event.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from src.schemas import ClassifierResult, UserProfile


# ---------------------------------------------------------------------------
# Human-readable display names for each agent slug
# ---------------------------------------------------------------------------

_HUMAN_NAMES: dict[str, str] = {
    "portfolio_health":      "Portfolio Health",
    "market_research":       "Market Research",
    "investment_strategy":   "Investment Strategy",
    "financial_planning":    "Financial Planning",
    "financial_calculator":  "Financial Calculator",
    "risk_assessment":       "Risk Assessment",
    "product_recommendation": "Product Recommendation",
    "predictive_analysis":   "Predictive Analysis",
    "customer_support":      "Customer Support",
    "general_query":         "General Query",
}


def _user_facing_message(agent_name: str, entities: dict) -> str:
    """
    Build the per-agent friendly one-sentence UI message.
    Injects ticker or topic where the spec calls for it.
    """
    # Pull the most relevant entity label for interpolation
    ticker_or_topic: str = ""
    if entities.get("tickers"):
        ticker_or_topic = entities["tickers"][0]
    elif entities.get("topics"):
        ticker_or_topic = entities["topics"][0]
    elif entities.get("sectors"):
        ticker_or_topic = entities["sectors"][0]

    messages: dict[str, str] = {
        "market_research": (
            f"Market research is coming soon — I've noted your question"
            f"{' about ' + ticker_or_topic if ticker_or_topic else ''}."
        ),
        "investment_strategy": (
            "Investment strategy guidance is on the way — your question has been logged."
        ),
        "financial_calculator": (
            "The financial calculator is not yet live, but your numbers have been captured."
        ),
        "financial_planning": (
            "Long-term financial planning features are coming soon."
        ),
        "risk_assessment": (
            "Risk analysis tools are in development — stay tuned."
        ),
        "product_recommendation": (
            "Product recommendations are not yet available in this build."
        ),
        "predictive_analysis": (
            "Predictive analysis features are coming in a future release."
        ),
        "customer_support": (
            "For support, please contact support@nestmind.ai while this feature is being built."
        ),
        "general_query": (
            "I'm here to help — this type of query will be fully supported soon."
        ),
        # portfolio_health is implemented, but include a fallback in case it is
        # somehow routed here during testing
        "portfolio_health": (
            "Portfolio health analysis will be available shortly."
        ),
    }

    return messages.get(
        agent_name,
        "This feature is not yet available in the current build.",
    )


async def run(
    user: UserProfile,
    classifier_result: ClassifierResult,
    agent_name: str,
) -> AsyncIterator[str]:
    """
    Yield a single structured stub response as a JSON string.

    The output strictly follows the spec contract:
      {
        "status":              "not_implemented",
        "classified_intent":   <str>,
        "target_agent":        <str>,
        "extracted_entities":  <object>,
        "message":             <str>,
        "user_facing_message": <str>
      }

    No markdown, no explanation — raw JSON only.
    """
    entities_dict = classifier_result.entities.to_dict()
    human_name = _HUMAN_NAMES.get(agent_name, agent_name.replace("_", " ").title())

    payload = {
        "status": "not_implemented",
        "classified_intent": classifier_result.intent,
        "target_agent": agent_name,
        "extracted_entities": entities_dict,
        "message": (
            f"The {human_name} agent is not yet available in this build. "
            "Your query has been correctly classified and will be handled "
            "by this agent in a future release."
        ),
        "user_facing_message": _user_facing_message(agent_name, entities_dict),
    }

    yield json.dumps(payload, indent=2)
