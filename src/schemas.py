"""
Pydantic schemas shared across the pipeline.

Design note: keeping schemas in one file avoids circular imports between
the classifier, agents, and HTTP layer.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent taxonomy — the only valid routing targets
# ---------------------------------------------------------------------------

class AgentName(str, Enum):
    portfolio_health = "portfolio_health"
    market_research = "market_research"
    investment_strategy = "investment_strategy"
    financial_planning = "financial_planning"
    financial_calculator = "financial_calculator"
    risk_assessment = "risk_assessment"
    product_recommendation = "product_recommendation"
    predictive_analysis = "predictive_analysis"
    customer_support = "customer_support"
    general_query = "general_query"


# ---------------------------------------------------------------------------
# Safety flag categories
# ---------------------------------------------------------------------------

class SafetyFlag(str, Enum):
    clean = "clean"
    insider_trading = "insider_trading"
    market_manipulation = "market_manipulation"
    money_laundering = "money_laundering"
    guaranteed_returns = "guaranteed_returns"
    reckless_advice = "reckless_advice"
    sanctions_evasion = "sanctions_evasion"
    fraud = "fraud"


# ---------------------------------------------------------------------------
# Classifier output schema
# ---------------------------------------------------------------------------

class SafetyVerdict(BaseModel):
    flag: SafetyFlag = SafetyFlag.clean
    note: str = ""


class ClassifierEntities(BaseModel):
    """All fields are optional — only include what's extractable."""
    tickers: Optional[list[str]] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    rate: Optional[float] = None
    period_years: Optional[int] = None
    frequency: Optional[str] = None     # daily|weekly|monthly|yearly
    horizon: Optional[str] = None       # 6_months|1_year|5_years
    time_period: Optional[str] = None   # today|this_week|this_month|this_year
    topics: Optional[list[str]] = None
    sectors: Optional[list[str]] = None
    index: Optional[str] = None         # S&P 500|FTSE 100|NIKKEI 225|MSCI World
    action: Optional[str] = None        # buy|sell|hold|hedge|rebalance
    goal: Optional[str] = None          # retirement|education|house|FIRE|emergency_fund

    def to_dict(self) -> dict[str, Any]:
        """Return only non-None fields."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ClassifierResult(BaseModel):
    intent: str
    agent: AgentName
    entities: ClassifierEntities = Field(default_factory=ClassifierEntities)
    safety_verdict: SafetyVerdict = Field(default_factory=SafetyVerdict)
    confidence: float = Field(ge=0.0, le=1.0)
    context_used: bool = False
    # Set by the classifier when it falls back due to LLM failure
    fallback: bool = False
    fallback_reason: str = ""  # timeout | api_error | parse_error | rate_limit | ""


# ---------------------------------------------------------------------------
# Safety guard output
# ---------------------------------------------------------------------------

class SafetyGuardResult(BaseModel):
    blocked: bool
    category: SafetyFlag = SafetyFlag.clean
    message: str = ""  # The refusal text sent to the user when blocked=True


# ---------------------------------------------------------------------------
# User / portfolio models (derived from fixtures schema)
# ---------------------------------------------------------------------------

class KYC(BaseModel):
    status: str  # "verified" | "pending" | "failed"


class Position(BaseModel):
    ticker: str
    exchange: str
    quantity: float
    avg_cost: float
    currency: str
    purchased_at: str


class UserPreferences(BaseModel):
    preferred_benchmark: Optional[str] = None
    reporting_currency: Optional[str] = None
    income_focus: Optional[bool] = None


class UserProfile(BaseModel):
    user_id: str
    name: str
    age: int
    country: str
    base_currency: str
    kyc: KYC
    risk_profile: str   # aggressive|moderate|conservative
    positions: list[Position] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)


# ---------------------------------------------------------------------------
# HTTP request/response shapes
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., description="Client-owned session ID")
    user: UserProfile


# ---------------------------------------------------------------------------
# SSE event types
# ---------------------------------------------------------------------------

class SSEEventType(str, Enum):
    metadata = "metadata"   # classifier result + routing decision
    token = "token"         # streaming LLM token
    done = "done"           # final event, marks end of stream
    error = "error"         # structured error — never a stack trace
