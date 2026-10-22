from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Route(str, Enum):
    cheap = "cheap"
    strong = "strong"
    agent = "agent"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User request to classify.")
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class RouteResponse(BaseModel):
    route_type: Route
    route: Route
    confidence: float = Field(..., ge=0.0, le=1.0)
    estimated_cost_usd: float = Field(..., ge=0.0)
    estimated_latency_ms: int = Field(..., ge=0)
    reason: str
    fallback_policy: Dict[str, Optional[str]]
    fallback_route: Optional[Route] = None
    signals: List[str] = Field(default_factory=list)
    latency_ms: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role, for example system/user/assistant.")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    route_type: Route
    route: Route
    confidence: float = Field(..., ge=0.0, le=1.0)
    estimated_cost_usd: float = Field(..., ge=0.0)
    estimated_latency_ms: int = Field(..., ge=0)
    reason: str
    fallback_policy: Dict[str, Optional[str]]
    fallback_route: Optional[Route] = None
    signals: List[str] = Field(default_factory=list)
    latency_ms: float
    mock: bool = True
    provider: str = "mock"
    model: str = "mock"
    selected_tool: Optional[str] = None
    selected_tools: List[str] = Field(default_factory=list)
    executed_route: Optional[Route] = None
    fallback_history: List[str] = Field(default_factory=list)
    router_mode: str = "rules"
