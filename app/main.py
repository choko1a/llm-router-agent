import logging
import time

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.llm_service import execute_chat_route
from app.logging_config import configure_logging
from app.models import ChatRequest, ChatResponse, HealthResponse, RouteRequest, RouteResponse
from app.providers.base import ProviderError
from app.routing import choose_route

configure_logging()
logger = logging.getLogger("llm_router_agent")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Minimal cost-aware LLM routing MVP with mock and real provider responses.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service=settings.app_name, version=settings.app_version)


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    settings.refresh()
    start = time.perf_counter()
    decision = choose_route(request.query)
    latency_ms = _elapsed_ms(start)
    _log_route(decision.route_type.value, latency_ms, decision.reason, decision.confidence)
    return RouteResponse(
        route_type=decision.route_type,
        route=decision.route_type,
        confidence=decision.confidence,
        estimated_cost_usd=decision.estimated_cost_usd,
        estimated_latency_ms=decision.estimated_latency_ms,
        reason=decision.reason,
        fallback_policy=decision.fallback_policy,
        fallback_route=decision.fallback_route,
        signals=decision.signals,
        latency_ms=latency_ms,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings.refresh()
    start = time.perf_counter()
    user_message = _last_user_message(request)
    decision = choose_route(user_message)
    try:
        execution = execute_chat_route(decision.route_type, user_message, request.messages)
    except ProviderError as exc:
        logger.exception("llm_generation_failed route=%s", decision.route_type.value)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    latency_ms = _elapsed_ms(start)
    _log_route(decision.route_type.value, latency_ms, decision.reason, decision.confidence)
    return ChatResponse(
        response=execution.response.content,
        route_type=decision.route_type,
        route=decision.route_type,
        confidence=decision.confidence,
        estimated_cost_usd=decision.estimated_cost_usd,
        estimated_latency_ms=decision.estimated_latency_ms,
        reason=decision.reason,
        fallback_policy=decision.fallback_policy,
        fallback_route=decision.fallback_route,
        signals=decision.signals,
        latency_ms=latency_ms,
        mock=execution.response.mock,
        provider=execution.response.provider,
        model=execution.response.model,
        selected_tool=execution.selected_tools[-1] if execution.selected_tools else None,
        selected_tools=execution.selected_tools,
        executed_route=execution.executed_route,
        fallback_history=execution.fallback_history,
        router_mode=settings.router_mode,
    )


def _last_user_message(request: ChatRequest) -> str:
    for message in reversed(request.messages):
        if message.role.lower() == "user":
            return message.content
    raise HTTPException(status_code=400, detail="At least one user message is required.")


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _log_route(route_name: str, latency_ms: float, reason: str, confidence: float) -> None:
    logger.info(
        "route_selected route=%s latency_ms=%.3f confidence=%.2f reason=%s",
        route_name,
        latency_ms,
        confidence,
        reason,
    )
