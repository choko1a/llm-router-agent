import json
from typing import Any, Dict, List

from app.models import ChatMessage, Route
from app.providers.base import ProviderError
from app.routing_config import FALLBACK_POLICY, ROUTE_PROFILES


ROUTER_SYSTEM_PROMPT = """You are an LLM router. Classify the user request.
Return only valid JSON with this schema:
{
  "route_type": "cheap|strong|agent",
  "confidence": 0.0,
  "reason": "short explanation",
  "signals": ["signal"]
}

Routes:
- cheap: FAQ, short questions, definitions, translation, formatting, brief answers.
- strong: reasoning, architecture, analysis, comparison, optimization, complex explanation.
- agent: search/web, code/debug, repository/file analysis, multi-step or tool use.
"""


def build_router_messages(query: str) -> List[ChatMessage]:
    return [
        ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=query),
    ]


def parse_router_json(content: str) -> Dict[str, Any]:
    try:
        payload = json.loads(_extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise ProviderError("LLM router returned invalid JSON.") from exc

    route_type = payload.get("route_type")
    if route_type not in {route.value for route in Route}:
        raise ProviderError("LLM router returned unsupported route_type: %s" % route_type)

    confidence = payload.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        raise ProviderError("LLM router returned non-numeric confidence.")

    reason = payload.get("reason", "")
    signals = payload.get("signals", [])
    if not isinstance(reason, str):
        reason = str(reason)
    if not isinstance(signals, list):
        signals = [str(signals)]

    normalized_confidence = max(0.0, min(float(confidence), 1.0))
    return {
        "route_type": Route(route_type),
        "confidence": round(normalized_confidence, 2),
        "reason": reason or "Route selected by LLM router.",
        "signals": [str(signal) for signal in signals],
    }


def build_llm_route_decision(parsed: Dict[str, Any]):
    from app.routing import RouteDecision, _fallback_route_for

    route_type = parsed["route_type"]
    profile = ROUTE_PROFILES[route_type.value]
    return RouteDecision(
        route_type=route_type,
        reason=parsed["reason"],
        confidence=parsed["confidence"],
        estimated_cost_usd=float(profile["estimated_cost_usd"]),
        estimated_latency_ms=int(profile["estimated_latency_ms"]),
        fallback_policy=FALLBACK_POLICY,
        fallback_route=_fallback_route_for(route_type),
        signals=["llm_router"] + parsed["signals"],
    )


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    return stripped[start : end + 1]
