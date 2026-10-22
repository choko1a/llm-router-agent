from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.models import Route
from app.routing_config import FALLBACK_POLICY, ROUTE_PROFILES, ROUTING_RULES


@dataclass(frozen=True)
class RouteDecision:
    route_type: Route
    reason: str
    confidence: float
    estimated_cost_usd: float
    estimated_latency_ms: int
    fallback_policy: Dict[str, Optional[str]]
    fallback_route: Optional[Route]
    signals: List[str]


def choose_route(query: str) -> RouteDecision:
    if settings.router_mode == "classifier":
        return choose_route_classifier(query)
    if settings.router_mode == "llm":
        return choose_route_llm(query)
    if settings.router_mode != "rules":
        raise ValueError("Unsupported ROUTER_MODE: %s" % settings.router_mode)
    return choose_route_rules(query)


def choose_route_rules(query: str) -> RouteDecision:
    normalized = query.lower().strip()
    words = normalized.split()
    signals: List[str] = []
    matched_rule = _best_matching_rule(normalized)

    if matched_rule is None:
        route_type = Route.cheap
        reason = "No high-risk signals found; defaulting to the low-cost route."
        confidence = 0.64
        signals.append("default:cheap")
    else:
        route_type = Route(matched_rule["route_type"])
        reason = matched_rule["reason"]
        confidence = float(matched_rule["confidence"])
        hits = _keyword_hits(normalized, matched_rule["keywords"])
        signals.extend("%s:%s" % (matched_rule["name"], keyword) for keyword in hits)

    if len(words) <= 12:
        signals.append("short_query")
        if route_type == Route.cheap:
            confidence += 0.06
    elif len(words) > 24:
        signals.append("long_query")
        if route_type == Route.cheap:
            route_type = Route.strong
            reason = "Request is long enough to need a stronger model."
            confidence = max(confidence, 0.76)
        else:
            confidence += 0.03

    if "?" in normalized and len(words) > 14:
        signals.append("long_question")
        confidence += 0.02

    profile = ROUTE_PROFILES[route_type.value]
    fallback_route = _fallback_route_for(route_type)

    return RouteDecision(
        route_type=route_type,
        reason=reason,
        confidence=round(min(confidence, 0.95), 2),
        estimated_cost_usd=float(profile["estimated_cost_usd"]),
        estimated_latency_ms=int(profile["estimated_latency_ms"]),
        fallback_policy=FALLBACK_POLICY,
        fallback_route=fallback_route,
        signals=signals,
    )


def choose_route_classifier(query: str) -> RouteDecision:
    from app.classifier_router import predict_route

    route_type, confidence = predict_route(query)
    profile = ROUTE_PROFILES[route_type.value]
    return RouteDecision(
        route_type=route_type,
        reason="Route selected by trainable TF-IDF LogisticRegression classifier.",
        confidence=confidence,
        estimated_cost_usd=float(profile["estimated_cost_usd"]),
        estimated_latency_ms=int(profile["estimated_latency_ms"]),
        fallback_policy=FALLBACK_POLICY,
        fallback_route=_fallback_route_for(route_type),
        signals=["classifier:tfidf_logistic_regression"],
    )


def choose_route_llm(query: str) -> RouteDecision:
    from app.llm_router import build_llm_route_decision, build_router_messages, parse_router_json
    from app.llm_service import generate_response
    from app.providers.base import ProviderError
    from app.providers.openrouter_provider import OpenRouterProvider
    from app.providers.retry import call_with_retries

    rules_decision = choose_route_rules(query)
    if getattr(settings, "use_mock", False):
        return RouteDecision(
            route_type=rules_decision.route_type,
            reason="LLM router is skipped in mock mode; rules router was used.",
            confidence=rules_decision.confidence,
            estimated_cost_usd=rules_decision.estimated_cost_usd,
            estimated_latency_ms=rules_decision.estimated_latency_ms,
            fallback_policy=rules_decision.fallback_policy,
            fallback_route=rules_decision.fallback_route,
            signals=["llm_router_mock_fallback"] + rules_decision.signals,
        )
    try:
        router_messages = build_router_messages(query)
        if getattr(settings, "llm_provider", "").lower() == "openrouter":
            response = call_with_retries(
                lambda: OpenRouterProvider().generate(
                    Route.cheap,
                    router_messages,
                    model_override=settings.openrouter_router_model,
                ),
                "router",
            ).value
        else:
            response = call_with_retries(
                lambda: generate_response(Route.cheap, router_messages, force_real=True),
                "router",
            ).value
        parsed = parse_router_json(response.content)
        llm_decision = build_llm_route_decision(parsed)
        if rules_decision.route_type == Route.agent and llm_decision.route_type != Route.agent:
            return RouteDecision(
                route_type=rules_decision.route_type,
                reason=(
                    "LLM router selected %s, but deterministic tool signals require agent route."
                    % llm_decision.route_type.value
                ),
                confidence=max(llm_decision.confidence, rules_decision.confidence),
                estimated_cost_usd=rules_decision.estimated_cost_usd,
                estimated_latency_ms=rules_decision.estimated_latency_ms,
                fallback_policy=rules_decision.fallback_policy,
                fallback_route=rules_decision.fallback_route,
                signals=["llm_router_tool_override"] + rules_decision.signals,
            )
        return llm_decision
    except ProviderError as exc:
        return RouteDecision(
            route_type=rules_decision.route_type,
            reason="LLM router failed, fell back to rules router: %s" % exc,
            confidence=rules_decision.confidence,
            estimated_cost_usd=rules_decision.estimated_cost_usd,
            estimated_latency_ms=rules_decision.estimated_latency_ms,
            fallback_policy=rules_decision.fallback_policy,
            fallback_route=rules_decision.fallback_route,
            signals=["llm_router_fallback"] + rules_decision.signals,
        )


def _best_matching_rule(text: str) -> Optional[Dict[str, Any]]:
    matches = []
    for rule in ROUTING_RULES:
        if _keyword_hits(text, rule["keywords"]):
            matches.append(rule)
    if not matches:
        return None
    return sorted(matches, key=lambda rule: rule["priority"], reverse=True)[0]


def _keyword_hits(text: str, keywords: List[str]) -> List[str]:
    return sorted(keyword for keyword in keywords if keyword in text)


def _fallback_route_for(route_type: Route) -> Optional[Route]:
    fallback_route = FALLBACK_POLICY[route_type.value]
    if fallback_route is None:
        return None
    return Route(fallback_route)
