import logging
from dataclasses import dataclass
from typing import List

from app.agents.agent_service import AgentService
from app.config import settings
from app.models import ChatMessage, Route
from app.providers.base import ProviderConfigurationError, ProviderError, ProviderResponse
from app.providers.mock_provider import MockProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider
from app.providers.retry import ModelCallFailed, call_with_retries
from app.routing_config import FALLBACK_POLICY

logger = logging.getLogger("llm_router_agent.llm_service")


@dataclass(frozen=True)
class ChatExecution:
    response: ProviderResponse
    executed_route: Route
    selected_tools: List[str]
    fallback_history: List[str]


def execute_chat_route(
    initial_route: Route, query: str, messages: List[ChatMessage]
) -> ChatExecution:
    route_type = initial_route
    fallback_history: List[str] = []
    while True:
        try:
            if route_type == Route.agent and settings.agent_enabled:
                agent_run = AgentService().run(query, messages)
                return ChatExecution(
                    response=agent_run.response,
                    executed_route=route_type,
                    selected_tools=[result.tool_name for result in agent_run.tool_results],
                    fallback_history=fallback_history,
                )
            if settings.use_mock:
                response = generate_response(route_type, messages)
            else:
                response = call_with_retries(
                    lambda: generate_response(route_type, messages), route_type.value
                ).value
            return ChatExecution(
                response=response,
                executed_route=route_type,
                selected_tools=[],
                fallback_history=fallback_history,
            )
        except ModelCallFailed as exc:
            next_route_name = FALLBACK_POLICY[route_type.value]
            if next_route_name is None:
                raise ProviderError("All available model routes failed after retries.") from exc
            logger.warning(
                "route_upgraded from=%s to=%s attempts=%s",
                route_type.value, next_route_name, exc.attempts,
            )
            fallback_history.append("%s->%s" % (route_type.value, next_route_name))
            route_type = Route(next_route_name)


def generate_response(
    route_type: Route,
    messages: List[ChatMessage],
    force_real: bool = False,
) -> ProviderResponse:
    if settings.use_mock and not force_real:
        return MockProvider().generate(route_type, messages)

    provider = _real_provider()

    return provider.generate(route_type, messages)


def _real_provider():
    provider_name = settings.llm_provider.lower()
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "openrouter":
        return OpenRouterProvider()
    raise ProviderConfigurationError("Unsupported LLM_PROVIDER: %s" % settings.llm_provider)
