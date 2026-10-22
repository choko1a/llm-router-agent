from typing import Any, Dict, List, Optional

from app.config import settings
from app.models import ChatMessage, Route
from app.providers.base import BaseProvider, ProviderConfigurationError, ProviderError, ProviderResponse
from app.providers.agent_types import AgentTurn
from app.providers.tool_calling import complete_agent_turn


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self) -> None:
        self.models: Dict[Route, str] = {
            Route.cheap: settings.openai_cheap_model,
            Route.strong: settings.openai_strong_model,
            Route.agent: settings.openai_strong_model,
        }

    def generate(
        self,
        route_type: Route,
        messages: List[ChatMessage],
        model_override: Optional[str] = None,
    ) -> ProviderResponse:
        api_key = settings.openai_api_key
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required when USE_MOCK=false.")

        model = model_override or self.models.get(route_type)
        if model is None:
            raise ProviderConfigurationError("No OpenAI model configured for route %s." % route_type.value)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, max_retries=0, timeout=30)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": item.role, "content": item.content} for item in messages],
                temperature=0.2,
            )
            content = completion.choices[0].message.content or ""
        except Exception as exc:
            raise ProviderError("OpenAI request failed: %s" % exc) from exc

        return ProviderResponse(
            content=content,
            provider=self.provider_name,
            model=model,
            mock=False,
        )

    def complete_agent_turn(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> AgentTurn:
        if not settings.openai_api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required when USE_MOCK=false.")
        if not settings.openai_strong_model:
            raise ProviderConfigurationError("OPENAI_STRONG_MODEL is required for agent mode.")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key, max_retries=0, timeout=30)
        except Exception as exc:
            raise ProviderError("OpenAI client initialization failed: %s" % exc) from exc
        return complete_agent_turn(
            client=client,
            model=settings.openai_strong_model,
            provider=self.provider_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

