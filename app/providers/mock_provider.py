from typing import List

from app.models import ChatMessage, Route
from app.providers.base import BaseProvider, ProviderResponse


class MockProvider(BaseProvider):
    provider_name = "mock"

    def generate(self, route_type: Route, messages: List[ChatMessage]) -> ProviderResponse:
        user_message = _last_user_content(messages)
        return ProviderResponse(
            content=(
                "Mock response via %s route. "
                "No external LLM API was called. Input preview: %s"
            )
            % (route_type.value, user_message[:120]),
            provider=self.provider_name,
            model="mock-%s" % route_type.value,
            mock=True,
        )


def _last_user_content(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content
    return messages[-1].content

