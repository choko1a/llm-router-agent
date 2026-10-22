from dataclasses import dataclass
from typing import List

from app.models import ChatMessage, Route


class ProviderError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""


class ProviderConfigurationError(ProviderError):
    """Raised when retrying with the current configuration cannot succeed."""


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    provider: str
    model: str
    mock: bool = False


class BaseProvider:
    provider_name = "base"

    def generate(self, route_type: Route, messages: List[ChatMessage]) -> ProviderResponse:
        raise NotImplementedError

