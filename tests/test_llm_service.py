from collections import Counter
from types import SimpleNamespace

import pytest

from app import llm_service
from app.models import ChatMessage, Route
from app.providers import retry
from app.providers.base import ProviderConfigurationError, ProviderError, ProviderResponse
from app.providers.retry import ModelCallFailed
from app.tools.base import ToolResult


MESSAGES = [ChatMessage(role="user", content="Hello")]


class FakeProvider:
    def __init__(self, failures):
        self.failures = failures
        self.calls = Counter()

    def generate(self, route_type, messages):
        self.calls[route_type] += 1
        if self.calls[route_type] <= self.failures.get(route_type, 0):
            raise ProviderError("temporary failure")
        return ProviderResponse(
            content=route_type.value + " response", provider="fake", model=route_type.value,
        )


def _configure(monkeypatch, provider, agent_service=None):
    monkeypatch.setattr(llm_service, "settings", SimpleNamespace(use_mock=False, agent_enabled=True))
    monkeypatch.setattr(llm_service, "_real_provider", lambda: provider)
    monkeypatch.setattr(retry, "settings", SimpleNamespace(model_max_retries=2))
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    if agent_service is not None:
        monkeypatch.setattr(llm_service, "AgentService", agent_service)


def test_cheap_succeeds_on_third_attempt_without_upgrade(monkeypatch):
    provider = FakeProvider({Route.cheap: 2})
    _configure(monkeypatch, provider)

    run = llm_service.execute_chat_route(Route.cheap, "Hello", MESSAGES)

    assert run.executed_route == Route.cheap
    assert run.fallback_history == []
    assert provider.calls == Counter({Route.cheap: 3})


def test_cheap_upgrades_to_strong_after_retries(monkeypatch):
    provider = FakeProvider({Route.cheap: 3})
    _configure(monkeypatch, provider)

    run = llm_service.execute_chat_route(Route.cheap, "Hello", MESSAGES)

    assert run.response.content == "strong response"
    assert run.executed_route == Route.strong
    assert run.fallback_history == ["cheap->strong"]
    assert provider.calls == Counter({Route.cheap: 3, Route.strong: 1})


def test_strong_upgrades_to_agent_after_retries(monkeypatch):
    provider = FakeProvider({Route.strong: 3})

    class FakeAgentService:
        def run(self, query, messages):
            assert query == "Hello"
            assert messages == MESSAGES
            return SimpleNamespace(
                response=ProviderResponse(content="agent answer", provider="fake", model="agent"),
                tool_results=[ToolResult(tool_name="web_search", success=True, content="found")],
            )

    _configure(monkeypatch, provider, FakeAgentService)

    run = llm_service.execute_chat_route(Route.strong, "Hello", MESSAGES)

    assert run.executed_route == Route.agent
    assert run.response.content == "agent answer"
    assert run.selected_tools == ["web_search"]
    assert run.fallback_history == ["strong->agent"]
    assert provider.calls == Counter({Route.strong: 3})


def test_cheap_then_strong_then_agent(monkeypatch):
    provider = FakeProvider({Route.cheap: 3, Route.strong: 3})

    class FakeAgentService:
        def run(self, query, messages):
            return SimpleNamespace(
                response=ProviderResponse(content="agent answer", provider="fake", model="agent"),
                tool_results=[],
            )

    _configure(monkeypatch, provider, FakeAgentService)

    run = llm_service.execute_chat_route(Route.cheap, "Hello", MESSAGES)

    assert run.executed_route == Route.agent
    assert run.fallback_history == ["cheap->strong", "strong->agent"]
    assert provider.calls == Counter({Route.cheap: 3, Route.strong: 3})


def test_agent_exhaustion_has_no_more_fallback(monkeypatch):
    provider = FakeProvider({})

    class FailingAgentService:
        def run(self, query, messages):
            raise ModelCallFailed("agent", 3)

    _configure(monkeypatch, provider, FailingAgentService)

    with pytest.raises(ProviderError, match="All available model routes failed"):
        llm_service.execute_chat_route(Route.agent, "Hello", MESSAGES)


def test_configuration_error_fails_without_retry_or_upgrade(monkeypatch):
    class ConfigErrorProvider(FakeProvider):
        def generate(self, route_type, messages):
            self.calls[route_type] += 1
            raise ProviderConfigurationError("API key is missing")

    provider = ConfigErrorProvider({})
    _configure(monkeypatch, provider)

    with pytest.raises(ProviderConfigurationError, match="API key is missing"):
        llm_service.execute_chat_route(Route.cheap, "Hello", MESSAGES)
    assert provider.calls == Counter({Route.cheap: 1})
