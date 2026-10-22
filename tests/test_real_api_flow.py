from copy import deepcopy
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app import llm_service
from app.main import app
from app.providers import retry
from app.providers.base import ProviderError, ProviderResponse
from app.agents.tool_registry import ToolRegistry
from app.tools.base import ToolResult


client = TestClient(app)


@pytest.fixture
def fake_openai_sdk(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="real provider answer", tool_calls=None))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append({"client_options": kwargs})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    return calls


def _configure_real_mode(monkeypatch, provider):
    monkeypatch.setattr(settings, "refresh", lambda: None)
    monkeypatch.setattr(settings, "router_mode", "rules")
    monkeypatch.setattr(settings, "use_mock", False)
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "agent_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-openai-key")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-openrouter-key")
    monkeypatch.setattr(settings, "openai_cheap_model", "openai-cheap")
    monkeypatch.setattr(settings, "openai_strong_model", "openai-strong")
    monkeypatch.setattr(settings, "openrouter_cheap_model", "router-cheap")
    monkeypatch.setattr(settings, "openrouter_strong_model", "router-strong")
    monkeypatch.setattr(settings, "openrouter_agent_model", "router-agent")
    monkeypatch.setattr(settings, "openrouter_code_model", "router-code")
    monkeypatch.setattr(settings, "agent_max_tool_calls", 3)


def test_real_openai_chat_uses_selected_model(monkeypatch, fake_openai_sdk) -> None:
    _configure_real_mode(monkeypatch, "openai")

    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Ping"}]})

    assert response.status_code == 200
    assert response.json()["provider"] == "openai"
    assert response.json()["model"] == "openai-cheap"
    assert response.json()["mock"] is False
    assert response.json()["response"] == "real provider answer"
    assert fake_openai_sdk[1]["model"] == "openai-cheap"


def test_real_chat_passes_previous_turns_to_model(monkeypatch, fake_openai_sdk) -> None:
    _configure_real_mode(monkeypatch, "openai")

    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "My name is Lin."},
                {"role": "assistant", "content": "Hello, Lin."},
                {"role": "user", "content": "What is my name?"},
            ]
        },
    )

    assert response.status_code == 200
    assert fake_openai_sdk[1]["messages"] == [
        {"role": "user", "content": "My name is Lin."},
        {"role": "assistant", "content": "Hello, Lin."},
        {"role": "user", "content": "What is my name?"},
    ]


def test_real_openrouter_chat_uses_selected_model(monkeypatch, fake_openai_sdk) -> None:
    _configure_real_mode(monkeypatch, "openrouter")

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Solve this math problem"}]},
    )

    assert response.status_code == 200
    assert response.json()["route_type"] == "strong"
    assert response.json()["provider"] == "openrouter"
    assert response.json()["model"] == "router-strong"
    assert response.json()["mock"] is False
    assert fake_openai_sdk[0]["client_options"]["base_url"] == settings.openrouter_base_url
    assert fake_openai_sdk[1]["model"] == "router-strong"


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [("openai", "openai-strong"), ("openrouter", "router-agent")],
)
def test_real_agent_uses_selected_provider(
    monkeypatch, fake_openai_sdk, provider, expected_model
) -> None:
    _configure_real_mode(monkeypatch, provider)

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Generate a FastAPI endpoint"}]},
    )

    assert response.status_code == 200
    assert response.json()["route_type"] == "agent"
    assert response.json()["selected_tool"] is None
    assert response.json()["selected_tools"] == []
    assert response.json()["provider"] == provider
    assert response.json()["model"] == expected_model
    assert response.json()["mock"] is False
    assert fake_openai_sdk[1]["model"] == expected_model
    assert fake_openai_sdk[1]["tool_choice"] == "auto"
    assert fake_openai_sdk[1]["parallel_tool_calls"] is False
    assert fake_openai_sdk[1]["messages"][-1] == {
        "role": "user", "content": "Generate a FastAPI endpoint",
    }


def test_real_mode_without_api_key_returns_error(monkeypatch) -> None:
    _configure_real_mode(monkeypatch, "openai")
    monkeypatch.setattr(settings, "openai_api_key", "")

    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Ping"}]})

    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_openrouter_agent_serial_tool_call_protocol(monkeypatch) -> None:
    _configure_real_mode(monkeypatch, "openrouter")
    calls = []
    queued = [
        SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="one", function=SimpleNamespace(name="web_search", arguments='{"query":"latest info"}'),
        )]),
        SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="two", function=SimpleNamespace(name="repo_analysis", arguments='{"query":"this repo"}'),
        )]),
        SimpleNamespace(content="Final combined answer", tool_calls=None),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=queued.pop(0))])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(ToolRegistry, "execute", lambda self, call: ToolResult(
        tool_name=call.name, success=True, content="result " + call.name,
    ))

    response = client.post("/chat", json={"messages": [
        {"role": "user", "content": "Earlier context"},
        {"role": "assistant", "content": "I will use it"},
        {"role": "user", "content": "Search the web, then analyze this repo"},
    ]})

    assert response.status_code == 200
    assert response.json()["response"] == "Final combined answer"
    assert response.json()["selected_tools"] == ["web_search", "repo_analysis"]
    assert response.json()["selected_tool"] == "repo_analysis"
    assert len(calls) == 3
    assert all(call["parallel_tool_calls"] is False for call in calls)
    assert calls[0]["messages"][-1] == {
        "role": "user", "content": "Search the web, then analyze this repo",
    }
    assert calls[1]["messages"][-2]["tool_calls"][0]["id"] == "one"
    assert calls[1]["messages"][-1]["tool_call_id"] == "one"
    assert calls[2]["messages"][-2]["tool_calls"][0]["id"] == "two"
    assert calls[2]["messages"][-1]["tool_call_id"] == "two"


def test_chat_reports_initial_and_executed_route_after_upgrade(monkeypatch) -> None:
    _configure_real_mode(monkeypatch, "openrouter")
    attempts = []

    class FakeProvider:
        def generate(self, route_type, messages):
            attempts.append(route_type.value)
            if route_type.value == "cheap":
                raise ProviderError("temporary outage")
            return ProviderResponse(content="strong answer", provider="openrouter", model="strong-model")

    monkeypatch.setattr(llm_service, "_real_provider", lambda: FakeProvider())
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)

    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Ping"}]})

    assert response.status_code == 200
    assert response.json()["route_type"] == "cheap"
    assert response.json()["executed_route"] == "strong"
    assert response.json()["fallback_history"] == ["cheap->strong"]
    assert response.json()["model"] == "strong-model"
    assert attempts == ["cheap", "cheap", "cheap", "strong"]
