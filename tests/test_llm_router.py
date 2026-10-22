from types import SimpleNamespace

from app import llm_service, routing
from app.models import Route
from app.providers.base import ProviderError, ProviderResponse
from app.providers import retry


def test_llm_router_uses_rules_without_real_call_in_mock_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        routing,
        "settings",
        SimpleNamespace(router_mode="llm", use_mock=True),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("mock mode must not call a real routing model")

    monkeypatch.setattr(llm_service, "generate_response", fail_if_called)

    decision = routing.choose_route("Ping")

    assert decision.route_type == Route.cheap
    assert decision.signals[0] == "llm_router_mock_fallback"


def test_llm_router_falls_back_to_rules_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(routing, "settings", SimpleNamespace(router_mode="llm"))

    def fake_generate_response(route_type, messages, force_real=False):
        assert route_type == Route.cheap
        assert force_real is True
        return ProviderResponse(
            content="not json",
            provider="fake",
            model="fake-cheap",
        )

    monkeypatch.setattr(llm_service, "generate_response", fake_generate_response)

    decision = routing.choose_route("Что такое LLM?")

    assert decision.route_type == Route.cheap
    assert decision.signals[0] == "llm_router_fallback"
    assert "fell back to rules router" in decision.reason


def test_llm_router_uses_valid_json(monkeypatch) -> None:
    monkeypatch.setattr(routing, "settings", SimpleNamespace(router_mode="llm"))

    def fake_generate_response(route_type, messages, force_real=False):
        return ProviderResponse(
            content=(
                '{"route_type":"agent","confidence":0.87,'
                '"reason":"Needs repository analysis","signals":["repo"]}'
            ),
            provider="fake",
            model="fake-cheap",
        )

    monkeypatch.setattr(llm_service, "generate_response", fake_generate_response)

    decision = routing.choose_route("Проанализируй README проекта")

    assert decision.route_type == Route.agent
    assert decision.confidence == 0.87
    assert decision.signals == ["llm_router", "repo"]


def test_llm_router_does_not_downgrade_tool_requests(monkeypatch) -> None:
    monkeypatch.setattr(routing, "settings", SimpleNamespace(router_mode="llm"))

    def fake_generate_response(route_type, messages, force_real=False):
        return ProviderResponse(
            content=(
                '{"route_type":"strong","confidence":0.7,'
                '"reason":"Looks analytical","signals":["analysis"]}'
            ),
            provider="fake",
            model="fake-cheap",
        )

    monkeypatch.setattr(llm_service, "generate_response", fake_generate_response)

    decision = routing.choose_route("Найди актуальные цены OpenAI API")

    assert decision.route_type == Route.agent
    assert decision.signals[0] == "llm_router_tool_override"


def test_llm_router_retries_failed_model_request(monkeypatch) -> None:
    monkeypatch.setattr(routing, "settings", SimpleNamespace(router_mode="llm", use_mock=False))
    monkeypatch.setattr(retry, "settings", SimpleNamespace(model_max_retries=2))
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    attempts = []

    def fake_generate_response(route_type, messages, force_real=False):
        attempts.append(route_type)
        if len(attempts) < 3:
            raise ProviderError("temporary router failure")
        return ProviderResponse(
            content='{"route_type":"strong","confidence":0.8}',
            provider="fake",
            model="router",
        )

    monkeypatch.setattr(llm_service, "generate_response", fake_generate_response)

    decision = routing.choose_route("Compare two approaches")

    assert decision.route_type == Route.strong
    assert attempts == [Route.cheap, Route.cheap, Route.cheap]
