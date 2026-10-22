from types import SimpleNamespace

from app import llm_service
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider


def test_openai_is_default_provider() -> None:
    assert llm_service.settings.llm_provider == "openai"
    assert llm_service.settings.use_mock is True


def test_real_provider_selects_openai(monkeypatch) -> None:
    monkeypatch.setattr(llm_service, "settings", SimpleNamespace(llm_provider="openai"))

    assert isinstance(llm_service._real_provider(), OpenAIProvider)


def test_real_provider_selects_openrouter(monkeypatch) -> None:
    monkeypatch.setattr(llm_service, "settings", SimpleNamespace(llm_provider="openrouter"))

    assert isinstance(llm_service._real_provider(), OpenRouterProvider)
