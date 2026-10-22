import json

from app import config as app_config
from app.config import Settings
from app.runtime_config import (
    ALLOWED_RUNTIME_KEYS,
    DEFAULT_RUNTIME_CONFIG,
    load_runtime_config,
    merged_runtime_config,
)


def test_runtime_config_loads_allowed_values(tmp_path) -> None:
    path = tmp_path / "runtime_config.json"
    path.write_text(
        json.dumps(
            {
                "ROUTER_MODE": "llm",
                "USE_MOCK": False,
                "LLM_PROVIDER": "openrouter",
                "OPENAI_CHEAP_MODEL": "openai-cheap",
                "OPENAI_STRONG_MODEL": "openai-strong",
                "OPENROUTER_ROUTER_MODEL": "openai/gpt-4o-mini",
                "OPENROUTER_CHEAP_MODEL": "openai/gpt-4o-mini",
                "OPENROUTER_STRONG_MODEL": "openai/gpt-4o",
                "OPENROUTER_AGENT_MODEL": "openai/gpt-4o-mini",
                "OPENROUTER_CODE_MODEL": "deepseek/deepseek-chat-v3.1",
            }
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config["ROUTER_MODE"] == "llm"
    assert config["USE_MOCK"] is False
    assert config["LLM_PROVIDER"] == "openrouter"
    assert config["OPENAI_CHEAP_MODEL"] == "openai-cheap"
    assert config["OPENROUTER_CODE_MODEL"] == "deepseek/deepseek-chat-v3.1"


def test_runtime_config_does_not_require_or_keep_api_keys(tmp_path) -> None:
    path = tmp_path / "runtime_config.json"
    path.write_text(
        json.dumps(
            {
                "OPENROUTER_API_KEY": "secret",
                "OPENAI_API_KEY": "secret",
                "TAVILY_API_KEY": "secret",
                "OPENROUTER_AGENT_MODEL": "openai/gpt-4o-mini",
            }
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert "OPENROUTER_API_KEY" not in config
    assert "OPENAI_API_KEY" not in config
    assert "TAVILY_API_KEY" not in config
    assert config["OPENROUTER_AGENT_MODEL"] == "openai/gpt-4o-mini"


def test_runtime_config_defaults_work_without_file(tmp_path, monkeypatch) -> None:
    missing_path = tmp_path / "missing_runtime_config.json"
    for key in ALLOWED_RUNTIME_KEYS:
        monkeypatch.delenv(key, raising=False)

    assert load_runtime_config(missing_path) == {}
    assert merged_runtime_config(missing_path) == DEFAULT_RUNTIME_CONFIG


def test_settings_runtime_config_overrides_model_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        app_config,
        "load_runtime_config",
        lambda: {
            "ROUTER_MODE": "llm",
            "USE_MOCK": False,
            "LLM_PROVIDER": "openrouter",
            "OPENAI_CHEAP_MODEL": "openai-cheap",
            "OPENAI_STRONG_MODEL": "openai-strong",
            "OPENROUTER_ROUTER_MODEL": "router-model",
            "OPENROUTER_CHEAP_MODEL": "cheap-model",
            "OPENROUTER_STRONG_MODEL": "strong-model",
            "OPENROUTER_AGENT_MODEL": "agent-model",
            "OPENROUTER_CODE_MODEL": "code-model",
        },
    )

    settings = Settings()

    assert settings.router_mode == "llm"
    assert settings.use_mock is False
    assert settings.llm_provider == "openrouter"
    assert settings.openai_cheap_model == "openai-cheap"
    assert settings.openai_strong_model == "openai-strong"
    assert settings.openrouter_router_model == "router-model"
    assert settings.openrouter_cheap_model == "cheap-model"
    assert settings.openrouter_strong_model == "strong-model"
    assert settings.openrouter_agent_model == "agent-model"
    assert settings.openrouter_code_model == "code-model"
