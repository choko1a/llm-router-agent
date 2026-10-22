import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.runtime_config import load_runtime_config


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _runtime_value(runtime_config, key: str, env_name: str, default):
    if key in runtime_config:
        return runtime_config[key]
    return os.getenv(env_name, default)


@dataclass
class Settings:
    app_name: str = ""
    app_version: str = ""
    log_level: str = ""
    router_mode: str = ""
    router_classifier_model_path: str = ""
    use_mock: bool = True
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_cheap_model: str = ""
    openai_strong_model: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = ""
    openrouter_router_model: str = ""
    openrouter_cheap_model: str = ""
    openrouter_strong_model: str = ""
    openrouter_agent_model: str = ""
    openrouter_code_model: str = ""
    agent_max_tool_calls: int = 3
    model_max_retries: int = 2
    agent_enabled: bool = True
    tavily_api_key: str = ""
    web_search_provider: str = ""
    web_search_max_results: int = 5
    web_search_mock: bool = False
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        runtime_config = load_runtime_config()
        self.app_name = os.getenv("APP_NAME", "LLM Router Agent")
        self.app_version = os.getenv("APP_VERSION", "0.1.0")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.router_mode = str(_runtime_value(runtime_config, "ROUTER_MODE", "ROUTER_MODE", "rules"))
        self.router_classifier_model_path = os.getenv(
            "ROUTER_CLASSIFIER_MODEL_PATH",
            "models/router_classifier.pkl",
        )
        self.use_mock = bool(runtime_config["USE_MOCK"]) if "USE_MOCK" in runtime_config else _env_bool("USE_MOCK", True)
        self.llm_provider = str(_runtime_value(runtime_config, "LLM_PROVIDER", "LLM_PROVIDER", "openai")).lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_cheap_model = str(_runtime_value(runtime_config, "OPENAI_CHEAP_MODEL", "OPENAI_CHEAP_MODEL", "gpt-4o-mini"))
        self.openai_strong_model = str(_runtime_value(runtime_config, "OPENAI_STRONG_MODEL", "OPENAI_STRONG_MODEL", "gpt-4o"))
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_router_model = str(
            _runtime_value(
                runtime_config,
                "OPENROUTER_ROUTER_MODEL",
                "OPENROUTER_ROUTER_MODEL",
                "openai/gpt-4o-mini",
            )
        )
        self.openrouter_cheap_model = str(
            _runtime_value(
                runtime_config,
                "OPENROUTER_CHEAP_MODEL",
                "OPENROUTER_CHEAP_MODEL",
                "openai/gpt-4o-mini",
            )
        )
        self.openrouter_strong_model = str(
            _runtime_value(runtime_config, "OPENROUTER_STRONG_MODEL", "OPENROUTER_STRONG_MODEL", "openai/gpt-4o")
        )
        self.openrouter_agent_model = str(
            _runtime_value(
                runtime_config,
                "OPENROUTER_AGENT_MODEL",
                "OPENROUTER_AGENT_MODEL",
                "openai/gpt-4o-mini",
            )
        )
        self.openrouter_code_model = str(
            _runtime_value(
                runtime_config,
                "OPENROUTER_CODE_MODEL",
                "OPENROUTER_CODE_MODEL",
                "deepseek/deepseek-chat-v3.1",
            )
        )
        self.agent_max_tool_calls = int(os.getenv("AGENT_MAX_TOOL_CALLS", "3"))
        self.model_max_retries = int(os.getenv("MODEL_MAX_RETRIES", "2"))
        self.agent_enabled = _env_bool("AGENT_ENABLED", True)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.web_search_provider = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
        self.web_search_max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
        self.web_search_mock = _env_bool("WEB_SEARCH_MOCK", False)
        self._initialized = True


settings = Settings()
