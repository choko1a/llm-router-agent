import json
import os
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = Path(
    os.getenv("LLM_ROUTER_RUNTIME_CONFIG_PATH", str(PROJECT_ROOT / "config" / "runtime_config.json"))
)

DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "ROUTER_MODE": "rules",
    "USE_MOCK": True,
    "LLM_PROVIDER": "openai",
    "OPENAI_CHEAP_MODEL": "gpt-4o-mini",
    "OPENAI_STRONG_MODEL": "gpt-4o",
    "OPENROUTER_ROUTER_MODEL": "openai/gpt-4o-mini",
    "OPENROUTER_CHEAP_MODEL": "openai/gpt-4o-mini",
    "OPENROUTER_STRONG_MODEL": "openai/gpt-4o",
    "OPENROUTER_AGENT_MODEL": "openai/gpt-4o-mini",
    "OPENROUTER_CODE_MODEL": "deepseek/deepseek-chat-v3.1",
}

ALLOWED_RUNTIME_KEYS = set(DEFAULT_RUNTIME_CONFIG)
ROUTER_MODES = {"rules", "classifier", "llm"}
LLM_PROVIDERS = {"openai", "openrouter"}


def load_runtime_config(path: Path = RUNTIME_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    return _sanitize_runtime_config(payload)


def merged_runtime_config(path: Path = RUNTIME_CONFIG_PATH) -> Dict[str, Any]:
    config = DEFAULT_RUNTIME_CONFIG.copy()
    config.update(_sanitize_runtime_config({
        key: value for key in ALLOWED_RUNTIME_KEYS
        if (value := os.getenv(key)) is not None
    }))
    config.update(load_runtime_config(path))
    return config


def _sanitize_runtime_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in ALLOWED_RUNTIME_KEYS:
            continue
        if key == "ROUTER_MODE":
            normalized = str(value).strip().lower()
            if normalized in ROUTER_MODES:
                sanitized[key] = normalized
            continue
        if key == "LLM_PROVIDER":
            normalized = str(value).strip().lower()
            if normalized in LLM_PROVIDERS:
                sanitized[key] = normalized
            continue
        if key == "USE_MOCK":
            sanitized[key] = _coerce_bool(value)
            continue
        if isinstance(value, str) and value.strip():
            sanitized[key] = value.strip()
    return sanitized


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
