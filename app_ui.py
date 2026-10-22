import json
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import dotenv_values, load_dotenv

from app.runtime_config import DEFAULT_RUNTIME_CONFIG, PROJECT_ROOT, RUNTIME_CONFIG_PATH, merged_runtime_config


load_dotenv()

BACKEND_URL = "http://127.0.0.1:8000"
MODEL_KEYS = tuple(key for key in DEFAULT_RUNTIME_CONFIG if key.endswith("_MODEL"))


def main() -> None:
    st.set_page_config(page_title="LLM Router Agent", layout="wide")
    st.title("LLM Router Agent")

    config = _sidebar()
    _chat(config)


def _sidebar() -> Dict[str, Any]:
    st.sidebar.header("Runtime settings")
    config = merged_runtime_config()

    selected: Dict[str, Any] = config.copy()
    selected["ROUTER_MODE"] = st.sidebar.selectbox(
        "ROUTER_MODE",
        ["rules", "classifier", "llm"],
        index=["rules", "classifier", "llm"].index(config["ROUTER_MODE"]),
    )
    selected["USE_MOCK"] = st.sidebar.selectbox(
        "USE_MOCK",
        [True, False],
        index=[True, False].index(bool(config["USE_MOCK"])),
        format_func=lambda value: str(value).lower(),
    )
    providers = ["openai", "openrouter"]
    selected["LLM_PROVIDER"] = st.sidebar.selectbox(
        "LLM_PROVIDER",
        providers,
        index=providers.index(config["LLM_PROVIDER"]),
    )
    if selected["LLM_PROVIDER"] == "openai":
        model_keys = ("OPENAI_CHEAP_MODEL", "OPENAI_STRONG_MODEL")
    else:
        model_keys = (
            "OPENROUTER_ROUTER_MODEL",
            "OPENROUTER_CHEAP_MODEL",
            "OPENROUTER_STRONG_MODEL",
            "OPENROUTER_AGENT_MODEL",
            "OPENROUTER_CODE_MODEL",
        )
    for key in model_keys:
        selected[key] = st.sidebar.text_input(key, value=str(config[key])).strip()
    if selected["LLM_PROVIDER"] == "openrouter":
        st.sidebar.caption("Agent Tool Calling uses OPENROUTER_AGENT_MODEL for every step; OPENROUTER_CODE_MODEL is a legacy setting.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Set the selected provider's API key in `.env`, then restart the backend. "
        "Saved runtime settings take priority over `.env` model IDs. "
        "Save these settings before chatting. Keys are never saved here."
    )

    if st.sidebar.button("Load model IDs from .env"):
        if _sync_models_from_env():
            st.rerun()
        else:
            st.sidebar.warning("No model IDs found in .env.")

    if st.sidebar.button("Save runtime settings", type="primary"):
        if any(not selected[key] for key in model_keys):
            st.sidebar.error("Model IDs cannot be empty.")
        else:
            _save_runtime_config(selected)
            st.sidebar.success("Saved to config/runtime_config.json")

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    selected["MEMORY_TURNS"] = st.sidebar.number_input(
        "Memory turns",
        min_value=1,
        max_value=20,
        value=6,
        step=1,
        help="Keep only this many recent user-and-assistant turns in the UI and model context.",
    )
    st.sidebar.caption("Older turns are removed from this UI session and are not sent to the model.")

    with st.sidebar.expander("Current models", expanded=True):
        for key in ("ROUTER_MODE", "USE_MOCK", "LLM_PROVIDER", *model_keys):
            value = selected[key]
            st.code(f"{key}={value}", language="text")

    return selected


def _chat(config: Dict[str, Any]) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    history_area = st.container()
    prompt = st.chat_input("Ask the router agent...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages = _trim_conversation(
        st.session_state.messages, int(config["MEMORY_TURNS"])
    )

    with history_area:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("metadata"):
                    _metadata(message["metadata"])

        if not prompt:
            return

        with st.chat_message("assistant"):
            with st.spinner("Routing and generating..."):
                try:
                    response = _send_chat(st.session_state.messages)
                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except (ValueError, AttributeError):
                        detail = str(exc)
                    st.error(f"Backend request failed: {detail}")
                    return
                except requests.RequestException as exc:
                    st.error(f"Backend request failed: {exc}")
                    return

            content = response.get("response", "")
            st.markdown(content)
            metadata = _response_metadata(response, config)
            _metadata(metadata)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "metadata": metadata,
                }
            )


def _trim_conversation(messages: List[Dict[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
    user_positions = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if len(user_positions) <= max_turns:
        return messages
    return messages[user_positions[-max_turns] :]


def _send_chat(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    conversation = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={"messages": conversation},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _response_metadata(response: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected route": response.get("route_type"),
        "executed route": response.get("executed_route") or response.get("route_type"),
        "fallback history": response.get("fallback_history") or [],
        "selected tool": response.get("selected_tool") or "none",
        "selected tools": response.get("selected_tools") or [],
        "selected model": response.get("model"),
        "provider": response.get("provider"),
        "router mode": response.get("router_mode") or config["ROUTER_MODE"],
        "latency ms": response.get("latency_ms"),
        "mock": response.get("mock"),
    }


def _metadata(metadata: Dict[str, Any]) -> None:
    with st.expander("Metadata", expanded=False):
        st.json(metadata)


def _save_runtime_config(config: Dict[str, Any]) -> None:
    safe_config = {key: config.get(key, DEFAULT_RUNTIME_CONFIG[key]) for key in DEFAULT_RUNTIME_CONFIG}
    path = Path(RUNTIME_CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sync_models_from_env() -> bool:
    env_values = dotenv_values(PROJECT_ROOT / ".env")
    model_values = {
        key: str(env_values[key]).strip()
        for key in MODEL_KEYS
        if env_values.get(key) and str(env_values[key]).strip()
    }
    if not model_values:
        return False
    config = merged_runtime_config()
    config.update(model_values)
    _save_runtime_config(config)
    return True


if __name__ == "__main__":
    main()
