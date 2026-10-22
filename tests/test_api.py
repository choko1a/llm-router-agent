import os

os.environ["USE_MOCK"] = "true"

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_route_simple() -> None:
    response = client.post("/route", json={"query": "Hello, what is LLM routing?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["route_type"] == "cheap"
    assert payload["route"] == "cheap"
    assert payload["confidence"] > 0
    assert payload["estimated_cost_usd"] > 0
    assert payload["estimated_latency_ms"] > 0
    assert payload["fallback_policy"]["cheap"] == "strong"


def test_route_reasoning_math() -> None:
    response = client.post(
        "/route",
        json={"query": "Solve this math problem and explain each reasoning step: 37 * 42."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route_type"] == "strong"
    assert payload["fallback_route"] == "agent"


def test_route_search_tools() -> None:
    response = client.post(
        "/route",
        json={"query": "Search the web for the latest pricing and summarize the sources."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route_type"] == "agent"
    assert payload["fallback_route"] is None


def test_route_code_debug() -> None:
    response = client.post(
        "/route",
        json={"query": "Debug this Python traceback and fix the failing test in the repo."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route_type"] == "agent"
    assert any(signal.startswith("code_or_debug:") for signal in payload["signals"])


def test_route_code_generation_guardrail() -> None:
    response = client.post(
        "/route",
        json={"query": "Сгенерируй FastAPI endpoint для health check"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route_type"] == "agent"
    assert any(signal.startswith("code_or_debug:") for signal in payload["signals"])


def test_route_privacy_sensitive() -> None:
    response = client.post(
        "/route",
        json={"query": "This contains private personal data, do not send it to a cheap model."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route_type"] == "strong"
    assert payload["fallback_route"] == "agent"
    assert any(signal.startswith("privacy_sensitive:") for signal in payload["signals"])


def test_chat_response() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Ping"}]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["mock"] is True
    assert payload["response"].startswith("Mock response")
    assert payload["route_type"] == "cheap"


def test_chat_agent_route_uses_agent_path_in_mock_mode() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Найди актуальные цены OpenAI API"}]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["route_type"] == "agent"
    assert payload["mock"] is True
    assert payload["provider"] == "mock"
    assert payload["model"] == "mock-agent"
    assert payload["selected_tool"] == "web_search"
    assert payload["router_mode"] in {"rules", "classifier", "llm"}
    assert "Selected tool: web_search" in payload["response"]
