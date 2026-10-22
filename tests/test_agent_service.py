import json
from types import SimpleNamespace

import pytest

from app.agents import agent_service
from app.agents.agent_service import AgentService
from app.agents.tool_registry import ToolRegistry
from app.models import ChatMessage
from app.providers.agent_types import AgentToolCall, AgentTurn
from app.providers.base import ProviderError
from app.providers import retry
from app.tools.base import ToolResult


def _call(call_id, name, arguments):
    return AgentToolCall(id=call_id, name=name, arguments=arguments)


def _turn(content=None, calls=None):
    return AgentTurn(content=content, tool_calls=calls or [], provider="openrouter", model="agent-model")


def _setup(monkeypatch, turns, max_calls=3):
    requests = []

    class FakeProvider:
        def complete_agent_turn(self, messages, tools, tool_choice):
            requests.append((list(messages), tools, tool_choice))
            return turns.pop(0)

    monkeypatch.setattr(agent_service, "settings", SimpleNamespace(
        use_mock=False, llm_provider="openrouter", agent_max_tool_calls=max_calls,
    ))
    monkeypatch.setattr(agent_service, "OpenRouterProvider", FakeProvider)
    return requests


def test_agent_can_answer_without_tool_and_preserves_user_history(monkeypatch):
    requests = _setup(monkeypatch, [_turn("I remember Lin.")])
    messages = [
        ChatMessage(role="user", content="My name is Lin."),
        ChatMessage(role="assistant", content="Hello, Lin."),
        ChatMessage(role="user", content="Search if needed, and recall my name."),
    ]
    run = AgentService().run(messages[-1].content, messages)
    assert run.response.content == "I remember Lin."
    assert run.tool_results == []
    assert requests[0][2] == "auto"
    assert requests[0][0][1:] == [
        {"role": "user", "content": "My name is Lin."},
        {"role": "assistant", "content": "Hello, Lin."},
        {"role": "user", "content": "Search if needed, and recall my name."},
    ]
    assert {tool["function"]["name"] for tool in requests[0][1]} == {
        "code_assistant", "repo_analysis", "web_search",
    }


def test_agent_uses_second_tool_after_first_result(monkeypatch):
    requests = _setup(monkeypatch, [
        _turn(calls=[_call("call_1", "code_assistant", '{"query":"Debug FastAPI"}')]),
        _turn(calls=[_call("call_2", "repo_analysis", '{"query":"Analyze this repo"}')]),
        _turn("Combined answer"),
    ])
    monkeypatch.setattr(ToolRegistry, "execute", lambda self, call: ToolResult(
        tool_name=call.name, success=True, content="result of " + call.name,
    ))
    run = AgentService().run("Debug and analyze", [ChatMessage(role="user", content="Debug and analyze")])
    assert run.response.content == "Combined answer"
    assert [r.tool_name for r in run.tool_results] == ["code_assistant", "repo_analysis"]
    assert [request[2] for request in requests] == ["auto", "auto", "auto"]
    second_history = requests[1][0]
    assert second_history[-2]["role"] == "assistant"
    assert second_history[-2]["tool_calls"][0]["id"] == "call_1"
    assert second_history[-1]["tool_call_id"] == "call_1"
    assert json.loads(second_history[-1]["content"])["content"] == "result of code_assistant"
    third_history = requests[2][0]
    assert third_history[-1]["tool_call_id"] == "call_2"
    assert third_history[1] == {"role": "user", "content": "Debug and analyze"}


@pytest.mark.parametrize("name,args,code", [
    ("__import__", '{"query":"x"}', "unknown_tool"),
    ("code_assistant", "{bad", "invalid_json"),
    ("code_assistant", "{}", "missing_query"),
    ("code_assistant", '{"query":42}', "invalid_query"),
    ("code_assistant", '{"query":"x","path":".env"}', "unexpected_arguments"),
])
def test_invalid_tool_requests_become_safe_results(monkeypatch, name, args, code):
    requests = _setup(monkeypatch, [
        _turn(calls=[_call("bad_call", name, args)]),
        _turn("I could not use that tool."),
    ])
    run = AgentService().run("Help", [ChatMessage(role="user", content="Help")])
    assert run.tool_results[0].success is False
    assert run.tool_results[0].metadata["error_code"] == code
    assert requests[1][0][-1]["tool_call_id"] == "bad_call"
    assert json.loads(requests[1][0][-1]["content"])["success"] is False


def test_limit_disables_more_tools_but_allows_final_answer(monkeypatch):
    requests = _setup(monkeypatch, [
        _turn(calls=[_call("first", "missing", "{}")]),
        _turn("Final answer"),
    ], max_calls=1)
    run = AgentService().run("Help", [ChatMessage(role="user", content="Help")])
    assert run.response.content == "Final answer"
    assert [request[2] for request in requests] == ["auto", "none"]


def test_registry_rejects_non_object_and_oversize_arguments():
    registry = ToolRegistry()
    assert registry.execute(_call("1", "code_assistant", "[]")).metadata["error_code"] == "invalid_arguments"
    assert registry.execute(_call("2", "code_assistant", json.dumps({"query": "x" * 4001}))).metadata["error_code"] == "invalid_query"


def test_agent_model_turn_is_retried_without_duplicate_tool_execution(monkeypatch):
    requests = []
    tool_executions = []

    class FlakyProvider:
        def complete_agent_turn(self, messages, tools, tool_choice):
            requests.append(tool_choice)
            if len(requests) == 1:
                return _turn(calls=[_call("once", "code_assistant", '{"query":"Check code"}')])
            if len(requests) <= 3:
                raise ProviderError("temporary failure")
            return _turn("Recovered answer")

    monkeypatch.setattr(agent_service, "settings", SimpleNamespace(
        use_mock=False, llm_provider="openrouter", agent_max_tool_calls=3,
    ))
    monkeypatch.setattr(agent_service, "OpenRouterProvider", FlakyProvider)
    monkeypatch.setattr(ToolRegistry, "execute", lambda self, call: (
        tool_executions.append(call.id) or ToolResult(
            tool_name=call.name, success=True, content="tool result",
        )
    ))
    monkeypatch.setattr(retry, "settings", SimpleNamespace(model_max_retries=2))
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)

    run = AgentService().run("Hello", [ChatMessage(role="user", content="Hello")])

    assert run.response.content == "Recovered answer"
    assert [result.tool_name for result in run.tool_results] == ["code_assistant"]
    assert tool_executions == ["once"]
    assert requests == ["auto", "auto", "auto", "auto"]
