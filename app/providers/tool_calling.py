from typing import Any, Dict, List

from app.providers.agent_types import AgentToolCall, AgentTurn
from app.providers.base import ProviderError


def complete_agent_turn(
    *, client: Any, model: str, provider: str, messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]], tool_choice: str,
) -> AgentTurn:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
            temperature=0.2,
        )
        message = completion.choices[0].message
        calls = [
            AgentToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        ]
        return AgentTurn(
            content=message.content,
            tool_calls=calls,
            provider=provider,
            model=model,
        )
    except Exception as exc:
        raise ProviderError("%s tool-calling request failed: %s" % (provider, exc)) from exc
