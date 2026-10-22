import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agents.prompts import AGENT_SYSTEM_PROMPT
from app.agents.tool_registry import ToolRegistry
from app.agents.tool_selector import select_tool
from app.config import settings
from app.models import ChatMessage
from app.providers.base import ProviderConfigurationError, ProviderError, ProviderResponse
from app.providers.mock_provider import MockProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterProvider
from app.providers.retry import call_with_retries
from app.tools.base import ToolResult

logger = logging.getLogger("llm_router_agent.agent")


@dataclass(frozen=True)
class AgentRun:
    response: ProviderResponse
    tool_results: List[ToolResult]
    latency_ms: float

    @property
    def tool_result(self) -> Optional[ToolResult]:
        return self.tool_results[-1] if self.tool_results else None


class AgentService:
    def run(self, query: str, messages: List[ChatMessage]) -> AgentRun:
        start = time.perf_counter()
        if settings.use_mock:
            return self._run_mock(query, start)

        provider = _agent_provider()
        registry = ToolRegistry()
        history: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *(
                {"role": message.role.lower(), "content": message.content}
                for message in messages
                if message.role.lower() in {"user", "assistant"}
            ),
        ]
        tool_results: List[ToolResult] = []
        max_calls = max(0, settings.agent_max_tool_calls)
        # A final model request is permitted after the last tool result.
        while True:
            choice = "auto" if len(tool_results) < max_calls else "none"
            turn = call_with_retries(
                lambda: provider.complete_agent_turn(history, registry.definitions(), choice),
                "agent",
            ).value
            if not turn.tool_calls:
                response = ProviderResponse(
                    content=turn.content or "",
                    provider=turn.provider,
                    model=turn.model,
                    mock=False,
                )
                elapsed = round((time.perf_counter() - start) * 1000, 3)
                logger.info(
                    "agent_completed tools=%s agent_model=%s latency_ms=%.3f",
                    [result.tool_name for result in tool_results], response.model, elapsed,
                )
                return AgentRun(response=response, tool_results=tool_results, latency_ms=elapsed)

            if choice == "none":
                raise ProviderError("Agent model returned tool calls after tool use was disabled.")

            history.append({
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [call.to_payload() for call in turn.tool_calls],
            })
            for call in turn.tool_calls:
                if not isinstance(call.id, str) or not call.id:
                    raise ProviderError("Agent model returned a tool call without an ID.")
                if len(tool_results) >= max_calls:
                    result = ToolResult(
                        tool_name=call.name if isinstance(call.name, str) else "unknown",
                        success=False,
                        content="Tool call limit reached.",
                        metadata={"error_code": "tool_limit_reached", "safe_failure": True},
                    )
                else:
                    result = registry.execute(call)
                tool_results.append(result)
                history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({
                        "success": result.success,
                        "content": result.content,
                        "metadata": result.metadata,
                    }, ensure_ascii=False, default=str),
                })

    def _run_mock(self, query: str, start: float) -> AgentRun:
        # Mock mode has no model to request tools, so retain deterministic demo behavior.
        tool = select_tool(query)
        try:
            result = tool.run(query)
        except Exception:
            logger.exception("mock_tool_failed selected_tool=%s", tool.name)
            result = ToolResult(
                tool_name=tool.name,
                success=False,
                content="The tool failed. Try a narrower request.",
                metadata={"error_code": "tool_failed", "safe_failure": True},
            )
        if result.tool_name in {"repo_analysis", "code_assistant"} and result.success:
            content = result.content
        else:
            content = (
                "Mock agent response. Selected tool: %s. Tool success: %s. "
                "Tool result summary: %s"
            ) % (result.tool_name, result.success, result.content[:500])
        response = ProviderResponse(
            content=content, provider=MockProvider.provider_name, model="mock-agent", mock=True,
        )
        return AgentRun(
            response=response,
            tool_results=[result],
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )


def _agent_provider():
    provider_name = settings.llm_provider.lower()
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "openrouter":
        return OpenRouterProvider()
    raise ProviderConfigurationError("Unsupported LLM_PROVIDER: %s" % settings.llm_provider)
