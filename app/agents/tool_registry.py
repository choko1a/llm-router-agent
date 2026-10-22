import json
import logging
from typing import Any, Dict, List, Optional

from app.providers.agent_types import AgentToolCall
from app.tools.base import Tool, ToolResult
from app.tools.code_assistant import CodeAssistantTool
from app.tools.repo_analysis import RepoAnalysisTool
from app.tools.web_search import WebSearchTool


logger = logging.getLogger("llm_router_agent.tools")

TOOL_DESCRIPTIONS = {
    "code_assistant": "Analyze code, errors, tests, and implementation requests. Does not execute code.",
    "repo_analysis": "Summarize this repository or a GitHub repository mentioned in the request.",
    "web_search": "Search the web for current information using the configured search provider.",
}


class ToolRegistry:
    def __init__(self) -> None:
        # Only these instances can be invoked. Model output is never used as an attribute or path.
        tools: List[Tool] = [CodeAssistantTool(), RepoAnalysisTool(), WebSearchTool()]
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The request to pass to this tool."}
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
            for name in self._tools
        ]

    def execute(self, call: AgentToolCall) -> ToolResult:
        name = call.name if isinstance(call.name, str) else "unknown"
        tool: Optional[Tool] = self._tools.get(name)
        if tool is None:
            return _error(name, "unknown_tool", "Tool is not available.")
        if not isinstance(call.arguments, str):
            return _error(name, "invalid_json", "Tool arguments must be a JSON object.")
        try:
            args = json.loads(call.arguments)
        except (ValueError, TypeError):
            return _error(name, "invalid_json", "Tool arguments contain invalid JSON.")
        if not isinstance(args, dict):
            return _error(name, "invalid_arguments", "Tool arguments must be an object.")
        if "query" not in args:
            return _error(name, "missing_query", "The query argument is required.")
        if set(args) != {"query"}:
            return _error(name, "unexpected_arguments", "Only the query argument is allowed.")
        query = args["query"]
        if not isinstance(query, str):
            return _error(name, "invalid_query", "The query argument must be a string.")
        if not query.strip() or len(query) > 4000:
            return _error(name, "invalid_query", "The query must contain 1 to 4000 characters.")
        try:
            return tool.run(query)
        except Exception:
            logger.exception("tool_failed selected_tool=%s", name)
            return _error(name, "tool_failed", "The tool failed. Try a narrower request.")


def _error(name: str, code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_name=name,
        success=False,
        content=message,
        metadata={"error_code": code, "safe_failure": True},
    )
