import logging
from typing import Any, Dict, List

from app.config import settings
from app.tools.base import Tool, ToolResult

logger = logging.getLogger("llm_router_agent.web_search")


class WebSearchTool(Tool):
    name = "web_search"
    description = "Web search tool backed by Tavily when configured."

    KEYWORDS = (
        "\u043d\u0430\u0439\u0434\u0438",
        "\u043f\u043e\u0438\u0449\u0438",
        "\u0430\u043a\u0442\u0443\u0430\u043b\u044c\u043d\u044b\u0435",
        "\u0441\u0432\u0435\u0436\u0438\u0435",
        "\u0446\u0435\u043d\u0430",
        "\u0446\u0435\u043d\u044b",
        "pricing",
        "search",
        "web",
    )

    def can_handle(self, query: str) -> bool:
        normalized = query.lower()
        return any(keyword in normalized for keyword in self.KEYWORDS)

    def run(self, query: str) -> ToolResult:
        if _should_use_mock():
            return _mock_result(query)
        if not settings.tavily_api_key:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="TAVILY_API_KEY is required for real web search.",
                metadata={"provider": "tavily", "mocked": False, "missing_api_key": True},
            )

        try:
            response = _search_tavily(query)
        except Exception as exc:
            error_code, message = _safe_search_error(exc)
            logger.warning(
                "web_search_failed error_code=%s exception_type=%s",
                error_code,
                type(exc).__name__,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                content=message,
                metadata={
                    "provider": "tavily",
                    "error_code": error_code,
                    "search_depth": "basic",
                    "mocked": False,
                },
            )

        results = _extract_results(response)
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=_format_results(results),
            metadata={
                "provider": "tavily",
                "result_count": len(results),
                "search_depth": "basic",
                "mocked": False,
            },
        )


def _should_use_mock() -> bool:
    return settings.use_mock or settings.web_search_mock


def _mock_result(query: str) -> ToolResult:
    return ToolResult(
        tool_name=WebSearchTool.name,
        success=True,
        content=(
            "Web search is using mock data. No live web request was made. "
            "Set USE_MOCK=false, WEB_SEARCH_MOCK=false, and TAVILY_API_KEY for live search."
        ),
        metadata={
            "provider": "mock",
            "mocked": True,
            "query_preview": query[:160],
        },
    )


def _search_tavily(query: str) -> Dict[str, Any]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    return client.search(
        query=query,
        search_depth="basic",
        max_results=settings.web_search_max_results,
        include_answer=False,
        include_raw_content=False,
        timeout=15,
    )


def _safe_search_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "tavily":
        return "dependency_missing", "Web search needs tavily-python installed in this Python environment."
    name = type(exc).__name__
    if name in {"InvalidAPIKeyError", "MissingAPIKeyError"}:
        return "invalid_api_key", "Tavily rejected the API key. Check TAVILY_API_KEY."
    if name in {"UsageLimitExceededError", "TavilyKeylessLimitError"}:
        return "quota_exceeded", "Tavily search quota is exhausted. Check the Tavily account."
    if name in {"Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutError"}:
        return "timeout", "Tavily search timed out. Try again later."
    if name in {"ConnectionError", "ConnectError", "APIConnectionError"}:
        return "network_error", "Cannot connect to Tavily. Check network access."
    return "search_failed", "Web search failed. Check backend logs for the error category."


def _extract_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = response.get("results", [])
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def _format_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Tavily search returned no results."

    lines: List[str] = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or "No URL"
        snippet = result.get("content") or result.get("snippet") or "No snippet available."
        lines.append("%s. %s\n   URL: %s\n   Snippet: %s" % (index, title, url, snippet))
    return "\n".join(lines)
