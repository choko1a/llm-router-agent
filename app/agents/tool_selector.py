from typing import List

from app.tools.base import Tool
from app.tools.code_assistant import CodeAssistantTool
from app.tools.repo_analysis import RepoAnalysisTool
from app.tools.web_search import WebSearchTool


def available_tools() -> List[Tool]:
    return [
        CodeAssistantTool(),
        RepoAnalysisTool(),
        WebSearchTool(),
    ]


def select_tool(query: str) -> Tool:
    for tool in available_tools():
        if tool.can_handle(query):
            return tool
    return WebSearchTool()
