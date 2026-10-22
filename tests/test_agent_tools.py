from types import SimpleNamespace

import pytest

from app.agents.tool_selector import select_tool
from app.tools import repo_analysis, web_search
from app.tools.code_assistant import CodeAssistantTool
from app.tools.code_debug import CodeDebugTool
from app.tools.repo_analysis import RepoAnalysisTool
from app.tools.web_search import WebSearchTool


def test_tool_selector_chooses_web_search() -> None:
    tool = select_tool("search актуальные цены OpenAI API")

    assert isinstance(tool, WebSearchTool)


def test_tool_selector_chooses_repo_analysis() -> None:
    tool = select_tool("Посмотри структуру проекта и README")

    assert isinstance(tool, RepoAnalysisTool)


def test_tool_selector_chooses_repo_analysis_for_repo_summary() -> None:
    assert isinstance(select_tool("опиши репозиторий"), RepoAnalysisTool)
    assert isinstance(select_tool("что делает этот проект"), RepoAnalysisTool)
    assert isinstance(select_tool("проанализируй https://github.com/langchain-ai/langgraph"), RepoAnalysisTool)


def test_tool_selector_chooses_code_assistant() -> None:
    tool = select_tool("pytest падает с traceback, помоги debug")

    assert isinstance(tool, CodeAssistantTool)
    assert tool.name == "code_assistant"


def test_tool_selector_chooses_code_assistant_for_generation() -> None:
    assert isinstance(select_tool("Сгенерируй FastAPI endpoint для health check"), CodeAssistantTool)
    assert isinstance(select_tool("Напиши pytest для функции sum_numbers(a, b)"), CodeAssistantTool)
    assert isinstance(select_tool("Сделай Dockerfile для FastAPI"), CodeAssistantTool)
    assert isinstance(select_tool("Нужен рефакторинг Python code"), CodeAssistantTool)


def test_repo_analysis_does_not_read_env() -> None:
    result = RepoAnalysisTool().run("Проанализируй файлы проекта")

    assert result.success is True
    assert ".env" in result.metadata["ignored"]
    assert ".env" not in result.metadata["read_files"]
    assert "OPENROUTER_API_KEY" not in result.content


def test_repo_analysis_reads_readme_and_returns_russian_summary(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "# Demo Repo\n\nПроект маршрутизирует LLM-запросы и запускает agent tools.",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("from fastapi import FastAPI\n", encoding="utf-8")

    result = RepoAnalysisTool(tmp_path).run("сделай summary репозитория")

    assert result.success is True
    assert result.metadata["read_files"] == ["README.md"]
    assert "## Краткое описание" in result.content
    assert "## Структура репозитория" in result.content
    assert "## Ключевые компоненты" in result.content
    assert "## Что уже реализовано" in result.content
    assert "## Что можно улучшить" in result.content
    assert "Проект маршрутизирует LLM-запросы" in result.content
    assert "FastAPI backend" in result.content


def test_repo_analysis_ignores_sensitive_dirs_and_binary_files(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# Safe Repo\n\nОписание проекта.", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secretsecretsecretsecret\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret-token", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = secret", encoding="utf-8")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "pyvenv.cfg").write_text("home = secret", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "evaluation_results.json").write_text('{"secret":"value"}', encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "router_classifier.pkl").write_bytes(b"pickle-secret")
    (tmp_path / "brief.docx").write_bytes(b"docx-secret")
    (tmp_path / "slides.pptx").write_bytes(b"pptx-secret")
    (tmp_path / "paper.pdf").write_bytes(b"pdf-secret")

    result = RepoAnalysisTool(tmp_path).run("проанализируй проект")

    assert result.metadata["read_files"] == ["README.md"]
    assert ".git" not in result.content
    assert ".venv" not in result.content
    assert "venv" not in result.content
    assert "evaluation_results.json" not in result.content
    assert "router_classifier.pkl" not in result.content
    assert "brief.docx" not in result.content
    assert "slides.pptx" not in result.content
    assert "paper.pdf" not in result.content
    assert "sk-secret" not in result.content
    assert "secret-token" not in result.content
    assert "pickle-secret" not in result.content


def test_repo_analysis_handles_public_github_url(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_analysis,
        "_fetch_github_top_level",
        lambda owner, repo: ["README.md", "docs/", "libs/", "pyproject.toml"],
    )
    monkeypatch.setattr(
        repo_analysis,
        "_fetch_github_readme",
        lambda owner, repo: (
            "# LangGraph\n\n"
            "LangGraph is a low-level orchestration framework for stateful agents. "
            "It supports graphs, persistence, checkpointing, and human-in-the-loop workflows."
        ),
    )

    result = RepoAnalysisTool().run("Проанализируй репозиторий https://github.com/langchain-ai/langgraph")

    assert result.success is True
    assert result.metadata["source"] == "github"
    assert result.metadata["repository"] == "langchain-ai/langgraph"
    assert result.metadata["clone_used"] is False
    assert result.metadata["read_files"] == ["README.md"]
    assert "## Краткое описание" in result.content
    assert "## Чем отличается от обычного LangChain" in result.content
    assert "LangGraph" in result.content
    assert "docs/" in result.content
    assert ".env" not in result.content
    assert ".git" not in result.content


def test_repo_analysis_github_rate_limit_falls_back_to_raw_readme(monkeypatch) -> None:
    monkeypatch.setattr(
        repo_analysis,
        "_fetch_github_top_level",
        lambda owner, repo: (_ for _ in ()).throw(RuntimeError("GitHub repository analysis failed: 403 rate limit")),
    )
    monkeypatch.setattr(
        repo_analysis,
        "_fetch_github_readme",
        lambda owner, repo: (_ for _ in ()).throw(RuntimeError("GitHub README API failed: 403 rate limit")),
    )
    monkeypatch.setattr(
        repo_analysis,
        "_fetch_github_readme_raw",
        lambda owner, repo: "# LangGraph\n\nRaw README fallback content.",
    )

    result = RepoAnalysisTool().run("Проанализируй https://github.com/langchain-ai/langgraph")

    assert result.success is True
    assert result.metadata["source"] == "github"
    assert result.metadata["clone_used"] is False
    assert result.metadata["warnings"]
    assert "Raw README fallback content" in result.content
    assert "Структуру не удалось получить" in result.content


def test_code_assistant_traceback_analysis_works() -> None:
    result = CodeAssistantTool().run("Исправь этот traceback: ModuleNotFoundError: No module named 'fastapi'")

    assert result.success is True
    assert result.tool_name == "code_assistant"
    assert result.metadata["intent"] == "debugging"
    assert result.metadata["executed_code"] is False
    assert "Probable Cause" in result.content
    assert "requirements.txt" in result.content


def test_code_assistant_generates_fastapi_endpoint() -> None:
    result = CodeAssistantTool().run("Сгенерируй FastAPI endpoint для health check")

    assert result.metadata["intent"] == "fastapi_endpoint"
    assert "```python" in result.content
    assert "@router.get(\"/health\")" in result.content
    assert "return {\"status\": \"ok\"}" in result.content


def test_code_assistant_generates_pytest_with_asserts() -> None:
    result = CodeAssistantTool().run("Напиши pytest для функции sum_numbers(a, b)")

    assert result.metadata["intent"] == "test_generation"
    assert "```python" in result.content
    assert "def test_sum_numbers_returns_expected_result" in result.content
    assert "assert sum_numbers(2, 3) == 5" in result.content


def test_code_assistant_does_not_execute_code() -> None:
    result = CodeAssistantTool().run("debug этот код: raise Exception('boom')")

    assert result.success is True
    assert result.metadata["executed_code"] is False
    assert result.metadata["executed_shell"] is False
    assert result.metadata["safe_static_analysis"] is True
    assert "no code, shell commands, eval, exec, or subprocess calls were executed" in result.content.lower()


def test_code_debug_alias_keeps_compatibility() -> None:
    result = CodeDebugTool().run("debug этот код: raise Exception('boom')")

    assert result.success is True
    assert result.tool_name == "code_assistant"
    assert result.metadata["executed_code"] is False


def test_web_search_uses_mock_when_use_mock_true(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search,
        "settings",
        SimpleNamespace(use_mock=True, web_search_mock=False, tavily_api_key="secret"),
    )

    result = WebSearchTool().run("search OpenAI pricing")

    assert result.success is True
    assert result.metadata["provider"] == "mock"
    assert result.metadata["mocked"] is True


def test_web_search_reports_missing_tavily_key_in_real_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search,
        "settings",
        SimpleNamespace(use_mock=False, web_search_mock=False, tavily_api_key=""),
    )

    result = WebSearchTool().run("search OpenAI pricing")

    assert result.success is False
    assert result.metadata["provider"] == "tavily"
    assert result.metadata["mocked"] is False
    assert result.metadata["missing_api_key"] is True


def test_web_search_normalizes_fake_tavily_response(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search,
        "settings",
        SimpleNamespace(
            use_mock=False,
            web_search_mock=False,
            tavily_api_key="fake",
            web_search_max_results=5,
        ),
    )
    monkeypatch.setattr(
        web_search,
        "_search_tavily",
        lambda query: {
            "results": [
                {
                    "title": "OpenAI pricing",
                    "url": "https://example.com/pricing",
                    "content": "Pricing summary snippet.",
                }
            ]
        },
    )

    result = WebSearchTool().run("search OpenAI pricing")

    assert result.success is True
    assert result.metadata["provider"] == "tavily"
    assert result.metadata["mocked"] is False
    assert result.metadata["result_count"] == 1
    assert "OpenAI pricing" in result.content
    assert "https://example.com/pricing" in result.content


def test_web_search_handles_tavily_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search,
        "settings",
        SimpleNamespace(
            use_mock=False,
            web_search_mock=False,
            tavily_api_key="fake",
            web_search_max_results=5,
        ),
    )

    def raise_tavily_error(query):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(web_search, "_search_tavily", raise_tavily_error)

    result = WebSearchTool().run("search OpenAI pricing")

    assert result.success is False
    assert result.metadata["provider"] == "tavily"
    assert result.metadata["mocked"] is False
    assert result.metadata["error_code"] == "search_failed"
    assert "Web search failed" in result.content


@pytest.mark.parametrize("error,code", [
    (ModuleNotFoundError("No module named 'tavily'", name="tavily"), "dependency_missing"),
    (type("InvalidAPIKeyError", (Exception,), {})("secret response"), "invalid_api_key"),
    (type("UsageLimitExceededError", (Exception,), {})("secret response"), "quota_exceeded"),
    (TimeoutError("secret response"), "timeout"),
])
def test_web_search_returns_actionable_safe_error(monkeypatch, error, code) -> None:
    monkeypatch.setattr(
        web_search,
        "settings",
        SimpleNamespace(
            use_mock=False,
            web_search_mock=False,
            tavily_api_key="secret",
            web_search_max_results=5,
        ),
    )

    def fail(query):
        raise error

    monkeypatch.setattr(web_search, "_search_tavily", fail)
    result = WebSearchTool().run("search OpenAI pricing")

    assert result.success is False
    assert result.metadata["error_code"] == code
    assert "secret response" not in result.content
