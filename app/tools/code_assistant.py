from app.tools.base import Tool, ToolResult


class CodeAssistantTool(Tool):
    name = "code_assistant"
    description = (
        "Safe static coding assistant for debugging, code generation, tests, configs, "
        "and lightweight review. Does not execute code or shell commands."
    )

    KEYWORDS = (
        "ошибка",
        "traceback",
        "exception",
        "debug",
        "исправь код",
        "тесты падают",
        "pytest",
        "error",
        "importerror",
        "modulenotfounderror",
        "syntaxerror",
        "runtimeerror",
        "fastapi",
        "endpoint",
        "api endpoint",
        "сгенерируй код",
        "создай endpoint",
        "напиши тест",
        "dockerfile",
        "requirements",
        "config",
        "python code",
        "helper function",
        "script",
        "рефакторинг",
        "refactor",
        "оптимизируй код",
        "code review",
        "review code",
    )

    CODE_MARKERS = (
        "traceback",
        "exception",
        "error",
        "assert",
        "def ",
        "class ",
        "```",
        "import ",
        "from ",
    )

    def can_handle(self, query: str) -> bool:
        normalized = query.lower()
        return any(keyword in normalized for keyword in self.KEYWORDS)

    def run(self, query: str) -> ToolResult:
        normalized = query.lower()
        intent = _detect_intent(normalized)
        content = _build_content(query, normalized, intent)
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=content,
            metadata={
                "intent": intent,
                "executed_code": False,
                "executed_shell": False,
                "read_secrets": False,
                "safe_static_analysis": True,
                "has_code_context": any(marker in normalized for marker in self.CODE_MARKERS),
            },
        )


def _detect_intent(normalized: str) -> str:
    if any(marker in normalized for marker in ("traceback", "exception", "error", "ошибка", "modulenotfounderror")):
        return "debugging"
    if any(marker in normalized for marker in ("pytest", "напиши тест", "test")):
        return "test_generation"
    if any(marker in normalized for marker in ("fastapi", "endpoint", "создай endpoint")):
        return "fastapi_endpoint"
    if "dockerfile" in normalized or "requirements" in normalized or "config" in normalized:
        return "config_snippet"
    if any(marker in normalized for marker in ("refactor", "рефакторинг", "оптимизируй", "code review")):
        return "code_improvement"
    return "code_generation"


def _build_content(query: str, normalized: str, intent: str) -> str:
    sections = [
        "## Code Assistant Brief",
        "Safe static assistance only: no code, shell commands, eval, exec, or subprocess calls were executed.",
        "",
        "## Request Type",
        f"- `{intent}`",
        "",
    ]

    if intent == "debugging":
        sections.extend(_debugging_sections(normalized))
    elif intent == "test_generation":
        sections.extend(_pytest_sections(normalized))
    elif intent == "fastapi_endpoint":
        sections.extend(_fastapi_sections(normalized))
    elif intent == "config_snippet":
        sections.extend(_config_sections(normalized))
    elif intent == "code_improvement":
        sections.extend(_improvement_sections())
    else:
        sections.extend(_generation_sections())

    sections.extend(
        [
            "",
            "## Original Request",
            query[:2000],
        ]
    )
    return "\n".join(sections)


def _debugging_sections(normalized: str) -> list[str]:
    probable_cause = "Inspect the first failing frame, exception type, import path, and recent dependency changes."
    suggested_fix = "Provide a minimal fix, then add or update a focused regression test."
    if "modulenotfounderror" in normalized or "no module named" in normalized:
        probable_cause = "A required dependency is not installed in the active environment or is missing from requirements.txt."
        suggested_fix = "Install the missing package in the project venv and add it to requirements.txt if the app imports it."

    return [
        "## Probable Cause",
        f"- {probable_cause}",
        "",
        "## Suggested Fix",
        f"- {suggested_fix}",
        "",
        "## Verification",
        "- Re-run the narrow failing test first, then the full pytest suite.",
    ]


def _pytest_sections(normalized: str) -> list[str]:
    function_name = "sum_numbers" if "sum_numbers" in normalized else "target_function"
    return [
        "## Generated Pytest Example",
        "```python",
        f"from app.module import {function_name}",
        "",
        "",
        f"def test_{function_name}_returns_expected_result():",
        f"    assert {function_name}(2, 3) == 5",
        "",
        "",
        f"def test_{function_name}_handles_zero():",
        f"    assert {function_name}(0, 4) == 4",
        "```",
        "",
        "## Notes",
        "- Adjust the import path to the real module before running the test.",
    ]


def _fastapi_sections(normalized: str) -> list[str]:
    path = "/health" if "health" in normalized else "/items/{item_id}"
    return [
        "## Generated FastAPI Endpoint",
        "```python",
        "from fastapi import APIRouter",
        "",
        "router = APIRouter()",
        "",
        "",
        f"@router.get(\"{path}\")",
        "def health_check():",
        "    return {\"status\": \"ok\"}",
        "```",
        "",
        "## Notes",
        "- Register the router in `app.main` with `app.include_router(router)` if the project uses routers.",
    ]


def _config_sections(normalized: str) -> list[str]:
    if "dockerfile" in normalized:
        return [
            "## Dockerfile Snippet",
            "```dockerfile",
            "FROM python:3.11-slim",
            "WORKDIR /app",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "COPY . .",
            "CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
            "```",
        ]
    return [
        "## Config Snippet",
        "```text",
        "# Add the minimal required setting here and keep secrets in .env only.",
        "KEY=value",
        "```",
    ]


def _improvement_sections() -> list[str]:
    return [
        "## Lightweight Review Checklist",
        "- Keep functions small and name side effects explicitly.",
        "- Add type hints around public interfaces.",
        "- Prefer dependency injection for external services.",
        "- Cover bug fixes with focused regression tests.",
    ]


def _generation_sections() -> list[str]:
    return [
        "## Generated Code Guidance",
        "- Produce a minimal, focused implementation.",
        "- Include imports, edge cases, and a short usage note.",
        "- Add pytest coverage for expected behavior and one edge case.",
    ]
