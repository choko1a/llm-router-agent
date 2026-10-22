import re
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.tools.base import Tool, ToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORED_FILES = {".env"}
IGNORED_SUFFIXES = {
    ".docx",
    ".pptx",
    ".pdf",
    ".exe",
    ".pkl",
    ".sqlite",
    ".mmap",
    ".pyc",
}
MAX_TOP_LEVEL_ENTRIES = 40
MAX_TREE_ENTRIES = 80
MAX_README_CHARS = 3500
MAX_GITHUB_RESPONSE_BYTES = 1_000_000
GITHUB_URL_PATTERN = re.compile(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:[/?#].*)?")
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
)


class RepoAnalysisTool(Tool):
    name = "repo_analysis"
    description = "Safely summarizes the local repository structure and README."

    KEYWORDS = (
        "репозиторий",
        "структура проекта",
        "readme",
        "файлы проекта",
        "архитектура проекта",
        "repo",
        "github.com",
        "опиши репозиторий",
        "сделай summary репозитория",
        "summary репозитория",
        "что делает этот проект",
        "проанализируй проект",
    )

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT

    def can_handle(self, query: str) -> bool:
        normalized = query.lower()
        return any(keyword in normalized for keyword in self.KEYWORDS)

    def run(self, query: str) -> ToolResult:
        github_repo = _extract_github_repo(query)
        if github_repo is not None:
            return _run_github_analysis(self.name, github_repo)

        root = self.project_root.resolve()
        entries = _safe_top_level_entries(root.iterdir())
        tree = _safe_project_tree(root)
        readme_text, read_files = _read_readme(root)
        components = _detect_components(root)
        content = _build_markdown_summary(root, entries, tree, readme_text, components)

        return ToolResult(
            tool_name=self.name,
            success=True,
            content=content,
            metadata={
                "ignored": sorted(IGNORED_DIRS | IGNORED_FILES | IGNORED_SUFFIXES),
                "entry_count": len(entries),
                "tree_entry_count": len(tree),
                "read_files": read_files,
                "max_readme_chars": MAX_README_CHARS,
            },
        )


def _run_github_analysis(tool_name: str, github_repo: tuple[str, str]) -> ToolResult:
    owner, repo = github_repo
    repo_label = f"{owner}/{repo}"
    warnings: List[str] = []
    try:
        root_entries = _fetch_github_top_level(owner, repo)
    except RuntimeError as exc:
        root_entries = []
        warnings.append(str(exc))
    try:
        readme_text = _fetch_github_readme(owner, repo)
    except RuntimeError as exc:
        readme_text = _fetch_github_readme_raw(owner, repo)
        warnings.append(str(exc))
    components = _detect_github_components(root_entries, readme_text)
    content = _build_github_markdown_summary(repo_label, root_entries, readme_text, components)

    return ToolResult(
        tool_name=tool_name,
        success=True,
        content=content,
        metadata={
            "source": "github",
            "repository": repo_label,
            "read_files": ["README.md"] if readme_text else [],
            "entry_count": len(root_entries),
            "ignored": sorted(IGNORED_DIRS | IGNORED_FILES | IGNORED_SUFFIXES),
            "network": "github_api_public",
            "clone_used": False,
            "max_github_response_bytes": MAX_GITHUB_RESPONSE_BYTES,
            "warnings": warnings,
        },
    )


def _safe_top_level_entries(paths: Iterable[Path]) -> List[str]:
    entries: List[str] = []
    for path in sorted(paths, key=lambda item: item.name.lower()):
        if _is_ignored_path(path):
            continue
        marker = "/" if path.is_dir() else ""
        entries.append(path.name + marker)
    return entries[:MAX_TOP_LEVEL_ENTRIES]


def _safe_project_tree(root: Path) -> List[str]:
    tree: List[str] = []
    _walk_safe_tree(root, root, 0, tree)
    return tree


def _walk_safe_tree(root: Path, current: Path, depth: int, tree: List[str]) -> None:
    if len(tree) >= MAX_TREE_ENTRIES or depth >= 3:
        return

    for path in sorted(current.iterdir(), key=lambda item: item.name.lower()):
        if len(tree) >= MAX_TREE_ENTRIES:
            return
        if _is_ignored_path(path, root):
            continue
        relative = path.relative_to(root)
        item_depth = len(relative.parts)
        prefix = "  " * (item_depth - 1)
        marker = "/" if path.is_dir() else ""
        tree.append(f"{prefix}- {path.name}{marker}")
        if path.is_dir():
            _walk_safe_tree(root, path, item_depth, tree)


def _read_readme(root: Path) -> tuple[str, List[str]]:
    for name in ("README.md", "readme.md"):
        path = root / name
        if path.exists() and not _is_ignored_path(path, root):
            text = path.read_text(encoding="utf-8", errors="replace")[:MAX_README_CHARS]
            return _redact_secrets(text.strip()), [path.name]
    return "README.md не найден.", []


def _extract_github_repo(query: str) -> Optional[tuple[str, str]]:
    match = GITHUB_URL_PATTERN.search(query)
    if not match:
        return None
    owner = match.group(1)
    repo = match.group(2).removesuffix(".git")
    return owner, repo


def _fetch_github_top_level(owner: str, repo: str) -> List[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    payload = _fetch_json(url)
    if not isinstance(payload, list):
        return []

    entries: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        item_type = str(item.get("type", ""))
        if not name or _is_ignored_remote_name(name):
            continue
        marker = "/" if item_type == "dir" else ""
        entries.append(name + marker)
    return entries[:MAX_TOP_LEVEL_ENTRIES]


def _fetch_github_readme(owner: str, repo: str) -> str:
    readme_meta = _fetch_json(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if not isinstance(readme_meta, dict):
        return ""
    download_url = readme_meta.get("download_url")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        return ""
    parsed = urlparse(download_url)
    if parsed.netloc not in {"raw.githubusercontent.com", "github.com"}:
        return ""
    return _redact_secrets(_fetch_text(download_url)[:MAX_README_CHARS].strip())


def _fetch_github_readme_raw(owner: str, repo: str) -> str:
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
        try:
            text = _fetch_text(url)
        except RuntimeError:
            continue
        if text.strip():
            return _redact_secrets(text[:MAX_README_CHARS].strip())
    return ""


def _fetch_json(url: str):
    text = _fetch_text(url)
    return json.loads(text)


def _fetch_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "llm-router-agent-repo-analysis",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            data = response.read(MAX_GITHUB_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"GitHub repository analysis failed: {exc}") from exc
    if len(data) > MAX_GITHUB_RESPONSE_BYTES:
        raise RuntimeError("GitHub response exceeded the lightweight size limit.")
    return data.decode("utf-8", errors="replace")


def _is_ignored_remote_name(name: str) -> bool:
    path = Path(name)
    return (
        name in IGNORED_DIRS
        or name in IGNORED_FILES
        or path.suffix.lower() in IGNORED_SUFFIXES
        or name == "reports"
        or name == "models"
    )


def _detect_components(root: Path) -> List[str]:
    checks = (
        ("FastAPI backend", ("app/main.py",)),
        ("rules router", ("app/routing.py", "app/routing_config.py")),
        ("classifier router", ("app/classifier_router.py",)),
        ("LLM router", ("app/llm_router.py",)),
        ("OpenAI/OpenRouter providers", ("app/providers/openai_provider.py", "app/providers/openrouter_provider.py")),
        ("agent route", ("app/agents/agent_service.py",)),
        ("tool selector", ("app/agents/tool_selector.py",)),
        ("repo_analysis tool", ("app/tools/repo_analysis.py",)),
        ("web_search tool", ("app/tools/web_search.py",)),
        ("code_assistant tool", ("app/tools/code_assistant.py", "app/tools/code_debug.py")),
        ("evaluation pipeline", ("scripts/evaluate_router.py", "data/eval_queries.jsonl")),
        ("trainable classifier", ("scripts/train_router_classifier.py", "models/router_classifier.pkl")),
        ("pytest coverage", ("tests",)),
    )
    components: List[str] = []
    for label, paths in checks:
        if any((root / path).exists() for path in paths):
            components.append(label)
    return components


def _detect_github_components(entries: Sequence[str], readme_text: str) -> List[str]:
    normalized_entries = {entry.rstrip("/").lower() for entry in entries}
    normalized_readme = readme_text.lower()
    components: List[str] = []

    checks = (
        ("README documentation", bool(readme_text)),
        ("Python package", "pyproject.toml" in normalized_entries or "setup.py" in normalized_entries),
        ("Source package", "libs" in normalized_entries or "src" in normalized_entries or "langgraph" in normalized_entries),
        ("Examples", "examples" in normalized_entries or "examples" in normalized_readme),
        ("Documentation", "docs" in normalized_entries or "documentation" in normalized_readme),
        ("Tests", "tests" in normalized_entries),
        ("Agent graphs", "graph" in normalized_readme or "stategraph" in normalized_readme),
        ("Persistence/checkpointing", "checkpoint" in normalized_readme or "persistence" in normalized_readme),
        ("Human-in-the-loop workflows", "human-in-the-loop" in normalized_readme or "human in the loop" in normalized_readme),
    )
    for label, present in checks:
        if present:
            components.append(label)
    return components


def _build_markdown_summary(
    root: Path,
    entries: Sequence[str],
    tree: Sequence[str],
    readme_text: str,
    components: Sequence[str],
) -> str:
    project_name = _project_title(readme_text) or root.name
    purpose = _purpose_from_readme(readme_text)

    return "\n".join(
        [
            "## Краткое описание",
            f"**{project_name}** - {purpose}",
            "",
            "## Структура репозитория",
            "Основные папки и файлы:",
            _format_bullets(entries),
            "",
            "Краткое дерево проекта:",
            _format_bullets(tree) if tree else "- Нет доступных файлов для показа.",
            "",
            "## Ключевые компоненты",
            _format_bullets(components) if components else "- Компоненты не распознаны по стандартным путям.",
            "",
            "## Что уже реализовано",
            _format_bullets(components) if components else "- По структуре проекта реализация пока не определена.",
            "",
            "## Что можно улучшить",
            "- Расширить repo_analysis чтением безопасных исходников по allowlist, если нужен более глубокий обзор.",
            "- Добавить оценку покрытия тестами по модулям и подсветку незакрытых сценариев.",
            "- Сохранять краткий machine-readable отчет для сравнения изменений архитектуры между версиями.",
            "",
            "### README.md",
            _trim_readme_for_display(readme_text),
        ]
    )


def _build_github_markdown_summary(
    repo_label: str,
    entries: Sequence[str],
    readme_text: str,
    components: Sequence[str],
) -> str:
    project_name = _project_title(readme_text) or repo_label
    purpose = _purpose_from_readme(readme_text)
    langchain_difference = _langgraph_difference(readme_text)

    return "\n".join(
        [
            "## Краткое описание",
            f"**{project_name}** - {purpose}",
            "",
            "## Структура репозитория",
            "Верхний уровень публичного GitHub-репозитория:",
            _format_bullets(entries) if entries else "- Структуру не удалось получить через GitHub API.",
            "",
            "## Ключевые компоненты",
            _format_bullets(components) if components else "- Компоненты не распознаны по README и верхнему уровню.",
            "",
            "## Что уже реализовано",
            "- Lightweight-анализ публичного GitHub URL без git clone.",
            "- Получение README.md и верхнего уровня репозитория через публичный GitHub API.",
            "- Ограничение размера сетевых ответов и игнорирование бинарных/секретных локальных путей.",
            "",
            "## Чем отличается от обычного LangChain",
            langchain_difference,
            "",
            "## Для каких AI agent scenarios подходит",
            "- Многошаговые agent workflows с явным состоянием.",
            "- Циклические графы принятия решений, где агент может возвращаться к предыдущим шагам.",
            "- Human-in-the-loop сценарии, где человеку нужно проверять или корректировать действия агента.",
            "- Долгоживущие агенты с persistence/checkpointing и восстановлением состояния.",
            "",
            "## Что можно улучшить",
            "- Добавить auth token опционально, чтобы мягче переживать GitHub API rate limits.",
            "- Поддержать выбор ветки или tag в GitHub URL.",
            "- Добавить allowlist чтения нескольких небольших markdown/docs файлов для более глубокого обзора.",
            "",
            "### README.md",
            _trim_readme_for_display(readme_text),
        ]
    )


def _is_ignored_path(path: Path, root: Optional[Path] = None) -> bool:
    try:
        parts = path.relative_to(root).parts if root else path.parts
    except ValueError:
        parts = path.parts

    if any(part in IGNORED_DIRS or part in IGNORED_FILES for part in parts):
        return True
    if len(parts) >= 2 and parts[-2] == "reports" and path.suffix.lower() == ".json":
        return True
    if len(parts) >= 2 and parts[-2] == "models" and path.suffix.lower() == ".pkl":
        return True
    return path.suffix.lower() in IGNORED_SUFFIXES


def _redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _project_title(readme_text: str) -> str:
    for line in readme_text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return ""


def _purpose_from_readme(readme_text: str) -> str:
    for line in readme_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        return stripped.rstrip(".") + "."
    return "локальный проект, для которого доступна безопасная структура файлов."


def _langgraph_difference(readme_text: str) -> str:
    normalized = readme_text.lower()
    if "low-level orchestration" in normalized or "stateful" in normalized:
        return (
            "LangGraph фокусируется на stateful orchestration: графах, циклах, persistence "
            "и управляемом выполнении agent workflow. Обычный LangChain чаще используется как "
            "набор компонентов и цепочек для вызова моделей, retrieval и интеграций, а LangGraph "
            "задает контрольный слой для надежных многошаговых агентов."
        )
    return (
        "LangGraph дополняет LangChain уровнем графовой orchestration для агентов: помогает "
        "явно моделировать состояние, переходы, циклы и восстановление выполнения, тогда как "
        "обычный LangChain чаще описывает линейные chains и интеграции."
    )


def _format_bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _trim_readme_for_display(readme_text: str) -> str:
    lines = [line.rstrip() for line in readme_text.splitlines() if line.strip()]
    if not lines:
        return "README.md не найден."
    excerpt = "\n".join(lines[:18])
    if len(readme_text) >= MAX_README_CHARS:
        excerpt += "\n\n_Фрагмент README ограничен по длине для безопасности._"
    return excerpt
