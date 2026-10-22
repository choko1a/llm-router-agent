from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool:
    name = "base"
    description = "Base tool interface."

    def can_handle(self, query: str) -> bool:
        raise NotImplementedError

    def run(self, query: str) -> ToolResult:
        raise NotImplementedError
