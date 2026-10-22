from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentToolCall:
    id: str
    name: str
    arguments: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class AgentTurn:
    content: Optional[str]
    tool_calls: List[AgentToolCall]
    provider: str
    model: str
