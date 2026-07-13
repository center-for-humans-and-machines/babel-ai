"""Context stack and message records."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List


class MessageSource(str, Enum):
    """Origin of a context entry."""

    SEED = "seed"
    AGENT = "agent"


@dataclass
class ConversationMessage:
    """One entry in the managed context stack."""

    turn_index: int
    role: str
    speaker: str
    content: str
    source: MessageSource
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialize for checkpoint JSON."""
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "speaker": self.speaker,
            "content": self.content,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        """Restore from checkpoint JSON."""
        return cls(
            turn_index=data["turn_index"],
            role=data["role"],
            speaker=data["speaker"],
            content=data["content"],
            source=MessageSource(data["source"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class ContextStack:
    """Ordered conversation history owned by the manager."""

    messages: List[ConversationMessage] = field(default_factory=list)

    def append(self, message: ConversationMessage) -> None:
        """Add a message to the stack."""
        self.messages.append(message)

    def total_characters(self) -> int:
        """Sum of content lengths."""
        return sum(len(m.content) for m in self.messages)

    def to_legacy_messages(self) -> List[Dict[str, str]]:
        """Format for existing Agent.generate_response."""
        return [
            {"role": m.role, "content": m.content} for m in self.messages
        ]

    def content_prefix(self) -> List[str]:
        """Text contents in order (for analyzer)."""
        return [m.content for m in self.messages]
