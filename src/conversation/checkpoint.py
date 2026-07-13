"""Periodic checkpoint save and restore."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from conversation.messages import ConversationMessage, ContextStack
from conversation.settings import AnalysisPolicy
from conversation.status import ConversationStatus
from conversation.turn_taking import TurnTakingState
from models.metrics import Metric

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Full recoverable manager state."""

    run_id: str
    status: ConversationStatus = ConversationStatus.PENDING
    turn_taking_state: TurnTakingState = field(
        default_factory=TurnTakingState
    )
    pending_llm_nudge: Optional[str] = None
    agent_turn_count: int = 0
    last_checkpoint_at: Optional[datetime] = None


class CheckpointWriter:
    """Atomic checkpoint files under a run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = run_dir / "checkpoint.json"

    def save(
        self,
        state: ConversationState,
        stack: ContextStack,
        metrics: List[Metric],
        turn_taking_data: dict,
        settings: dict,
    ) -> None:
        """Write checkpoint atomically."""
        payload = {
            "run_id": state.run_id,
            "status": state.status.value,
            "messages": [m.to_dict() for m in stack.messages],
            "metrics": [m.to_dict() for m in metrics],
            "turn_taking": turn_taking_data,
            "turn_taking_state": {
                "next_agent_index": state.turn_taking_state.next_agent_index,
                "fixed_order_cursor": (
                    state.turn_taking_state.fixed_order_cursor
                ),
            },
            "pending_llm_nudge": state.pending_llm_nudge,
            "agent_turn_count": state.agent_turn_count,
            "settings": settings,
            "saved_at": datetime.now().isoformat(),
        }
        tmp = self.checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.checkpoint_path)
        state.last_checkpoint_at = datetime.now()
        logger.info("Checkpoint saved to %s", self.checkpoint_path)

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load checkpoint payload."""
        return json.loads(path.read_text())

    @staticmethod
    def restore_stack(data: Dict[str, Any]) -> ContextStack:
        """Rebuild context stack from checkpoint."""
        stack = ContextStack()
        for raw in data["messages"]:
            stack.append(ConversationMessage.from_dict(raw))
        return stack

    @staticmethod
    def restore_state(data: Dict[str, Any]) -> ConversationState:
        """Rebuild conversation state from checkpoint."""
        ts = data.get("turn_taking_state", {})
        return ConversationState(
            run_id=data["run_id"],
            status=ConversationStatus(data["status"]),
            turn_taking_state=TurnTakingState(
                next_agent_index=ts.get("next_agent_index", 0),
                fixed_order_cursor=ts.get("fixed_order_cursor", 0),
            ),
            pending_llm_nudge=data.get("pending_llm_nudge"),
            agent_turn_count=data.get("agent_turn_count", 0),
        )
