"""Periodic checkpoint save and restore."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from conversation.agent_state import snapshot_agents
from conversation.messages import ContextStack, ConversationMessage
from conversation.settings import ConversationSettings
from conversation.status import ConversationStatus
from conversation.turn_taking import TurnTakingState
from models.configs import FetcherConfig
from models.metrics import AgentMetric, AnalysisResult, FetcherMetric, Metric

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Full recoverable manager state."""

    run_id: str
    status: ConversationStatus = ConversationStatus.PENDING
    turn_taking_state: TurnTakingState = field(default_factory=TurnTakingState)
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
        agents: Optional[List[Any]] = None,
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
            "agent_states": snapshot_agents(agents or []),
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

    @staticmethod
    def restore_settings(data: Dict[str, Any]) -> ConversationSettings:
        """Rebuild conversation settings saved in the checkpoint."""
        return ConversationSettings.model_validate(data["settings"])

    @staticmethod
    def restore_metrics(data: Dict[str, Any]) -> List[Metric]:
        """Rebuild metric rows from checkpoint dictionaries."""
        metrics: List[Metric] = []
        for raw in data.get("metrics", []):
            analysis = _restore_analysis(raw.get("analysis"))
            if raw.get("fetcher_config") is not None:
                fetcher_cfg = raw["fetcher_config"]
                if isinstance(fetcher_cfg, dict):
                    fetcher_cfg = FetcherConfig.model_validate(fetcher_cfg)
                metrics.append(
                    FetcherMetric(
                        iteration=raw["iteration"],
                        timestamp=_parse_timestamp(raw["timestamp"]),
                        role=raw["role"],
                        content=raw["content"],
                        analysis=analysis,
                        fetcher_config=fetcher_cfg,
                    )
                )
                continue
            metrics.append(
                AgentMetric(
                    iteration=raw["iteration"],
                    timestamp=_parse_timestamp(raw["timestamp"]),
                    role=raw["role"],
                    content=raw["content"],
                    analysis=analysis,
                    agent_id=raw.get("agent_id", "unknown"),
                    agent_config=raw.get("agent_config"),
                    speaker=raw.get("speaker"),
                    used_generic_fallback=raw.get("used_generic_fallback"),
                )
            )
        return metrics


def _restore_analysis(raw: Any) -> Optional[AnalysisResult]:
    """Convert a stored analysis blob back into ``AnalysisResult``."""
    if raw is None:
        return None
    if isinstance(raw, AnalysisResult):
        return raw
    if isinstance(raw, dict):
        fields = AnalysisResult.model_fields.keys()
        payload = {key: raw.get(key) for key in fields if key in raw}
        return AnalysisResult(**payload)
    return None


def _parse_timestamp(value: Any) -> datetime:
    """Parse ISO timestamps stored in checkpoint JSON."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now()
