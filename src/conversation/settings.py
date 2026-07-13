"""Configuration for ConversationManager."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TurnTakingMethod(str, Enum):
    """How the manager picks the next speaker."""

    ROUND_ROBIN = "round_robin"
    FIXED_ORDER = "fixed_order"


class AnalysisPolicy(str, Enum):
    """When drift/collapse metrics are computed."""

    PER_TURN = "per_turn"
    ON_CHECKPOINT = "on_checkpoint"
    AT_END = "at_end"


class ConversationSettings(BaseModel):
    """Runtime settings for ConversationManager."""

    turn_taking_method: TurnTakingMethod = TurnTakingMethod.ROUND_ROBIN
    fixed_order: Optional[List[int]] = Field(
        default=None,
        description="Agent indices for FIXED_ORDER, e.g. [0, 1, 0, 1]",
    )
    analysis_policy: AnalysisPolicy = AnalysisPolicy.ON_CHECKPOINT
    checkpoint_enabled: bool = True
    checkpoint_interval_seconds: int = Field(default=120, ge=1)
    max_iterations: int = Field(default=100, ge=1)
    max_total_characters: int = Field(default=1_000_000, ge=1)
