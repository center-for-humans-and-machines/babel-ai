"""Canonical multi-agent configuration (pillar E)."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from eliza.interventions import GenericInterventionMode


class AgentType(str, Enum):
    """Conversation participant kinds."""

    LLM = "llm"
    RULE_BASED = "rule_based"
    MIRROR = "mirror"


class LLMAgentConfig(BaseModel):
    """LLM-backed conversation agent."""

    type: Literal[AgentType.LLM] = AgentType.LLM
    provider: str
    model: str
    system_prompt: str = "You are having a conversation."


class RuleBasedAgentConfig(BaseModel):
    """Deterministic rule partner (ELIZA)."""

    type: Literal[AgentType.RULE_BASED] = AgentType.RULE_BASED
    partner: Literal["eliza"] = "eliza"
    generic_intervention: GenericInterventionMode = (
        GenericInterventionMode.PASSTHROUGH
    )


class MirrorAgentConfig(BaseModel):
    """Echo previous speaker (ladder level 1)."""

    type: Literal[AgentType.MIRROR] = AgentType.MIRROR


AgentConfig = Annotated[
    Union[LLMAgentConfig, RuleBasedAgentConfig, MirrorAgentConfig],
    Field(discriminator="type"),
]
