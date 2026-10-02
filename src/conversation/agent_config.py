"""Canonical multi-agent configuration (pillar E)."""

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from eliza.interventions import GenericInterventionMode


class AgentType(str, Enum):
    """Conversation participant kinds."""

    LLM = "llm"
    RULE_BASED = "rule_based"
    MIRROR = "mirror"
    SCAFFOLDER = "scaffolder"
    RAG_SCAFFOLDER = "rag_scaffolder"


class LLMAgentConfig(BaseModel):
    """LLM-backed conversation agent."""

    type: Literal[AgentType.LLM] = AgentType.LLM
    provider: str
    model: str
    system_prompt: str = "You are having a conversation."
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    forgetting: Optional[int] = Field(
        default=None,
        description="Number of past messages to use (None = all)",
        ge=1,
    )


class RuleBasedAgentConfig(BaseModel):
    """Deterministic rule partner (ELIZA)."""

    type: Literal[AgentType.RULE_BASED] = AgentType.RULE_BASED
    partner: Literal["eliza"] = "eliza"
    generic_intervention: GenericInterventionMode = (
        GenericInterventionMode.PASSTHROUGH
    )
    topic_switch_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    feed_sources: list[str] = Field(default_factory=lambda: ["topic_bank"])


class MirrorAgentConfig(BaseModel):
    """Echo previous speaker (ladder level 1)."""

    type: Literal[AgentType.MIRROR] = AgentType.MIRROR


class ScaffolderAgentConfig(BaseModel):
    """Deterministic three-behavior scaffolding agent."""

    type: Literal[AgentType.SCAFFOLDER] = AgentType.SCAFFOLDER
    min_content_tokens: int = Field(default=8, ge=1)
    novelty_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    continuity_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    history_window: int = Field(default=8, ge=1)
    stuck_turns: int = Field(default=2, ge=1)
    memory_cooldown: int = Field(default=2, ge=0)
    memory_size: int = Field(default=20, ge=1)
    similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    novelty_nudge_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    topic_source: Literal["topic_bank"] = "topic_bank"
    random_seed: int = 0


class RagScaffolderAgentConfig(ScaffolderAgentConfig):
    """Web-search-grounded scaffolder sharing the three-behavior policy."""

    type: Literal[AgentType.RAG_SCAFFOLDER] = AgentType.RAG_SCAFFOLDER
    word_model: str = "glove-wiki-gigaword-100"
    num_words: int = Field(default=5, ge=1)
    search_backend: Literal["duckduckgo", "wikipedia", "auto"] = "duckduckgo"
    search_results: int = Field(default=5, ge=1)
    fetch_page: bool = True
    max_source_chars: int = Field(default=4000, ge=100)
    search_timeout: float = Field(default=8.0, gt=0.0)
    context_turns: int = Field(default=6, ge=1)
    nudge_system_prompt: str = (
        "You are a research assistant embedded in a conversation. You help "
        "a language model stay novel and grounded. Always answer with a "
        "single short instruction (one or two sentences) that the model "
        "should follow."
    )
    nudge_max_tokens: Optional[int] = Field(default=300, ge=1)


AgentConfig = Annotated[
    Union[
        LLMAgentConfig,
        RuleBasedAgentConfig,
        MirrorAgentConfig,
        ScaffolderAgentConfig,
        RagScaffolderAgentConfig,
    ],
    Field(discriminator="type"),
]
