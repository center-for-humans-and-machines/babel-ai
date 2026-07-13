"""Build ConversationAgent instances from canonical config."""

from typing import List

from agent import Agent
from api.enums import APIModels, Provider
from conversation.agent_config import (
    AgentConfig,
    AgentType,
    LLMAgentConfig,
    MirrorAgentConfig,
    RuleBasedAgentConfig,
    ScaffolderAgentConfig,
)
from conversation.agents import (
    ConversationAgent,
    LLMConversationAgent,
    MirrorConversationAgent,
    RuleBasedConversationAgent,
    ScaffolderConversationAgent,
)
from models.configs import AgentConfig as LegacyAgentConfig


def build_conversation_agents(
    configs: List[AgentConfig],
) -> List[ConversationAgent]:
    """Instantiate agents from the unified ``agents`` config list."""
    return [
        build_agent(config, speaker=_default_speaker(config, index))
        for index, config in enumerate(configs)
    ]


def build_agent(config: AgentConfig, *, speaker: str) -> ConversationAgent:
    """Instantiate one conversation agent from config."""
    if config.type is AgentType.LLM:
        return LLMConversationAgent(
            agent=Agent(_legacy_agent_config(config)),
            speaker=speaker,
        )
    if config.type is AgentType.RULE_BASED:
        return RuleBasedConversationAgent(config=config, speaker=speaker)
    if config.type is AgentType.MIRROR:
        return MirrorConversationAgent(speaker=speaker)
    if config.type is AgentType.SCAFFOLDER:
        return ScaffolderConversationAgent(config=config, speaker=speaker)
    raise ValueError(f"unknown agent type: {config.type}")


def _default_speaker(config: AgentConfig, index: int) -> str:
    """Pick a stable speaker label for transcript rows."""
    if isinstance(config, RuleBasedAgentConfig):
        return config.partner
    if isinstance(config, MirrorAgentConfig):
        return "mirror"
    if isinstance(config, ScaffolderAgentConfig):
        return "scaffolder"
    return f"agent_{index}"


def _legacy_agent_config(config: LLMAgentConfig) -> LegacyAgentConfig:
    """Map canonical LLM config onto the legacy AgentConfig model."""
    provider = Provider(config.provider)
    model_enum = provider.get_model_enum()
    model = _resolve_model(model_enum, config.model)
    return LegacyAgentConfig(
        provider=provider,
        model=model,
        system_prompt=config.system_prompt,
    )


def _resolve_model(model_enum: type, model_name: str) -> APIModels:
    """Resolve a model string to the provider-specific enum member."""
    for member in model_enum:
        if member.value == model_name:
            return member
    raise ValueError(
        f"model {model_name!r} is not valid for {model_enum.__name__}"
    )
