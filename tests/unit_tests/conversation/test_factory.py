"""Tests for conversation agent factory mapping."""

from conversation.agent_config import LLMAgentConfig
from conversation.agents import LLMConversationAgent
from conversation.factory import build_agent
from models.configs import AgentConfig as LegacyAgentConfig


def test_llm_agent_config_defaults_match_legacy():
    cfg = LLMAgentConfig(provider="openai", model="gpt-4-0125-preview")
    assert cfg.temperature == 1.0
    assert cfg.max_tokens is None
    assert cfg.frequency_penalty == 0.0
    assert cfg.presence_penalty == 0.0
    assert cfg.top_p == 1.0
    assert cfg.forgetting is None


def test_build_llm_agent_maps_generation_params():
    cfg = LLMAgentConfig(
        provider="openai",
        model="gpt-4-0125-preview",
        system_prompt="Test prompt.",
        temperature=0.7,
        max_tokens=150,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        top_p=0.9,
        forgetting=3,
    )
    agent = build_agent(cfg, speaker="agent_0")
    assert isinstance(agent, LLMConversationAgent)
    legacy = agent._agent.config
    assert isinstance(legacy, LegacyAgentConfig)
    assert legacy.system_prompt == "Test prompt."
    assert legacy.temperature == 0.7
    assert legacy.max_tokens == 150
    assert legacy.frequency_penalty == 0.1
    assert legacy.presence_penalty == 0.2
    assert legacy.top_p == 0.9
    assert legacy.forgetting == 3
