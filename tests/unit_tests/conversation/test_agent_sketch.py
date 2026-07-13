"""Tests for conversation agent adapters."""

from conversation.agent_config import MirrorAgentConfig, RuleBasedAgentConfig
from conversation.agents import (
    MirrorConversationAgent,
    RuleBasedConversationAgent,
)
from conversation.factory import build_agent
from conversation.messages import (
    ContextStack,
    ConversationMessage,
    MessageSource,
)
from eliza.interventions import GenericInterventionMode


def _stack(*messages: tuple[str, str, str]) -> ContextStack:
    return ContextStack(
        messages=[
            ConversationMessage(
                turn_index=index,
                role=role,
                speaker=speaker,
                content=content,
                source=MessageSource.SEED,
            )
            for index, (role, speaker, content) in enumerate(messages)
        ]
    )


def test_rule_based_config_defaults():
    cfg = RuleBasedAgentConfig()
    assert cfg.partner == "eliza"
    assert cfg.generic_intervention is GenericInterventionMode.PASSTHROUGH


def test_build_rule_based_agent():
    agent = build_agent(RuleBasedAgentConfig(), speaker="eliza")
    assert isinstance(agent, RuleBasedConversationAgent)
    assert agent.speaker == "eliza"


def test_rule_based_agent_generates_eliza_reply():
    agent = RuleBasedConversationAgent(RuleBasedAgentConfig())
    turn = agent.generate(_stack(("user", "seed", "Men are all alike.")))
    assert turn.content == "In what way?"
    assert not turn.used_generic_fallback
    assert turn.eliza_branch == "keyword:like"


def test_build_mirror_agent():
    agent = build_agent(MirrorAgentConfig(), speaker="echo")
    assert isinstance(agent, MirrorConversationAgent)
    assert agent.speaker == "echo"


def test_mirror_generates_latest_peer_message():
    agent = MirrorConversationAgent(speaker="echo")
    turn = agent.generate(
        _stack(
            ("user", "seed", "first"),
            ("assistant", "echo", "ignored"),
            ("user", "other", "latest"),
        )
    )
    assert turn.content == "latest"
