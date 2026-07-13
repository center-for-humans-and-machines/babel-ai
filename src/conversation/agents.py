"""Conversation agent protocol and adapters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

from agent import Agent
from conversation.messages import ContextStack
from conversation.scaffolder import ThreeBehaviorScaffolder
from eliza.interventions import build_intervention
from eliza.live_feed import load_topic_bank_topics
from eliza.session import PartnerSession

if TYPE_CHECKING:
    from conversation.agent_config import (
        RuleBasedAgentConfig,
        ScaffolderAgentConfig,
    )


@dataclass
class AgentTurn:
    """Result of one agent generation step."""

    content: str
    llm_nudge: Optional[str] = None
    used_generic_fallback: bool = False
    eliza_branch: Optional[str] = None
    eliza_keyword: Optional[str] = None
    eliza_reassembly: Optional[str] = None
    scaffolder_action: Optional[str] = None
    scaffolder_informative: Optional[bool] = None
    scaffolder_novelty: Optional[float] = None
    scaffolder_continuity: Optional[float] = None
    scaffolder_content_tokens: Optional[int] = None
    scaffolder_meta_detected: Optional[bool] = None
    scaffolder_memory_size: Optional[int] = None
    scaffolder_topic_source_turn: Optional[int] = None


class ConversationAgent(Protocol):
    """Agent that participates in a managed conversation."""

    agent_id: str
    speaker: str

    def generate(self, stack: ContextStack) -> AgentTurn:
        """Produce the next turn from the current context."""
        ...


class LLMConversationAgent:
    """Wraps the legacy LLM Agent for ConversationManager."""

    def __init__(self, agent: Agent, speaker: str):
        self._agent = agent
        self.agent_id = str(agent.id)
        self.speaker = speaker

    def generate(self, stack: ContextStack) -> AgentTurn:
        """Call the LLM with the legacy message format."""
        content = self._agent.generate_response(stack.to_legacy_messages())
        return AgentTurn(content=content)


class RuleBasedConversationAgent:
    """Wraps PartnerSession for ConversationManager."""

    def __init__(
        self,
        config: "RuleBasedAgentConfig",
        speaker: str = "eliza",
    ) -> None:
        self.agent_id = f"rule_based:{config.partner}"
        self.speaker = speaker
        self._config = config
        self._session = PartnerSession(
            intervention=build_intervention(
                config.generic_intervention,
                topic_switch_probability=config.topic_switch_probability,
                feed_sources=config.feed_sources,
            ),
            script_name="doctor",
        )

    def generate(self, stack: ContextStack) -> AgentTurn:
        """Produce the next ELIZA turn from the context stack."""
        turn = self._session.respond(stack.to_legacy_messages())
        return AgentTurn(
            content=turn.text,
            used_generic_fallback=turn.used_generic_fallback,
            eliza_branch=turn.eliza_branch,
            eliza_keyword=turn.eliza_keyword,
            eliza_reassembly=turn.eliza_reassembly,
        )

    def export_state(self) -> dict:
        """Serialize ELIZA session state for checkpoint resume."""
        return self._session.export_state()

    def import_state(self, data: dict) -> None:
        """Restore ELIZA session state from a checkpoint."""
        self._session.import_state(data)


class MirrorConversationAgent:
    """Echoes the latest message from a different speaker."""

    def __init__(self, speaker: str = "mirror") -> None:
        self.agent_id = "mirror"
        self.speaker = speaker

    def generate(self, stack: ContextStack) -> AgentTurn:
        """Return the latest peer message unchanged."""
        for message in reversed(stack.messages):
            if message.speaker != self.speaker:
                return AgentTurn(content=message.content)
        raise ValueError(
            "mirror agent needs a prior message from another speaker"
        )


class ScaffolderConversationAgent:
    """Adapt the three-behavior policy to the conversation protocol."""

    def __init__(
        self,
        config: "ScaffolderAgentConfig",
        speaker: str = "scaffolder",
    ) -> None:
        self.agent_id = "scaffolder"
        self.speaker = speaker
        self._policy = ThreeBehaviorScaffolder(
            load_topic_bank_topics(),
            min_content_tokens=config.min_content_tokens,
            novelty_threshold=config.novelty_threshold,
            continuity_threshold=config.continuity_threshold,
            history_window=config.history_window,
            stuck_turns=config.stuck_turns,
            memory_cooldown=config.memory_cooldown,
            memory_size=config.memory_size,
            similarity_threshold=config.similarity_threshold,
            random_seed=config.random_seed,
        )

    def generate(self, stack: ContextStack) -> AgentTurn:
        """Choose a deterministic scaffold for the latest LLM turn."""
        turn = self._policy.respond(
            stack.messages,
            speaker=self.speaker,
        )
        scores = turn.scores
        return AgentTurn(
            content=turn.content,
            scaffolder_action=turn.action.value,
            scaffolder_informative=scores.informative,
            scaffolder_novelty=scores.novelty,
            scaffolder_continuity=scores.continuity,
            scaffolder_content_tokens=scores.content_tokens,
            scaffolder_meta_detected=scores.meta_detected,
            scaffolder_memory_size=turn.memory_size,
            scaffolder_topic_source_turn=turn.topic_source_turn,
        )

    def export_state(self) -> dict:
        """Serialize policy state for checkpoint resume."""
        return self._policy.export_state()

    def import_state(self, data: dict) -> None:
        """Restore policy state from a checkpoint."""
        self._policy.import_state(data)
