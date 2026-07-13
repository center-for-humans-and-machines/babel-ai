"""Tests for the deterministic three-behavior scaffolder."""

import pytest

from conversation.agent_config import ScaffolderAgentConfig
from conversation.agents import ScaffolderConversationAgent
from conversation.factory import build_agent
from conversation.messages import (
    ContextStack,
    ConversationMessage,
    MessageSource,
)
from conversation.scaffolder import (
    NoveltyNudgeKind,
    ScaffolderAction,
    ThreeBehaviorScaffolder,
    content_tokens,
    representative_sentence,
)


def _message(index: int, content: str, speaker: str = "llm"):
    return ConversationMessage(
        turn_index=index,
        role="assistant",
        speaker=speaker,
        content=content,
        source=MessageSource.AGENT,
    )


def _policy(**kwargs) -> ThreeBehaviorScaffolder:
    defaults = {
        "min_content_tokens": 3,
        "novelty_threshold": 0.20,
        "continuity_threshold": 0.05,
        "stuck_turns": 1,
        "memory_cooldown": 0,
        "random_seed": 7,
    }
    defaults.update(kwargs)
    return ThreeBehaviorScaffolder(
        ["ocean currents", "public libraries", "fermentation"],
        **defaults,
    )


def test_factory_builds_scaffolder_agent():
    config = ScaffolderAgentConfig()
    agent = build_agent(config, speaker="scaffolder")
    assert isinstance(agent, ScaffolderConversationAgent)
    assert agent.speaker == "scaffolder"
    assert config.history_window == 8


def test_informative_turn_is_remembered_and_encouraged():
    policy = _policy()
    turn = policy.respond(
        [_message(0, "Ocean currents transport heat across the planet.")],
        speaker="scaffolder",
    )
    state = policy.export_state()
    assert turn.action is ScaffolderAction.THRIVE_PROTECTION
    assert turn.scores.informative
    assert len(state["memory"]) == 1


def test_meta_turn_is_not_informative_even_when_long():
    policy = _policy()
    scores = policy.score(
        "I am here to help with any questions, so feel free to ask me."
    )
    assert scores.meta_detected
    assert not scores.informative


def test_first_noninformative_turn_gets_stuck_hysteresis():
    policy = _policy(stuck_turns=2)
    turn = policy.respond([_message(0, "Okay.")], speaker="scaffolder")
    assert turn.action is ScaffolderAction.THRIVE_PROTECTION
    assert not turn.scores.informative


def test_stuck_turn_resurfaces_and_removes_oldest_memory():
    policy = _policy()
    messages = [
        _message(0, "Ocean currents transport heat across the planet."),
        _message(1, "Forests support diverse plants and animals."),
        _message(2, "Okay."),
    ]
    turn = policy.respond(messages, speaker="scaffolder")
    assert turn.action is ScaffolderAction.MEMORY_RESURFACE
    assert turn.topic_source_turn == 0
    assert turn.memory_size == 1


def test_empty_memory_injects_deterministic_topic():
    first = _policy()
    second = _policy()
    message = [_message(0, "Okay.")]
    left = first.respond(message, speaker="scaffolder")
    right = second.respond(message, speaker="scaffolder")
    assert left.action is ScaffolderAction.TOPIC_INJECTION
    assert left.content == right.content


def test_topic_deck_does_not_repeat_before_exhaustion():
    policy = _policy()
    outputs = []
    messages = []
    for index in range(3):
        messages.append(_message(index * 2, "Okay."))
        turn = policy.respond(messages, speaker="scaffolder")
        outputs.append(turn.content)
        messages.append(_message(index * 2 + 1, turn.content, "scaffolder"))
    assert len(set(outputs)) == 3


def test_checkpoint_state_restores_exact_next_decision():
    original = _policy()
    messages = [_message(0, "Okay.")]
    first = original.respond(messages, speaker="scaffolder")
    messages.append(_message(1, first.content, "scaffolder"))
    restored = _policy()
    restored.import_state(original.export_state())
    messages.append(_message(2, "Still okay."))
    expected = original.respond(messages, speaker="scaffolder")
    actual = restored.respond(messages, speaker="scaffolder")
    assert actual == expected


def test_cooldown_skips_recent_memory_and_injects():
    policy = _policy(memory_cooldown=2)
    messages = [
        _message(0, "Ocean currents transport heat across the planet."),
        _message(1, "Okay."),
    ]
    turn = policy.respond(messages, speaker="scaffolder")
    assert turn.action is ScaffolderAction.TOPIC_INJECTION


def test_similar_topics_are_deduplicated():
    policy = _policy(similarity_threshold=0.50, stuck_turns=2)
    first = policy.respond(
        [_message(0, "Ocean currents transport heat across the planet.")],
        speaker="scaffolder",
    )
    messages = [
        _message(0, "Ocean currents transport heat across the planet."),
        _message(1, first.content, "scaffolder"),
        _message(2, "Ocean currents move heat around the planet."),
    ]
    policy.respond(messages, speaker="scaffolder")
    assert len(policy.export_state()["memory"]) == 1


def test_chinese_content_uses_character_tokens():
    tokens = content_tokens("山重水复之后可以继续进行成语接龙")
    assert len(tokens) >= 8


def test_replay_moves_from_thriving_through_stuck_to_memory():
    policy = _policy(stuck_turns=2)
    messages = [
        _message(
            0,
            "Recursive descent parses expressions with grammar functions.",
        )
    ]
    thriving = policy.respond(messages, speaker="scaffolder")
    messages.extend(
        [
            _message(1, thriving.content, "scaffolder"),
            _message(
                2,
                "Recursive descent parses expressions with grammar functions.",
            ),
        ]
    )
    probation = policy.respond(messages, speaker="scaffolder")
    messages.extend(
        [
            _message(3, probation.content, "scaffolder"),
            _message(
                4,
                "I am here to help with questions, so feel free to ask.",
            ),
        ]
    )
    resurfaced = policy.respond(messages, speaker="scaffolder")
    assert thriving.scores.informative
    assert probation.action is ScaffolderAction.THRIVE_PROTECTION
    assert not probation.scores.informative
    assert resurfaced.action is ScaffolderAction.MEMORY_RESURFACE
    assert resurfaced.topic_source_turn == 0


def test_policy_rejects_missing_or_reprocessed_peer_turns():
    with pytest.raises(ValueError, match="at least one topic"):
        ThreeBehaviorScaffolder([])
    with pytest.raises(ValueError, match="between 0 and 1"):
        ThreeBehaviorScaffolder(["topic"], novelty_nudge_rate=1.1)
    policy = _policy()
    with pytest.raises(ValueError, match="prior peer"):
        policy.respond([], speaker="scaffolder")
    messages = [_message(0, "Okay.")]
    policy.respond(messages, speaker="scaffolder")
    with pytest.raises(ValueError, match="new peer"):
        policy.respond(messages, speaker="scaffolder")


def test_memory_state_and_topic_cycle_cover_exhaustion_paths():
    policy = _policy(memory_size=1, continuity_threshold=0.0)
    first = policy.respond(
        [_message(0, "Ocean currents transport heat across the planet.")],
        speaker="scaffolder",
    )
    messages = [
        _message(0, "Ocean currents transport heat across the planet."),
        _message(1, first.content, "scaffolder"),
        _message(2, "Libraries preserve books and support local learning."),
    ]
    policy.respond(messages, speaker="scaffolder")
    state = policy.export_state()
    assert len(state["memory"]) == 1
    restored = _policy(memory_size=1, continuity_threshold=0.0)
    restored.import_state(state)
    assert restored.export_state() == state

    injector = _policy()
    seen = []
    injection_messages = []
    for index in range(4):
        injection_messages.append(_message(index * 2, "Okay."))
        turn = injector.respond(
            injection_messages,
            speaker="scaffolder",
        )
        seen.append(turn.content)
        injection_messages.append(
            _message(index * 2 + 1, turn.content, "scaffolder")
        )
    assert len(set(seen[:3])) == 3
    assert seen[3]


def test_representative_sentence_handles_empty_and_long_text():
    assert representative_sentence("", []) == ""
    summary = representative_sentence("x" * 200, [], max_length=20)
    assert summary == ("x" * 19) + "…"


def test_novelty_nudge_runs_once_per_five_informative_turns():
    policy = _policy(
        novelty_nudge_rate=0.20,
        novelty_threshold=0.0,
        continuity_threshold=0.0,
    )
    messages = []
    nudges = []
    for index in range(10):
        messages.append(
            _message(
                index * 2,
                "Ocean currents shape climate systems in coastal regions.",
            )
        )
        turn = policy.respond(messages, speaker="scaffolder")
        nudges.append(turn.novelty_nudge_kind)
        messages.append(_message(index * 2 + 1, turn.content, "scaffolder"))
    assert [index for index, kind in enumerate(nudges) if kind] == [4, 9]


def test_novelty_nudge_rotates_all_requested_strategies():
    policy = _policy(
        novelty_nudge_rate=1.0,
        novelty_threshold=0.0,
        continuity_threshold=0.0,
    )
    messages = []
    turns = []
    for index in range(6):
        messages.append(
            _message(
                index * 2,
                "Ocean ocean currents currents climate climate.",
            )
        )
        turn = policy.respond(messages, speaker="scaffolder")
        turns.append(turn)
        messages.append(_message(index * 2 + 1, turn.content, "scaffolder"))
    assert [turn.novelty_nudge_kind for turn in turns[:4]] == [
        NoveltyNudgeKind.CONNECTED_NOVELTY,
        NoveltyNudgeKind.SINGLE_CONCEPT,
        NoveltyNudgeKind.COMBINED_CONCEPTS,
        NoveltyNudgeKind.MODEL_TOPIC_SWITCH,
    ]
    assert '"climate"' in turns[1].content
    assert '"climate"' in turns[2].content
    assert '"currents"' in turns[2].content
    assert "leave" in turns[5].content


def test_agent_turn_exposes_novelty_nudge_trace():
    agent = ScaffolderConversationAgent(
        ScaffolderAgentConfig(
            min_content_tokens=3,
            novelty_threshold=0.0,
            continuity_threshold=0.0,
            novelty_nudge_rate=1.0,
        )
    )
    turn = agent.generate(
        ContextStack(
            messages=[
                _message(
                    0,
                    "Ocean currents shape climate systems globally.",
                )
            ]
        )
    )
    assert turn.scaffolder_action == "thrive_protection"
    assert turn.scaffolder_novelty_nudge_kind == "connected_novelty"
