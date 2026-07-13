"""Tests for the flat-layout ELIZA partner."""

import logging

import pytest

from eliza.interventions import (
    GenericContext,
    GenericInterventionMode,
    LiveFeedIntervention,
    LLMNudgeIntervention,
    PassthroughIntervention,
    build_intervention,
)
from eliza.live_feed import LiveFeedStub
from eliza.session import PartnerSession


def test_paper_example_and_prefixes():
    turn = PartnerSession().respond(
        [{"role": "user", "content": "Men are all alike."}]
    )
    assert turn.text == "In what way?"
    assert "Eliza:" not in turn.text
    assert "You:" not in turn.text


def test_paper_examples_keep_rule_state():
    session = PartnerSession()
    first = session.respond(
        [{"role": "user", "content": "Men are all alike."}]
    )
    second = session.respond(
        [
            {"role": "user", "content": "Men are all alike."},
            {"role": "assistant", "content": first.text},
            {
                "role": "user",
                "content": "They're always bugging us about something.",
            },
        ]
    )
    assert second.text == "Can you think of a specific example?"


def test_memory_stack_persists_between_turns():
    session = PartnerSession()
    session.respond([{"role": "user", "content": "My mother is kind."}])
    expected = PartnerSession.strip_prefixes(session._memory_stack[-1])
    turn = session.respond([{"role": "user", "content": "xyzzy"}])
    assert turn.text == expected


def test_passthrough_generic_response_needs_no_network():
    turn = PartnerSession(PassthroughIntervention()).respond(
        [{"role": "user", "content": "xyzzy"}]
    )
    assert turn.text == "Please go on."
    assert turn.used_generic_fallback


def test_sessions_deepcopy_their_rule_state():
    first = PartnerSession()
    second = PartnerSession()
    assert (
        first.respond([{"role": "user", "content": "xyzzy"}]).text
        == "Please go on."
    )
    assert (
        second.respond([{"role": "user", "content": "xyzzy"}]).text
        == "Please go on."
    )
    assert (
        first.respond([{"role": "user", "content": "xyzzy"}]).text
        == "I am not sure I understand you fully."
    )


def test_llm_nudge_rotates_hints():
    intervention = LLMNudgeIntervention(["First hint.", "Second hint."])
    context = GenericContext("x", "Default.", [], 0)
    assert intervention.on_generic(context).partner_text == "First hint."
    assert intervention.on_generic(context).partner_text == "Second hint."
    assert intervention.on_generic(context).partner_text == "First hint."


def test_intervention_factory_and_empty_nudge():
    context = GenericContext("x", "Default.", [], 0)
    assert isinstance(
        build_intervention(GenericInterventionMode.PASSTHROUGH),
        PassthroughIntervention,
    )
    assert LLMNudgeIntervention().on_generic(context).partner_text is None
    with pytest.raises(NotImplementedError):
        build_intervention(GenericInterventionMode.CUSTOM)
    with pytest.raises(ValueError, match="non-empty user message"):
        PartnerSession().respond([])


def test_live_feed_stubs_warn_and_passthrough(caplog):
    caplog.set_level(logging.WARNING)
    context = GenericContext("x", "Default.", [], 0)
    result = LiveFeedIntervention(LiveFeedStub()).on_generic(context)
    assert result.partner_text is None
    assert "unavailable" in caplog.text
    assert LiveFeedStub().fetch_headline() is None
    assert "not implemented" in caplog.text
