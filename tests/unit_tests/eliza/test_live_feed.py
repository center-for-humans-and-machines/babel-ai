"""Tests for ELIZA live-feed topic providers."""

import io
import json

import pytest

from eliza.interventions import GenericContext, LiveFeedIntervention
from eliza.live_feed import (
    CombinedFeed,
    HackerNewsFeed,
    LiveFeedStub,
    TopicBankFeed,
    build_feed,
    format_topic_switch,
)
from eliza.session import PartnerSession


class _FixedFeed:
    """Deterministic feed for intervention tests."""

    def __init__(self, topic: str | None) -> None:
        self._topic = topic

    def pick_topic(self) -> str | None:
        return self._topic


def test_format_topic_switch():
    text = format_topic_switch("urban gardening")
    assert text == "I came across urban gardening. What comes to mind?"


def test_topic_bank_feed_reads_bundled_topics():
    feed = TopicBankFeed(rng=__import__("random").Random(0))
    topic = feed.pick_topic()
    assert topic
    assert topic in feed._topics


def test_topic_bank_feed_handles_missing_file(tmp_path):
    missing = tmp_path / "missing.json"
    feed = TopicBankFeed(path=missing)
    assert feed.pick_topic() is None


def test_live_feed_stub_uses_topic_bank():
    feed = LiveFeedStub(rng=__import__("random").Random(1))
    assert feed.pick_topic()


def test_build_feed_defaults_to_topic_bank():
    feed = build_feed()
    assert feed.pick_topic()


def test_combined_feed_returns_first_available_topic():
    feeds = [
        TopicBankFeed(rng=__import__("random").Random(0)),
        _FixedFeed(None),
    ]
    combined = CombinedFeed(feeds, rng=__import__("random").Random(0))
    assert combined.pick_topic()


def test_hacker_news_feed_fetches_title():
    top_ids = [101, 102]
    item = {"title": "Sample headline"}

    def opener(url, timeout=None):
        if url.endswith("topstories.json"):
            return io.BytesIO(json.dumps(top_ids).encode())
        return io.BytesIO(json.dumps(item).encode())

    feed = HackerNewsFeed(rng=__import__("random").Random(0), opener=opener)
    assert feed.pick_topic() in {"Sample headline"}


def test_live_feed_intervention_respects_probability():
    context = GenericContext("x", "Default.", [], 0)
    always = LiveFeedIntervention(
        _FixedFeed("sleep and productivity"),
        topic_switch_probability=1.0,
        rng=lambda: 0.0,
    )
    never = LiveFeedIntervention(
        _FixedFeed("sleep and productivity"),
        topic_switch_probability=0.0,
        rng=lambda: 0.0,
    )
    switched = always.on_generic(context)
    assert switched.topic_switched
    assert "sleep and productivity" in (switched.partner_text or "")
    assert never.on_generic(context).partner_text is None


def test_live_feed_intervention_falls_back_without_topic():
    context = GenericContext("x", "Default.", [], 0)
    intervention = LiveFeedIntervention(
        _FixedFeed(None),
        topic_switch_probability=1.0,
        rng=lambda: 0.0,
    )
    result = intervention.on_generic(context)
    assert result.partner_text is None
    assert not result.topic_switched


def test_partner_session_topic_switch_branch_label():
    intervention = LiveFeedIntervention(
        _FixedFeed("community radio stations"),
        topic_switch_probability=1.0,
        rng=lambda: 0.0,
    )
    turn = PartnerSession(intervention=intervention).respond(
        [{"role": "user", "content": "xyzzy"}]
    )
    assert turn.eliza_branch == "generic:topic_switch"
    assert "community radio stations" in turn.text
    assert turn.used_generic_fallback


def test_partner_session_live_feed_passthrough_branch():
    intervention = LiveFeedIntervention(
        _FixedFeed("ignored topic"),
        topic_switch_probability=0.0,
        rng=lambda: 0.0,
    )
    turn = PartnerSession(intervention=intervention).respond(
        [{"role": "user", "content": "xyzzy"}]
    )
    assert turn.eliza_branch == "generic:$"
    assert "ignored topic" not in turn.text


@pytest.mark.parametrize(
    "draw,expected_switch",
    [(0.49, True), (0.51, False)],
)
def test_live_feed_probability_threshold(draw, expected_switch):
    context = GenericContext("x", "Default.", [], 0)
    intervention = LiveFeedIntervention(
        _FixedFeed("fermentation in home kitchens"),
        topic_switch_probability=0.5,
        rng=lambda: draw,
    )
    result = intervention.on_generic(context)
    assert result.topic_switched is expected_switch


def test_build_feed_skips_unknown_sources(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    feed = build_feed(["unknown_source"])
    assert "Unknown feed source" in caplog.text
    assert feed.pick_topic()


def test_topic_bank_feed_handles_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    feed = TopicBankFeed(path=bad)
    assert feed.pick_topic() is None


def test_hacker_news_feed_handles_network_errors():
    def opener(url, timeout=None):
        raise OSError("offline")

    feed = HackerNewsFeed(opener=opener)
    assert feed.pick_topic() is None


def test_combined_feed_returns_none_when_all_empty():
    combined = CombinedFeed([_FixedFeed(None), _FixedFeed(None)])
    assert combined.pick_topic() is None

