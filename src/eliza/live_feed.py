"""Topic feeds for ELIZA generic-response topic switches."""

from __future__ import annotations

import json
import logging
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

_TOPIC_BANK_PATH = Path(__file__).parent / "topic_bank.json"
_TOPIC_SWITCH_TEMPLATE = "I came across {topic}. What comes to mind?"
_HN_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
_HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"


class LiveFeedProvider(Protocol):
    """Supply topics for generic-response switches."""

    def pick_topic(self) -> str | None:
        """Return one topic or no result."""
        ...


def format_topic_switch(topic: str) -> str:
    """Format a feed topic as an ELIZA partner line."""
    return _TOPIC_SWITCH_TEMPLATE.format(topic=topic)


class TopicBankFeed:
    """Read topics from a local JSON bank."""

    def __init__(
        self,
        path: Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._path = path or _TOPIC_BANK_PATH
        self._rng = rng or random.Random()
        self._topics = self._load_topics()

    def _load_topics(self) -> list[str]:
        return load_topic_bank_topics(self._path)

    def pick_topic(self) -> str | None:
        """Return a random topic from the bank."""
        if not self._topics:
            return None
        return self._rng.choice(self._topics)


def load_topic_bank_topics(path: Path | None = None) -> list[str]:
    """Load the bundled topic bank as a deterministic list."""
    resolved = path or _TOPIC_BANK_PATH
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Topic bank unavailable at %s: %s", resolved, exc)
        return []
    topics = payload.get("topics", [])
    return [str(topic).strip() for topic in topics if str(topic).strip()]


class HackerNewsFeed:
    """Fetch a random headline from Hacker News top stories."""

    def __init__(
        self,
        timeout: float = 5.0,
        rng: random.Random | None = None,
        opener: Callable[..., object] | None = None,
    ) -> None:
        self._timeout = timeout
        self._rng = rng or random.Random()
        self._open = opener or urllib.request.urlopen

    def pick_topic(self) -> str | None:
        """Return one top-story title or no result."""
        try:
            with self._open(_HN_TOP_URL, timeout=self._timeout) as response:
                story_ids = json.loads(response.read())
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            TimeoutError,
        ) as exc:
            logger.warning("Hacker News top stories unavailable: %s", exc)
            return None
        if not story_ids:
            return None
        sample = story_ids[:20]
        item_id = self._rng.choice(sample)
        item_url = _HN_ITEM_URL.format(item_id=item_id)
        try:
            with self._open(item_url, timeout=self._timeout) as response:
                item = json.loads(response.read())
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            TimeoutError,
        ) as exc:
            logger.warning("Hacker News item %s unavailable: %s", item_id, exc)
            return None
        title = str(item.get("title", "")).strip()
        return title or None


class CombinedFeed:
    """Try feeds in order until one returns a topic."""

    def __init__(
        self,
        feeds: list[LiveFeedProvider],
        rng: random.Random | None = None,
    ) -> None:
        self._feeds = feeds
        self._rng = rng or random.Random()

    def pick_topic(self) -> str | None:
        """Return the first topic found across shuffled feeds."""
        order = list(self._feeds)
        self._rng.shuffle(order)
        for feed in order:
            topic = feed.pick_topic()
            if topic:
                return topic
        return None


class LiveFeedStub:
    """Offline feed backed by the bundled topic bank."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._bank = TopicBankFeed(rng=rng)

    def pick_topic(self) -> str | None:
        """Return a bundled topic without network access."""
        return self._bank.pick_topic()


def build_feed(
    sources: list[str] | str | None = None,
    *,
    topic_bank_path: Path | None = None,
    rng: random.Random | None = None,
) -> LiveFeedProvider:
    """Build a feed from configured sources."""
    resolved = ["topic_bank"] if sources is None else sources
    if isinstance(resolved, str):
        resolved = [resolved]
    feeds: list[LiveFeedProvider] = []
    for source in resolved:
        if source == "topic_bank":
            feeds.append(TopicBankFeed(path=topic_bank_path, rng=rng))
            continue
        if source == "hackernews":
            feeds.append(HackerNewsFeed(rng=rng))
            continue
        logger.warning("Unknown feed source %r; skipping.", source)
    if not feeds:
        return LiveFeedStub(rng=rng)
    if len(feeds) == 1:
        return feeds[0]
    return CombinedFeed(feeds, rng=rng)
