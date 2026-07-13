"""Optional handlers for ELIZA's generic ``$`` response."""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from .live_feed import LiveFeedProvider, build_feed, format_topic_switch

logger = logging.getLogger(__name__)


class GenericInterventionMode(str, Enum):
    """How the partner handles rdimaio ``$`` fallback."""

    PASSTHROUGH = "passthrough"
    LLM_NUDGE = "llm_nudge"
    LIVE_FEED = "live_feed"
    CUSTOM = "custom"


@dataclass
class GenericContext:
    """Context passed to a generic-response intervention."""

    user_turn: str
    default_response: str
    messages: list[dict[str, Any]]
    turn_index: int


@dataclass
class GenericResult:
    """Outcome of a generic-response intervention."""

    partner_text: str | None = None
    topic_switched: bool = False


class GenericIntervention(Protocol):
    """Extension point for vendored ELIZA's ``$`` branch."""

    def on_generic(self, ctx: GenericContext) -> GenericResult:
        """Return replacement text or ``None`` for the default."""
        ...


class PassthroughIntervention:
    """Keep the upstream generic response."""

    def on_generic(self, ctx: GenericContext) -> GenericResult:
        """Preserve the upstream response."""
        return GenericResult()


class LLMNudgeIntervention:
    """Cycle through local hint responses without making network calls."""

    def __init__(self, hints: list[str] | None = None) -> None:
        self._hints = hints or []
        self._next_hint = 0

    def on_generic(self, ctx: GenericContext) -> GenericResult:
        """Return the next configured hint, if one exists."""
        if not self._hints:
            return GenericResult()
        hint = self._hints[self._next_hint % len(self._hints)]
        self._next_hint += 1
        return GenericResult(partner_text=hint)


class LiveFeedIntervention:
    """Sometimes replace generic ``$`` text with a feed topic switch."""

    def __init__(
        self,
        feed: LiveFeedProvider,
        *,
        topic_switch_probability: float = 0.5,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self._feed = feed
        self._probability = topic_switch_probability
        self._rng = rng or random.random

    def on_generic(self, ctx: GenericContext) -> GenericResult:
        """Maybe return a topic-switch line from the configured feed."""
        if self._rng() >= self._probability:
            return GenericResult()
        topic = self._feed.pick_topic()
        if not topic:
            logger.warning("Live feed returned no topic; using ELIZA default.")
            return GenericResult()
        return GenericResult(
            partner_text=format_topic_switch(topic),
            topic_switched=True,
        )


def build_intervention(
    mode: GenericInterventionMode | str,
    *,
    hints: list[str] | None = None,
    feed: LiveFeedProvider | None = None,
    topic_switch_probability: float = 0.5,
    feed_sources: list[str] | None = None,
) -> GenericIntervention:
    """Build a generic-response intervention."""
    resolved = GenericInterventionMode(mode)
    if resolved is GenericInterventionMode.PASSTHROUGH:
        return PassthroughIntervention()
    if resolved is GenericInterventionMode.LLM_NUDGE:
        return LLMNudgeIntervention(hints=hints)
    if resolved is GenericInterventionMode.LIVE_FEED:
        resolved_feed = feed or build_feed(feed_sources)
        return LiveFeedIntervention(
            resolved_feed,
            topic_switch_probability=topic_switch_probability,
        )
    raise NotImplementedError(
        f"intervention mode {resolved!r} not implemented"
    )
