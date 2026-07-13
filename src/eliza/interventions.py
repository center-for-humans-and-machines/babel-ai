"""Optional handlers for ELIZA's generic ``$`` response."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .live_feed import LiveFeedProvider, LiveFeedStub

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
    """Reserved live-feed hook that currently preserves ELIZA output."""

    def __init__(self, feed: LiveFeedProvider) -> None:
        self._feed = feed

    def on_generic(self, ctx: GenericContext) -> GenericResult:
        """Warn that feeds are unavailable and keep the default."""
        logger.warning("Live-feed intervention is unavailable; using ELIZA.")
        return GenericResult()


def build_intervention(
    mode: GenericInterventionMode | str,
    *,
    hints: list[str] | None = None,
    feed: LiveFeedProvider | None = None,
) -> GenericIntervention:
    """Build a generic-response intervention."""
    resolved = GenericInterventionMode(mode)
    if resolved is GenericInterventionMode.PASSTHROUGH:
        return PassthroughIntervention()
    if resolved is GenericInterventionMode.LLM_NUDGE:
        return LLMNudgeIntervention(hints=hints)
    if resolved is GenericInterventionMode.LIVE_FEED:
        return LiveFeedIntervention(feed or LiveFeedStub())
    raise NotImplementedError(
        f"intervention mode {resolved!r} not implemented"
    )
