"""Live-feed interface reserved for a future implementation."""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class LiveFeedProvider(Protocol):
    """Fetch a current headline, when a provider is available."""

    def fetch_headline(self) -> str | None:
        """Return a headline or no result."""
        ...


class LiveFeedStub:
    """Offline placeholder that performs no network request."""

    def fetch_headline(self) -> None:
        """Report that live feeds are not implemented."""
        logger.warning("Live feed is not implemented.")
        return None
