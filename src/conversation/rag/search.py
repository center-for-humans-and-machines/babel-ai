"""Web search and page fetching for the RAG scaffolder.

Search is behind the :class:`SearchClient` protocol so backends can be
swapped without touching the scaffolder. Two key-free backends are
provided:

* :class:`DuckDuckGoSearchClient` scrapes DuckDuckGo's HTML endpoint.
* :class:`WikipediaSearchClient` uses the MediaWiki search API.

DuckDuckGo frequently serves an anti-bot challenge from datacenter or
proxied networks; :class:`FallbackSearchClient` chains backends so a
challenge degrades to the next source instead of failing the turn.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import requests

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
# Wikimedia blocks browser-like agents without context; use a descriptive UA.
_BOT_UA = "BabelAI/0.1 (research; mailto:research@example.com)"
_DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class SearchResult:
    """One search hit."""

    title: str
    url: str
    snippet: str


class SearchClient(Protocol):
    """Return search hits for a query."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` results for ``query``."""
        ...


class DuckDuckGoSearchClient:
    """Scrape DuckDuckGo HTML results (no API key required)."""

    def __init__(
        self, *, timeout: float = 8.0, session: Any | None = None
    ) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` parsed DuckDuckGo results."""
        if not query.strip() or top_k < 1:
            return []
        response = self._session.post(
            _DUCKDUCKGO_HTML,
            data={"q": query},
            headers={"User-Agent": _BROWSER_UA},
            timeout=self._timeout,
        )
        if response.status_code != 200:
            logger.warning(
                "DuckDuckGo returned status %s; no results.",
                response.status_code,
            )
            return []
        return _parse_duckduckgo_results(response.text, top_k)


class WikipediaSearchClient:
    """Search Wikipedia via the MediaWiki API (no API key required)."""

    def __init__(
        self, *, timeout: float = 8.0, session: Any | None = None
    ) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` Wikipedia article hits."""
        if not query.strip() or top_k < 1:
            return []
        response = self._session.get(
            _WIKIPEDIA_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": top_k,
            },
            headers={"User-Agent": _BOT_UA},
            timeout=self._timeout,
        )
        if response.status_code != 200:
            logger.warning(
                "Wikipedia returned status %s; no results.",
                response.status_code,
            )
            return []
        return _parse_wikipedia_results(response.json(), top_k)


class FallbackSearchClient:
    """Try each backend in order until one returns results."""

    def __init__(self, clients: Sequence[SearchClient]) -> None:
        self._clients = list(clients)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the first backend's non-empty result set."""
        for client in self._clients:
            try:
                results = client.search(query, top_k)
            except Exception as exc:
                logger.warning(
                    "Search backend %s failed: %s",
                    type(client).__name__,
                    exc,
                )
                continue
            if results:
                return results
        return []


def build_search_client(
    backend: str = "duckduckgo",
    *,
    timeout: float = 8.0,
    session: Any | None = None,
) -> SearchClient:
    """Build a search client for a configured ``search_backend``."""
    if backend == "wikipedia":
        return WikipediaSearchClient(timeout=timeout, session=session)
    if backend == "auto":
        return FallbackSearchClient(
            [
                DuckDuckGoSearchClient(timeout=timeout, session=session),
                WikipediaSearchClient(timeout=timeout, session=session),
            ]
        )
    return DuckDuckGoSearchClient(timeout=timeout, session=session)


def _parse_duckduckgo_results(html: str, top_k: int) -> list[SearchResult]:
    """Parse the DuckDuckGo HTML endpoint into structured results."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for node in soup.select(".result"):
        link = node.select_one(".result__a")
        if link is None:
            continue
        url = _unwrap_duckduckgo_url(str(link.get("href", "")))
        if not url:
            continue
        snippet_node = node.select_one(".result__snippet")
        snippet = (
            snippet_node.get_text(" ", strip=True) if snippet_node else ""
        )
        results.append(
            SearchResult(
                title=link.get_text(" ", strip=True),
                url=url,
                snippet=snippet,
            )
        )
        if len(results) >= top_k:
            break
    return results


def _unwrap_duckduckgo_url(href: str) -> str:
    """Resolve DuckDuckGo's ``/l/?uddg=`` redirect to the target URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return target or href
    return href


def _parse_wikipedia_results(payload: Any, top_k: int) -> list[SearchResult]:
    """Convert a MediaWiki search payload into structured results."""
    from bs4 import BeautifulSoup

    hits = payload.get("query", {}).get("search", [])
    results: list[SearchResult] = []
    for hit in hits[:top_k]:
        title = str(hit.get("title", "")).strip()
        if not title:
            continue
        snippet = BeautifulSoup(
            str(hit.get("snippet", "")), "html.parser"
        ).get_text(" ", strip=True)
        slug = urllib.parse.quote(title.replace(" ", "_"))
        results.append(
            SearchResult(
                title=title,
                url=f"https://en.wikipedia.org/wiki/{slug}",
                snippet=snippet,
            )
        )
    return results


def fetch_page_text(
    url: str,
    *,
    timeout: float = 8.0,
    max_chars: int = 4000,
    session: Any | None = None,
) -> str:
    """Fetch a page and return its visible text, truncated to ``max_chars``."""
    from bs4 import BeautifulSoup

    client = session or requests.Session()
    response = client.get(
        url, headers={"User-Agent": _BOT_UA}, timeout=timeout
    )
    if response.status_code != 200:
        raise OSError(f"page fetch returned status {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "nav", "header", "footer"]
    ):
        tag.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return text[:max_chars]


__all__ = [
    "DuckDuckGoSearchClient",
    "FallbackSearchClient",
    "SearchClient",
    "SearchResult",
    "WikipediaSearchClient",
    "build_search_client",
    "fetch_page_text",
]
