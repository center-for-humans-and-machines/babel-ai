"""RAG components for the retrieval-augmented scaffolder."""

from conversation.rag.nudge import (
    RagMode,
    RagNudge,
    RagNudgeProvider,
    RagUnavailable,
)
from conversation.rag.scaffolder import RagScaffolder
from conversation.rag.search import (
    DuckDuckGoSearchClient,
    FallbackSearchClient,
    SearchClient,
    SearchResult,
    WikipediaSearchClient,
    build_search_client,
    fetch_page_text,
)
from conversation.rag.word_sampler import RandomWordSampler

__all__ = [
    "DuckDuckGoSearchClient",
    "FallbackSearchClient",
    "RagMode",
    "RagNudge",
    "RagNudgeProvider",
    "RagScaffolder",
    "RagUnavailable",
    "RandomWordSampler",
    "SearchClient",
    "SearchResult",
    "WikipediaSearchClient",
    "build_search_client",
    "fetch_page_text",
]
