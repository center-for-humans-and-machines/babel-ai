"""Generate a grounded nudge with the experiment's configured LLM."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from api.llm_interface import LLMInterface
from conversation.messages import ConversationMessage
from conversation.rag.search import (
    SearchClient,
    SearchResult,
    fetch_page_text,
)
from conversation.rag.word_sampler import RandomWordSampler
from models.configs import AgentConfig

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant embedded in a conversation. You help a "
    "language model stay novel and grounded. Always answer with a single "
    "short instruction (one or two sentences) that the model should follow."
)

_LLMCall = Callable[..., str]
_PageFetcher = Callable[..., str]


class RagMode(str, Enum):
    """Which scaffold behavior the RAG pipeline is serving."""

    NOVELTY = "novelty"
    TOPIC = "topic"


class RagUnavailable(RuntimeError):
    """Raised when the RAG pipeline cannot produce a grounded nudge."""


@dataclass(frozen=True)
class RagNudge:
    """A grounded nudge plus the provenance that produced it."""

    content: str
    mode: RagMode
    words: tuple[str, ...]
    query: str
    source_url: Optional[str]
    source_title: Optional[str]


class RagNudgeProvider:
    """Sample words, search the web, and ask an LLM for a grounded nudge."""

    def __init__(
        self,
        *,
        sampler: RandomWordSampler,
        search_client: SearchClient,
        llm_config: AgentConfig,
        num_words: int = 5,
        top_k: int = 5,
        fetch_page: bool = True,
        max_source_chars: int = 4000,
        search_timeout: float = 8.0,
        context_turns: int = 6,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: Optional[int] = 300,
        seed: int = 0,
        llm_fn: Optional[_LLMCall] = None,
        page_fetcher: Optional[_PageFetcher] = None,
    ) -> None:
        if num_words < 1:
            raise ValueError("num_words must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._sampler = sampler
        self._search_client = search_client
        self._llm_config = llm_config
        self._num_words = num_words
        self._top_k = top_k
        self._fetch_page = fetch_page
        self._max_source_chars = max_source_chars
        self._search_timeout = search_timeout
        self._context_turns = context_turns
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._rng = random.Random(seed)
        self._llm = llm_fn or LLMInterface.generate_response
        self._page_fetcher = page_fetcher or fetch_page_text

    def produce(
        self,
        mode: RagMode,
        messages: list[ConversationMessage],
        *,
        speaker: str,
    ) -> RagNudge:
        """Run the full pipeline or raise :class:`RagUnavailable`."""
        try:
            words = self._sampler.sample(self._num_words)
        except Exception as exc:  # loading/network failures must not break
            raise RagUnavailable(f"word sampling failed: {exc}") from exc
        query, results = self._search_words(words)
        result = self._rng.choice(results)
        source = self._source_text(result)
        prompt = self._build_messages(mode, messages, speaker, result, source)
        try:
            content = self._llm(
                messages=prompt,
                provider=self._llm_config.provider,
                model=self._llm_config.model,
                temperature=self._llm_config.temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise RagUnavailable(f"llm nudge failed: {exc}") from exc
        content = (content or "").strip()
        if not content:
            raise RagUnavailable("llm returned an empty nudge")
        return RagNudge(
            content=content,
            mode=mode,
            words=tuple(words),
            query=query,
            source_url=result.url or None,
            source_title=result.title or None,
        )

    def _search_words(
        self, words: tuple[str, ...] | list[str]
    ) -> tuple[str, list[SearchResult]]:
        """Search the joined words, narrowing the query until it hits."""
        last_error: Exception | None = None
        for query in _candidate_queries(words):
            try:
                results = self._search_client.search(query, self._top_k)
            except Exception as exc:
                last_error = exc
                continue
            if results:
                return query, results
        if last_error is not None:
            raise RagUnavailable(
                f"search failed: {last_error}"
            ) from last_error
        raise RagUnavailable("search returned no results")

    def _source_text(self, result: SearchResult) -> str:
        if self._fetch_page and result.url:
            try:
                page = self._page_fetcher(
                    result.url,
                    timeout=self._search_timeout,
                    max_chars=self._max_source_chars,
                )
                if page.strip():
                    return page
            except Exception as exc:
                logger.warning(
                    "RAG page fetch failed for %s: %s", result.url, exc
                )
        return result.snippet

    def _build_messages(
        self,
        mode: RagMode,
        messages: list[ConversationMessage],
        speaker: str,
        result: SearchResult,
        source: str,
    ) -> list[dict[str, str]]:
        context = self._conversation_context(messages, speaker)
        if mode is RagMode.NOVELTY:
            task = (
                "Write one short novelty nudge for the conversation above. "
                "It must introduce a genuinely new but clearly connected idea, "
                "grounded in the source. Address the model directly."
            )
        else:
            task = (
                "Write one short instruction that steers the conversation "
                "toward the new topic suggested by the source, while keeping "
                "one clear connection to the current discussion."
            )
        user = (
            f"Conversation so far:\n{context}\n\n"
            f"Source: {result.title} ({result.url})\n"
            f"{source}\n\n"
            f"Task: {task}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def _conversation_context(
        self, messages: list[ConversationMessage], speaker: str
    ) -> str:
        peers = [message for message in messages if message.speaker != speaker]
        recent = peers[-self._context_turns :]
        lines = [f"{message.speaker}: {message.content}" for message in recent]
        return "\n".join(lines) if lines else "(empty)"

    def export_state(self) -> dict[str, Any]:
        """Serialize RNG state so resume picks the same source deterministically."""
        sampler_state = getattr(self._sampler, "export_state", None)
        return {
            "rng_state": _encode_rng_state(self._rng.getstate()),
            "sampler": sampler_state() if callable(sampler_state) else None,
        }

    def import_state(self, data: dict[str, Any]) -> None:
        """Restore RNG state from a checkpoint."""
        if "rng_state" in data and data["rng_state"] is not None:
            self._rng.setstate(_decode_rng_state(data["rng_state"]))
        sampler_state = getattr(self._sampler, "import_state", None)
        if callable(sampler_state) and data.get("sampler") is not None:
            sampler_state(data["sampler"])


def _candidate_queries(words: object) -> list[str]:
    """Query candidates from most to least specific.

    Five arbitrary embedding words rarely match a full-text search, so the
    provider falls back to progressively shorter prefixes while still
    preferring the direct join.
    """
    cleaned = [str(word) for word in words if str(word).strip()]
    if not cleaned:
        return []
    candidates = [" ".join(cleaned)]
    if len(cleaned) >= 2:
        candidates.append(" ".join(cleaned[:2]))
    candidates.append(cleaned[0])
    seen: set[str] = set()
    unique: list[str] = []
    for query in candidates:
        if query and query not in seen:
            seen.add(query)
            unique.append(query)
    return unique


def _encode_rng_state(state: tuple) -> dict[str, Any]:
    version, internal, gauss_next = state
    return {
        "version": int(version),
        "internal": [int(value) for value in internal],
        "gauss_next": gauss_next,
    }


def _decode_rng_state(payload: Any) -> tuple:
    if isinstance(payload, dict):
        return (
            int(payload["version"]),
            tuple(int(value) for value in payload["internal"]),
            payload.get("gauss_next"),
        )
    version, internal, gauss_next = payload
    return int(version), tuple(int(value) for value in internal), gauss_next
