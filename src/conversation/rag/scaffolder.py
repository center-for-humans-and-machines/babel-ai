"""Retrieval-augmented scaffolder built on the three-behavior policy.

It reuses the deterministic state machine (scoring, memory, hysteresis) and
only overrides how novelty nudges and topic injections are produced: random
words -> web search -> LLM-generated grounded text. Any failure falls back
to the deterministic wording so conversations never break.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Iterable

from conversation.messages import ConversationMessage
from conversation.rag.nudge import RagMode, RagNudgeProvider, RagUnavailable
from conversation.scaffolder import (
    NoveltyNudgeKind,
    ScaffoldNudge,
    ThreeBehaviorScaffolder,
)

logger = logging.getLogger(__name__)


class RagScaffolder(ThreeBehaviorScaffolder):
    """Three-behavior policy whose nudges are grounded in a web search."""

    def __init__(
        self,
        topics: Iterable[str],
        *,
        provider: RagNudgeProvider,
        **kwargs: Any,
    ) -> None:
        super().__init__(topics, **kwargs)
        self._provider = provider

    def _produce_novelty(
        self,
        messages: list[ConversationMessage],
        speaker: str,
    ) -> ScaffoldNudge:
        return self._run_provider(RagMode.NOVELTY, messages, speaker)

    def _produce_topic(
        self,
        messages: list[ConversationMessage],
        speaker: str,
    ) -> ScaffoldNudge:
        return self._run_provider(RagMode.TOPIC, messages, speaker)

    def _run_provider(
        self,
        mode: RagMode,
        messages: list[ConversationMessage],
        speaker: str,
    ) -> ScaffoldNudge:
        try:
            nudge = self._provider.produce(mode, messages, speaker=speaker)
        except RagUnavailable as exc:
            return self._fallback(mode, messages, speaker, str(exc))
        except Exception as exc:  # defensive: never break the conversation
            logger.exception("Unexpected RAG scaffolder failure: %s", exc)
            return self._fallback(mode, messages, speaker, str(exc))
        return ScaffoldNudge(
            content=nudge.content,
            novelty_nudge_kind=(
                NoveltyNudgeKind.RAG_SEARCH
                if mode is RagMode.NOVELTY
                else None
            ),
            rag_used=True,
            rag_mode=mode.value,
            rag_words=nudge.words,
            rag_query=nudge.query,
            rag_source_url=nudge.source_url,
            rag_source_title=nudge.source_title,
        )

    def _fallback(
        self,
        mode: RagMode,
        messages: list[ConversationMessage],
        speaker: str,
        reason: str,
    ) -> ScaffoldNudge:
        logger.warning(
            "RAG scaffolder falling back to deterministic %s: %s",
            mode.value,
            reason,
        )
        if mode is RagMode.NOVELTY:
            base = super()._produce_novelty(messages, speaker)
        else:
            base = super()._produce_topic(messages, speaker)
        return replace(
            base,
            rag_used=False,
            rag_mode=mode.value,
            rag_fallback_reason=reason,
        )

    def export_state(self) -> dict[str, Any]:
        """Serialize policy and RAG provider state for checkpoint resume."""
        state = super().export_state()
        state["rag"] = self._provider.export_state()
        return state

    def import_state(self, data: dict[str, Any]) -> None:
        """Restore policy and RAG provider state from a checkpoint."""
        super().import_state(data)
        if data.get("rag") is not None:
            self._provider.import_state(data["rag"])
