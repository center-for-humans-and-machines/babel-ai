"""Deterministic three-behavior scaffolding policy."""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from conversation.messages import ConversationMessage

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_META_PHRASES = (
    "i am here to help",
    "i'm here to help",
    "feel free to ask",
    "let me know",
    "my capabilities",
    "i am designed to",
    "i'm designed to",
)
_BOILERPLATE = (
    "feel free to",
    "let me know",
    "i am here to help",
    "i'm here to help",
)
_ENCOURAGEMENTS = (
    "Add one concrete detail.",
    "Give one specific example.",
    "Develop one consequence.",
    "Name one edge case.",
)


class ScaffolderAction(str, Enum):
    """Observable branch selected by the scaffolder."""

    THRIVE_PROTECTION = "thrive_protection"
    MEMORY_RESURFACE = "memory_resurface"
    TOPIC_INJECTION = "topic_injection"


@dataclass(frozen=True)
class TopicCard:
    """One removable topic remembered from an informative turn."""

    summary: str
    features: frozenset[str]
    source_turn: int
    created_at: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        data = asdict(self)
        data["features"] = sorted(self.features)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicCard":
        """Restore a serialized topic card."""
        return cls(
            summary=str(data["summary"]),
            features=frozenset(data["features"]),
            source_turn=int(data["source_turn"]),
            created_at=int(data["created_at"]),
        )


@dataclass(frozen=True)
class DecisionScores:
    """Lexical signals used for one policy decision."""

    informative: bool
    novelty: float
    continuity: float
    content_tokens: int
    meta_detected: bool


@dataclass(frozen=True)
class ScaffolderTurn:
    """Text and trace data emitted by the policy."""

    content: str
    action: ScaffolderAction
    scores: DecisionScores
    memory_size: int
    topic_source_turn: int | None = None


class ThreeBehaviorScaffolder:
    """Protect thriving, resurface memory, or inject a new topic."""

    def __init__(
        self,
        topics: Iterable[str],
        *,
        min_content_tokens: int = 8,
        novelty_threshold: float = 0.25,
        continuity_threshold: float = 0.10,
        history_window: int = 8,
        stuck_turns: int = 2,
        memory_cooldown: int = 2,
        memory_size: int = 20,
        similarity_threshold: float = 0.70,
        random_seed: int = 0,
    ) -> None:
        self.min_content_tokens = min_content_tokens
        self.novelty_threshold = novelty_threshold
        self.continuity_threshold = continuity_threshold
        self.history_window = history_window
        self.stuck_turns = stuck_turns
        self.memory_cooldown = memory_cooldown
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.random_seed = random_seed
        self._base_topics = tuple(
            topic.strip() for topic in topics if topic.strip()
        )
        if not self._base_topics:
            raise ValueError("scaffolder requires at least one topic")
        self._memory: list[TopicCard] = []
        self._history: list[frozenset[str]] = []
        self._processed_turns: set[int] = set()
        self._peer_turn_count = 0
        self._noninformative_streak = 0
        self._encouragement_index = 0
        self._topic_cycle = 0
        self._topic_deck: list[str] = []
        self._topic_index = 0
        self._reset_topic_deck()

    def respond(
        self,
        messages: list[ConversationMessage],
        *,
        speaker: str,
    ) -> ScaffolderTurn:
        """Evaluate the latest peer turn and choose one behavior."""
        peers = [message for message in messages if message.speaker != speaker]
        if not peers:
            raise ValueError("scaffolder needs a prior peer message")
        unseen = [
            message
            for message in peers
            if message.turn_index not in self._processed_turns
        ]
        if not unseen:
            raise ValueError("scaffolder needs a new peer message")
        for message in unseen[:-1]:
            self._ingest_seed(message)
        latest = unseen[-1]
        self._processed_turns.add(latest.turn_index)
        scores = self.score(latest.content)
        features = text_features(latest.content)
        self._history.append(features)
        self._peer_turn_count += 1

        if scores.informative:
            self._noninformative_streak = 0
            self._remember(latest, features)
            return self._encourage(scores)

        self._noninformative_streak += 1
        if self._noninformative_streak < self.stuck_turns:
            return self._encourage(scores)

        card = self._pop_memory(features)
        if card is not None:
            return ScaffolderTurn(
                content=(
                    f"Earlier this thread appeared: {card.summary} "
                    "Add a new angle to it."
                ),
                action=ScaffolderAction.MEMORY_RESURFACE,
                scores=scores,
                memory_size=len(self._memory),
                topic_source_turn=card.source_turn,
            )
        topic = self._next_topic()
        return ScaffolderTurn(
            content=(
                f"New topic: {topic}. "
                "Give one informative connection or observation."
            ),
            action=ScaffolderAction.TOPIC_INJECTION,
            scores=scores,
            memory_size=0,
        )

    def score(self, text: str) -> DecisionScores:
        """Score lexical novelty and continuity without model inference."""
        features = text_features(text)
        tokens = content_tokens(text)
        recent = self._history[-self.history_window :]
        seen = frozenset().union(*recent) if recent else frozenset()
        novelty = _difference_ratio(features, seen)
        active = recent[-1] if recent else frozenset()
        continuity = _overlap_ratio(features, active)
        raw_normalized = _basic_normalize(text)
        meta_detected = any(
            phrase in raw_normalized for phrase in _META_PHRASES
        )
        enough_content = len(tokens) >= self.min_content_tokens
        related = not active or continuity >= self.continuity_threshold
        informative = (
            enough_content
            and novelty >= self.novelty_threshold
            and related
            and not meta_detected
        )
        return DecisionScores(
            informative=informative,
            novelty=novelty,
            continuity=continuity,
            content_tokens=len(tokens),
            meta_detected=meta_detected,
        )

    def export_state(self) -> dict[str, Any]:
        """Serialize mutable state for exact checkpoint recovery."""
        return {
            "memory": [card.to_dict() for card in self._memory],
            "history": [sorted(features) for features in self._history],
            "processed_turns": sorted(self._processed_turns),
            "peer_turn_count": self._peer_turn_count,
            "noninformative_streak": self._noninformative_streak,
            "encouragement_index": self._encouragement_index,
            "topic_cycle": self._topic_cycle,
            "topic_deck": self._topic_deck,
            "topic_index": self._topic_index,
        }

    def import_state(self, data: dict[str, Any]) -> None:
        """Restore mutable state from a checkpoint."""
        self._memory = [
            TopicCard.from_dict(card) for card in data.get("memory", [])
        ]
        self._history = [
            frozenset(features) for features in data.get("history", [])
        ]
        self._processed_turns = set(data.get("processed_turns", []))
        self._peer_turn_count = int(data.get("peer_turn_count", 0))
        self._noninformative_streak = int(
            data.get("noninformative_streak", 0)
        )
        self._encouragement_index = int(
            data.get("encouragement_index", 0)
        )
        self._topic_cycle = int(data.get("topic_cycle", 0))
        self._topic_deck = list(data.get("topic_deck", []))
        self._topic_index = int(data.get("topic_index", 0))

    def _ingest_seed(self, message: ConversationMessage) -> None:
        features = text_features(message.content)
        self._processed_turns.add(message.turn_index)
        self._history.append(features)
        self._peer_turn_count += 1
        if len(content_tokens(message.content)) >= self.min_content_tokens:
            self._remember(message, features)

    def _remember(
        self,
        message: ConversationMessage,
        features: frozenset[str],
    ) -> None:
        if not features:
            return
        if any(
            _similarity(features, card.features) >= self.similarity_threshold
            for card in self._memory
        ):
            return
        card = TopicCard(
            summary=representative_sentence(message.content, self._history),
            features=features,
            source_turn=message.turn_index,
            created_at=self._peer_turn_count,
        )
        self._memory.append(card)
        if len(self._memory) > self.memory_size:
            self._memory.pop(0)

    def _pop_memory(
        self,
        latest_features: frozenset[str],
    ) -> TopicCard | None:
        for index, card in enumerate(self._memory):
            age = self._peer_turn_count - card.created_at
            similarity = _similarity(latest_features, card.features)
            if (
                age >= self.memory_cooldown
                and similarity < self.similarity_threshold
            ):
                return self._memory.pop(index)
        return None

    def _encourage(self, scores: DecisionScores) -> ScaffolderTurn:
        content = _ENCOURAGEMENTS[
            self._encouragement_index % len(_ENCOURAGEMENTS)
        ]
        self._encouragement_index += 1
        return ScaffolderTurn(
            content=content,
            action=ScaffolderAction.THRIVE_PROTECTION,
            scores=scores,
            memory_size=len(self._memory),
        )

    def _next_topic(self) -> str:
        if self._topic_index >= len(self._topic_deck):
            self._topic_cycle += 1
            self._reset_topic_deck()
        topic = self._topic_deck[self._topic_index]
        self._topic_index += 1
        return topic

    def _reset_topic_deck(self) -> None:
        self._topic_deck = list(self._base_topics)
        rng = random.Random(self.random_seed + self._topic_cycle)
        rng.shuffle(self._topic_deck)
        self._topic_index = 0


def normalize_text(text: str) -> str:
    """Normalize text for deterministic lexical comparison."""
    normalized = _basic_normalize(text)
    for phrase in _BOILERPLATE:
        normalized = normalized.replace(phrase, " ")
    return " ".join(normalized.split())


def _basic_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


def content_tokens(text: str) -> list[str]:
    """Extract simple multilingual content tokens."""
    tokens: list[str] = []
    for token in _WORD_RE.findall(normalize_text(text)):
        if _CJK_RE.search(token):
            tokens.extend(char for char in token if _CJK_RE.match(char))
        elif len(token) > 1:
            tokens.append(token)
    return tokens


def text_features(text: str) -> frozenset[str]:
    """Build word and character-trigram features."""
    normalized = normalize_text(text)
    words = {f"w:{token}" for token in content_tokens(normalized)}
    compact = re.sub(r"\s+", " ", normalized)
    trigrams = {
        f"c:{compact[index:index + 3]}"
        for index in range(max(0, len(compact) - 2))
        if compact[index:index + 3].strip()
    }
    return frozenset(words | trigrams)


def representative_sentence(
    text: str,
    history: list[frozenset[str]],
    max_length: int = 180,
) -> str:
    """Select the sentence with the most features unseen in history."""
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_RE.split(text)
        if sentence.strip()
    ]
    if not sentences:
        return text.strip()[:max_length]
    seen = frozenset().union(*history[:-1]) if len(history) > 1 else frozenset()
    best = max(
        sentences,
        key=lambda sentence: len(text_features(sentence) - seen),
    )
    if len(best) <= max_length:
        return best
    return f"{best[:max_length - 1].rstrip()}…"


def _difference_ratio(
    current: frozenset[str],
    previous: frozenset[str],
) -> float:
    return len(current - previous) / len(current) if current else 0.0


def _overlap_ratio(
    current: frozenset[str],
    previous: frozenset[str],
) -> float:
    return len(current & previous) / len(current) if current else 0.0


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    smaller = min(len(left), len(right))
    return len(left & right) / smaller if smaller else 0.0
