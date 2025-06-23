from enum import Enum
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from babel_ai.analyzer import Analyzer
    from babel_ai.prompt_fetcher import BasePromptFetcher


class FetcherType(Enum):
    """Fetcher types for prompts."""

    SHAREGPT = "sharegpt"
    RANDOM = "random"
    INFINITE_CONVERSATION = "infinite_conversation"
    TOPICAL_CHAT = "topical_chat"

    def get_fetcher_class(self) -> Type["BasePromptFetcher"]:
        """Get the corresponding fetcher class for this type.

        Returns:
            The fetcher class corresponding to this enum value
        """
        # Import here to avoid circular imports
        from babel_ai.prompt_fetcher import (
            InfiniteConversationFetcher,
            RandomPromptFetcher,
            ShareGPTConversationFetcher,
            TopicalChatConversationFetcher,
        )

        mapping = {
            FetcherType.RANDOM: RandomPromptFetcher,
            FetcherType.SHAREGPT: ShareGPTConversationFetcher,
            FetcherType.INFINITE_CONVERSATION: InfiniteConversationFetcher,
            FetcherType.TOPICAL_CHAT: TopicalChatConversationFetcher,
        }
        return mapping[self]


class AgentSelectionMethod(Enum):
    """Agent selection methods."""

    ROUND_ROBIN = "round_robin"


class AnalyzerType(Enum):
    """Analyzer types for LLM outputs."""

    SIMILARITY = "similarity"

    def get_class(self) -> Type["Analyzer"]:
        """Get the corresponding analyzer class for this type."""
        # Import here to avoid circular imports
        from babel_ai.analyzer import SimilarityAnalyzer

        mapping = {
            AnalyzerType.SIMILARITY: SimilarityAnalyzer,
        }
        return mapping[self]
