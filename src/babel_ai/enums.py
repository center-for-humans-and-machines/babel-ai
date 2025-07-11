import logging
from enum import Enum
from typing import TYPE_CHECKING, Generator, List, Type

if TYPE_CHECKING:
    from babel_ai.agent import Agent
    from babel_ai.analyzer import Analyzer
    from babel_ai.prompt_fetcher import BasePromptFetcher

logger = logging.getLogger(__name__)


class FetcherType(Enum):
    """Fetcher types for loading conversation prompts.

    This enum defines the different data sources and formats that can be used
    to load conversation prompts for drift experiments.

    Available types:
        SHAREGPT: Loads conversations from ShareGPT dataset format
        RANDOM: Generates random prompts for testing
        INFINITE_CONVERSATION: Loads from Infinite Conversation dataset
        TOPICAL_CHAT: Loads from Topical Chat dataset format

    Example:
        >>> fetcher_type = FetcherType.SHAREGPT
        >>> fetcher_class = fetcher_type.get_fetcher_class()
        >>> fetcher = fetcher_class(data_path="conversations.json")
        >>> conversation = fetcher.get_conversation()
    """

    SHAREGPT = "sharegpt"
    RANDOM = "random"
    INFINITE_CONVERSATION = "infinite_conversation"
    TOPICAL_CHAT = "topical_chat"

    def get_fetcher_class(self) -> Type["BasePromptFetcher"]:
        """Get the corresponding fetcher class for this type.

        Returns:
            The fetcher class corresponding to this enum value
        """
        logger.info(f"Getting fetcher class for fetcher type: {self.value}")

        # Import here to avoid circular imports
        logger.debug("Importing fetcher classes to avoid circular imports")
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
        logger.debug(f"Fetcher class for {self.value}: {mapping[self]}")
        return mapping[self]

    def get_kwargs_mapping(self) -> List[str]:
        """
        Get the kwargs for the fetcher.
        Depending on the fetcher type.
        """
        logger.info(f"Getting requiredkwargs mapping for {self.value}")
        required_kwargs = {
            FetcherType.RANDOM: ["category"],
            FetcherType.SHAREGPT: [
                "data_path",
                "min_messages",
                "max_messages",
            ],
            FetcherType.INFINITE_CONVERSATION: [
                "data_path",
                "min_messages",
                "max_messages",
            ],
            FetcherType.TOPICAL_CHAT: [
                "data_path",
                "second_data_path",
                "min_messages",
                "max_messages",
            ],
        }
        logger.debug(
            "Required kwargs mapping for "
            f"{self.value}: {required_kwargs[self]}"
        )
        return required_kwargs[self]


class AgentSelectionMethod(Enum):
    """Agent selection methods for choosing which agent responds next.

    This enum defines different strategies for selecting which agent
    from a pool of agents should generate the next response
    in a conversation.

    Available methods:
        ROUND_ROBIN: Cycles through agents in order, giving each agent a turn
            before starting over at the beginning. This ensures even
            distribution of responses across all agents.

    Example:
        >>> method = AgentSelectionMethod.ROUND_ROBIN
        >>> agents = [Agent1(), Agent2(), Agent3()]
        >>> generator = method.get_generator(agents)
        >>> next(generator)  # Returns Agent1
        >>> next(generator)  # Returns Agent2
        >>> next(generator)  # Returns Agent3
        >>> next(generator)  # Returns Agent1 again
    """

    ROUND_ROBIN = "round_robin"

    def get_generator(
        self, agents: List[Type["Agent"]]
    ) -> Generator[Type["Agent"], None, None]:
        """Get the corresponding generator for this method."""
        logger.info(f"Getting generator for {self.value}")
        logger.debug(f"Agents: {[a.id for a in agents]}")

        # Import here to avoid circular imports
        logger.debug(
            "Importing round_robin_agent_selection to avoid circular imports"
        )
        from babel_ai.agent import round_robin_agent_selection

        if self == AgentSelectionMethod.ROUND_ROBIN:
            logger.debug(
                f"Using round robin agent selection with {len(agents)} agents."
            )
            return round_robin_agent_selection(agents)


class AnalyzerType(Enum):
    """Analyzer types for measuring drift in LLM outputs.

    This enum defines different strategies for analyzing and measuring how LLM
    responses change or drift over the course of a conversation.

    Available types:
        SIMILARITY: Measures lexical and semantic similarity between responses
            to detect changes in language patterns and content. Includes both
            pairwise comparisons and sliding window analysis.

    Example:
        >>> analyzer_type = AnalyzerType.SIMILARITY
        >>> analyzer_class = analyzer_type.get_class()
        >>> analyzer = analyzer_class(analyze_window=5)
        >>> # Analyzer will compute similarity metrics between responses
    """

    SIMILARITY = "similarity"

    def get_class(self) -> Type["Analyzer"]:
        """Get the corresponding analyzer class for this type."""
        logger.info(f"Getting analyzer class for {self.value}")

        # Import here to avoid circular imports
        logger.debug("Importing SimilarityAnalyzer to avoid circular imports")
        from babel_ai.analyzer import SimilarityAnalyzer

        mapping = {
            AnalyzerType.SIMILARITY: SimilarityAnalyzer,
        }
        logger.debug(f"Analyzer class for {self.value}: {mapping[self]}")
        return mapping[self]
