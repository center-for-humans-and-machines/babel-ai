"""Tests for all enums in babel_ai.enums module."""

from babel_ai.analyzer import Analyzer, SimilarityAnalyzer
from babel_ai.enums import AgentSelectionMethod, AnalyzerType, FetcherType
from babel_ai.prompt_fetcher import (
    BasePromptFetcher,
    InfiniteConversationFetcher,
    RandomPromptFetcher,
    ShareGPTConversationFetcher,
    TopicalChatConversationFetcher,
)


class TestFetcherType:
    """Test suite for FetcherType enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert FetcherType.RANDOM.value == "random"
        assert FetcherType.SHAREGPT.value == "sharegpt"
        assert (
            FetcherType.INFINITE_CONVERSATION.value == "infinite_conversation"
        )
        assert FetcherType.TOPICAL_CHAT.value == "topical_chat"

    def test_get_fetcher_class_random(self):
        """Test getting RandomPromptFetcher class."""
        fetcher_class = FetcherType.RANDOM.get_fetcher_class()
        assert fetcher_class == RandomPromptFetcher

    def test_get_fetcher_class_sharegpt(self):
        """Test getting ShareGPTConversationFetcher class."""
        fetcher_class = FetcherType.SHAREGPT.get_fetcher_class()
        assert fetcher_class == ShareGPTConversationFetcher

    def test_get_fetcher_class_infinite_conversation(self):
        """Test getting InfiniteConversationFetcher class."""
        fetcher_class = FetcherType.INFINITE_CONVERSATION.get_fetcher_class()
        assert fetcher_class == InfiniteConversationFetcher

    def test_get_fetcher_class_topical_chat(self):
        """Test getting TopicalChatConversationFetcher class."""
        fetcher_class = FetcherType.TOPICAL_CHAT.get_fetcher_class()
        assert fetcher_class == TopicalChatConversationFetcher

    def test_all_fetcher_types_have_classes(self):
        """Test that all enum values have corresponding fetcher classes."""
        for fetcher_type in FetcherType:
            fetcher_class = fetcher_type.get_fetcher_class()
            assert fetcher_class is not None
            assert issubclass(fetcher_class, BasePromptFetcher)

    def test_fetcher_type_string_values_are_unique(self):
        """Test that all fetcher type string values are unique."""
        values = [fetcher_type.value for fetcher_type in FetcherType]
        assert len(values) == len(
            set(values)
        ), "Fetcher type values not unique"

    def test_fetcher_enum_to_class_creation_pipeline(self):
        """Test the complete pipeline from FetcherType to fetcher creation."""
        # Test with RandomPromptFetcher (no additional args needed)
        fetcher_class = FetcherType.RANDOM.get_fetcher_class()
        fetcher = BasePromptFetcher.create_fetcher(FetcherType.RANDOM)

        assert isinstance(fetcher, fetcher_class)
        assert isinstance(fetcher, RandomPromptFetcher)


class TestAnalyzerType:
    """Test the AnalyzerType enum."""

    def test_analyzer_type_values(self):
        """Test that AnalyzerType has expected values."""
        assert AnalyzerType.SIMILARITY.value == "similarity"

    def test_get_analyzer_class(self):
        """Test getting analyzer class from enum."""
        analyzer_class = AnalyzerType.SIMILARITY.get_class()
        assert analyzer_class == SimilarityAnalyzer
        assert isinstance(analyzer_class, type)
        assert issubclass(analyzer_class, Analyzer)

    def test_analyzer_type_enum_completeness(self):
        """Test that all enum values have corresponding analyzer classes."""
        for analyzer_type in AnalyzerType:
            analyzer_class = analyzer_type.get_class()
            assert analyzer_class is not None
            assert issubclass(analyzer_class, Analyzer)

    def test_all_analyzer_types_can_be_created(self):
        """Test that all analyzer types can be successfully created."""
        for analyzer_type in AnalyzerType:
            analyzer = Analyzer.create_analyzer(analyzer_type)
            assert isinstance(analyzer, Analyzer)
            assert hasattr(analyzer, "analyze")

    def test_analyzer_type_string_values_are_unique(self):
        """Test that all analyzer type string values are unique."""
        values = [analyzer_type.value for analyzer_type in AnalyzerType]
        assert len(values) == len(
            set(values)
        ), "Analyzer type values not unique"


class TestAgentSelectionMethod:
    """Test suite for AgentSelectionMethod enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert AgentSelectionMethod.ROUND_ROBIN.value == "round_robin"

    def test_agent_selection_method_string_values_are_unique(self):
        """Test that all agent selection method string values are unique."""
        values = [method.value for method in AgentSelectionMethod]
        assert len(values) == len(
            set(values)
        ), "Agent selection method values not unique"
