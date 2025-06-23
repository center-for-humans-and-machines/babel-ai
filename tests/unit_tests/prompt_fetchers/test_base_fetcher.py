"""Tests for BasePromptFetcher and factory functionality."""

from unittest.mock import patch

import pytest

from babel_ai.enums import FetcherType
from babel_ai.prompt_fetcher import (
    BasePromptFetcher,
    InfiniteConversationFetcher,
    RandomPromptFetcher,
    ShareGPTConversationFetcher,
    TopicalChatConversationFetcher,
)


class TestBasePromptFetcher:
    """Test suite for BasePromptFetcher."""

    def test_base_prompt_fetcher_is_abstract(self):
        """Test that BasePromptFetcher cannot be instantiated."""
        with pytest.raises(TypeError):
            BasePromptFetcher()

    def test_create_fetcher_random(self):
        """Test creating a RandomPromptFetcher."""
        fetcher = BasePromptFetcher.create_fetcher(FetcherType.RANDOM)
        assert isinstance(fetcher, RandomPromptFetcher)

    def test_create_fetcher_sharegpt(self):
        """Test creating a ShareGPTConversationFetcher."""
        with patch("builtins.open"), patch("json.load", return_value=[]):
            fetcher = BasePromptFetcher.create_fetcher(
                FetcherType.SHAREGPT, data_path="dummy_path.json"
            )
            assert isinstance(fetcher, ShareGPTConversationFetcher)

    def test_create_fetcher_infinite_conversation(self):
        """Test creating an InfiniteConversationFetcher."""
        with patch("pathlib.Path.glob", return_value=[]):
            fetcher = BasePromptFetcher.create_fetcher(
                FetcherType.INFINITE_CONVERSATION, data_dir="dummy_dir"
            )
            assert isinstance(fetcher, InfiniteConversationFetcher)

    def test_create_fetcher_topical_chat(self):
        """Test creating a TopicalChatConversationFetcher."""
        with patch("builtins.open"), patch(
            "pathlib.Path.exists", return_value=True
        ):
            fetcher = BasePromptFetcher.create_fetcher(
                FetcherType.TOPICAL_CHAT,
                rare_file_path="dummy_rare.jsonl",
                freq_file_path="dummy_freq.jsonl",
            )
            assert isinstance(fetcher, TopicalChatConversationFetcher)
