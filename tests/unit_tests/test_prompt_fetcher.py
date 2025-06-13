"""Tests for the base prompt fetcher classes."""

from unittest.mock import MagicMock, patch

import pytest

from babel_ai.prompt_fetcher import BasePromptFetcher, PromptFetcher


class TestBasePromptFetcher:
    """Test suite for BasePromptFetcher."""

    def test_base_prompt_fetcher_is_abstract(self):
        """Test that BasePromptFetcher cannot be instantiated."""
        with pytest.raises(TypeError):
            BasePromptFetcher()


class TestPromptFetcher:
    """Test suite for PromptFetcher."""

    def test_init(self):
        """Test PromptFetcher initialization."""
        fetcher = PromptFetcher()
        assert fetcher.headers == {
            "User-Agent": "Mozilla/5.0 (compatible; PromptFetcher/1.0)"
        }

    @patch("requests.get")
    def test_get_random_prompt_with_category(self, mock_get):
        """Test getting a random prompt with specific category."""
        fetcher = PromptFetcher()

        # Test creative category
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"children": [{"data": {"title": "Test prompt"}}]}
        }
        mock_get.return_value = mock_response

        prompt = fetcher.get_random_prompt(category="creative")
        assert prompt == "Test prompt"
        mock_get.assert_called_once()

    def test_get_random_prompt_with_category_none(self):
        """Test getting a random prompt with no category."""
        fetcher = PromptFetcher()

        # Mock the individual methods
        with patch.object(
            fetcher, "_get_writing_prompt"
        ) as mock_writing, patch.object(
            fetcher, "_get_analytical_prompt"
        ) as mock_analytical, patch.object(
            fetcher, "_get_conversational_prompt"
        ) as mock_conversational:

            # Set up mock responses
            mock_writing.return_value = "Test writing prompt"
            mock_analytical.return_value = "Test analytical prompt"
            mock_conversational.return_value = "Test conversational prompt"

            # Run get_random_prompt multiple times
            # !!! This is only stochastically true, if it
            # !!! fails, run the test again.
            for _ in range(100):  # Run enough times to likely hit all methods
                fetcher.get_random_prompt()

            # Verify each method was called at least once
            assert mock_writing.call_count > 0
            assert mock_analytical.call_count > 0
            assert mock_conversational.call_count > 0

            # Verify total calls equals number of runs
            total_calls = (
                mock_writing.call_count
                + mock_analytical.call_count
                + mock_conversational.call_count
            )
            assert total_calls == 100

    @patch("requests.get")
    def test_get_random_prompt_fallback(self, mock_get):
        """Test fallback behavior when request fails."""
        fetcher = PromptFetcher()
        mock_get.side_effect = Exception("API Error")

        prompt = fetcher.get_random_prompt()
        assert prompt in [
            "Share your thoughts on an interesting topic.",
            "Tell me about something you find fascinating.",
            "Describe a concept that intrigues you.",
            "What's on your mind?",
        ]

    @patch("requests.get")
    def test_get_writing_prompt(self, mock_get):
        """Test fetching writing prompts from Reddit."""
        fetcher = PromptFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"children": [{"data": {"title": "Test writing prompt"}}]}
        }
        mock_get.return_value = mock_response

        prompt = fetcher._get_writing_prompt()
        assert prompt == "Test writing prompt"
        mock_get.assert_called_once_with(
            "https://www.reddit.com/r/WritingPrompts/new.json",
            headers=fetcher.headers,
        )

    @patch("requests.get")
    def test_get_analytical_prompt(self, mock_get):
        """Test generating analytical prompts."""
        fetcher = PromptFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = ["word1", "word2"]
        mock_get.return_value = mock_response

        prompt = fetcher._get_analytical_prompt()
        assert any(
            template in prompt
            for template in [
                "Analyze the relationship between",
                "Compare and contrast",
                "Explain how",
                "What are the implications of",
            ]
        )
        assert "word1" in prompt and "word2" in prompt

    @patch("requests.get")
    def test_get_conversational_prompt(self, mock_get):
        """Test generating conversational prompts."""
        fetcher = PromptFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = ["testword"]
        mock_get.return_value = mock_response

        prompt = fetcher._get_conversational_prompt()
        assert any(
            template in prompt
            for template in [
                "What are your thoughts on",
                "How does",
                "Why is",
                "Share your perspective on",
            ]
        )
        assert "testword" in prompt

    @patch("requests.get")
    def test_get_random_words(self, mock_get):
        """Test fetching random words."""
        fetcher = PromptFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = ["word1", "word2"]
        mock_get.return_value = mock_response

        words = fetcher._get_random_words(count=2)
        assert words == ["word1", "word2"]
        mock_get.assert_called_once_with(
            "https://random-word-api.herokuapp.com/word?number=2",
            headers=fetcher.headers,
        )
