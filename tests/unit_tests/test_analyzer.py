"""Tests for the analyzer module."""

import logging
from unittest.mock import patch

import pytest

from babel_ai.analyzer import Analyzer, SimilarityAnalyzer
from babel_ai.enums import AnalyzerType
from models import AnalysisResult


@pytest.fixture
def analyzer():
    """Create a SimilarityAnalyzer instance for testing."""
    return SimilarityAnalyzer()


def test_analyzer_init():
    """Test SimilarityAnalyzer initialization."""
    # Test default initialization
    default_analyzer = SimilarityAnalyzer()
    assert default_analyzer.semantic_model_name == "all-MiniLM-L6-v2"
    assert default_analyzer.token_model_name == "gpt2"
    assert default_analyzer.analyze_window == 20

    # Test custom initialization
    custom_analyzer = SimilarityAnalyzer(analyze_window=10)
    assert custom_analyzer.analyze_window == 10


def test_analyze_word_stats(analyzer):
    """Test word statistics analysis."""
    text = "The quick brown fox jumps over the lazy dog"
    (
        word_count,
        unique_word_count,
        coherence_score,
    ) = analyzer._analyze_word_stats(text)

    assert word_count == 9
    assert unique_word_count == 8
    assert coherence_score == 8 / 9


def test_analyze_lexical_similarity(analyzer):
    """Test lexical similarity analysis."""
    # Test similar text case with window_size=1 (direct similarity)
    outputs = [
        "Other sentence",
        "The quick brown super fox",
        "The quick brown super good fox",
    ]

    result = analyzer._analyze_lexical_similarity(outputs, window_size=1)

    assert result == 5 / 6  # 5 intersection words and 6 union words

    # Test non-similar text case
    outputs = ["Other sentence", "The unsimilar fox", "The quick brown fox"]

    result = analyzer._analyze_lexical_similarity(outputs, window_size=1)

    assert result == 2 / 5  # 2 intersection words and 5 union words


def test_analyze_semantic_similarity(analyzer):
    """Test semantic similarity analysis."""
    # Test similar but not identical text with window_size=1
    outputs = [
        "A fast brown fox",
        "A slow not brown dog",
        "The quick brown fox",
    ]

    result = analyzer._analyze_semantic_similarity(outputs, window_size=1)
    assert -1.0 <= result <= 1.0  # Ensure within valid range

    # Test identical text
    outputs = ["A lazy dog", "The quick brown fox", "The quick brown fox"]

    result = analyzer._analyze_semantic_similarity(outputs, window_size=1)
    assert result == 1.0


def test_analyze_lexical_similarity_window(analyzer):
    """Test lexical similarity analysis with rolling window."""
    outputs = [
        "The quick brown fox",
        "The fast brown fox",
        "The slow brown fox",
        "The lazy brown fox",
        "The quick brown cat",  # Current output
    ]

    # Test with window_size=2 (average of last 2 comparisons)
    result = analyzer._analyze_lexical_similarity(outputs, window_size=2)
    assert result is not None
    assert 0.0 <= result <= 1.0

    # Test with window_size=4 (average of last 4 comparisons)
    result = analyzer._analyze_lexical_similarity(outputs, window_size=4)
    assert result is not None
    assert 0.0 <= result <= 1.0

    # Test with window_size larger than available outputs
    result = analyzer._analyze_lexical_similarity(outputs, window_size=10)
    assert result is not None
    assert 0.0 <= result <= 1.0


def test_analyze_semantic_similarity_window(analyzer):
    """Test semantic similarity analysis with rolling window."""
    outputs = [
        "The quick brown fox",
        "The fast brown fox",
        "The slow brown fox",
        "The lazy brown fox",
        "The quick brown cat",  # Current output
    ]

    # Test with window_size=2 (average of last 2 comparisons)
    result = analyzer._analyze_semantic_similarity(outputs, window_size=2)
    assert result is not None
    assert -1.0 <= result <= 1.0

    # Test with window_size=4 (average of last 4 comparisons)
    result = analyzer._analyze_semantic_similarity(outputs, window_size=4)
    assert result is not None
    assert -1.0 <= result <= 1.0

    # Test with window_size larger than available outputs
    result = analyzer._analyze_semantic_similarity(outputs, window_size=10)
    assert result is not None
    assert -1.0 <= result <= 1.0


def test_semantic_similarity_clamping(analyzer):
    """Test that semantic similarity properly clamps faulty cos_sim values."""
    outputs = ["The quick brown fox", "A fast brown dog"]

    # Test clamping values above 1.0
    with patch("babel_ai.analyzer.cos_sim") as mock_cos_sim:
        # Mock cos_sim to return a value above 1.0
        mock_cos_sim.return_value.item.return_value = 1.5

        result = analyzer._analyze_semantic_similarity(outputs)

        # Should be clamped to 1.0
        assert result == 1.0
        mock_cos_sim.assert_called_once()

    # Test clamping values below -1.0
    with patch("babel_ai.analyzer.cos_sim") as mock_cos_sim:
        # Mock cos_sim to return a value below -1.0
        mock_cos_sim.return_value.item.return_value = -1.8

        result = analyzer._analyze_semantic_similarity(outputs)

        # Should be clamped to -1.0
        assert result == -1.0
        mock_cos_sim.assert_called_once()

    # Test multiple faulty values with different window sizes
    outputs = ["First text", "Second text", "Third text", "Fourth text"]

    with patch("babel_ai.analyzer.cos_sim") as mock_cos_sim:
        # Mock cos_sim to return alternating faulty values
        mock_cos_sim.return_value.item.side_effect = [2.5, -3.0, 0.5]

        result = analyzer._analyze_semantic_similarity(outputs, window_size=3)

        # Should average the clamped values: (1.0 + (-1.0) + 0.5) / 3 = 0.5/3
        expected_avg = (1.0 + (-1.0) + 0.5) / 3
        assert result == pytest.approx(expected_avg, rel=1e-6)
        assert mock_cos_sim.call_count == 3


def test_semantic_similarity_clamping_with_logging(analyzer, caplog):
    """Test that clamping logs appropriate warning messages."""
    outputs = ["The quick brown fox", "A fast brown dog"]

    with patch("babel_ai.analyzer.cos_sim") as mock_cos_sim:
        # Mock cos_sim to return a value above 1.0
        mock_cos_sim.return_value.item.return_value = 2.3

        with caplog.at_level(logging.DEBUG):
            result = analyzer._analyze_semantic_similarity(outputs)

            # Check that warning was logged
            assert "Cosine similarity is 2.3" in caplog.text
            assert "Similarity will be clamped to [-1, 1]" in caplog.text
            assert "Clamped cosine similarity: 1.0" in caplog.text

            # Should be clamped to 1.0
            assert result == 1.0


def test_analyze_full(analyzer):
    """Test the full analysis pipeline."""
    outputs = ["The quick brown fox", "A fast brown fox", "A lazy dog"]

    result = analyzer.analyze(outputs)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.token_perplexity, float)
    assert result.token_perplexity >= 1.0
    assert result.lexical_similarity is not None
    assert result.semantic_similarity is not None
    assert result.lexical_similarity_window is not None
    assert result.semantic_similarity_window is not None


def test_analyze_empty_text(analyzer):
    """Test analysis with empty text."""
    text = ""
    result = analyzer._analyze_word_stats(text)

    assert result == (0, 0, 0)


def test_analyze_single_output(analyzer):
    """Test analysis with only one output."""
    outputs = ["The quick brown fox"]
    result = analyzer.analyze(outputs)

    assert isinstance(result, AnalysisResult)
    assert result.word_count
    assert result.unique_word_count
    assert result.coherence_score
    assert result.token_perplexity
    assert result.lexical_similarity is None
    assert result.semantic_similarity is None
    assert result.lexical_similarity_window is None
    assert result.semantic_similarity_window is None


def test_token_perplexity_basic(analyzer):
    """Test basic token perplexity analysis."""
    text = "The quick brown fox"
    result = analyzer._analyze_token_perplexity(text)

    assert result >= 1.0


def test_token_perplexity_empty_text(analyzer, caplog):
    """Test token perplexity analysis with empty text."""
    text = ""
    with caplog.at_level(logging.WARNING):
        result = analyzer._analyze_token_perplexity(text)
        assert "Input text is empty. Returning max perplexity." in caplog.text

    assert result == float("inf")


def test_token_perplexity_single_token(analyzer, caplog):
    """Test token perplexity analysis with single token input."""
    text = "hello"
    with caplog.at_level(logging.WARNING):
        result = analyzer._analyze_token_perplexity(text)
        assert (
            "Input text has only one token. Perplexity calculation "
            "requires at least two tokens. Returning max perplexity."
        ) in caplog.text

    assert result == float("inf")


def test_token_perplexity_long_text(analyzer):
    """Test token perplexity analysis with a very long input."""

    # Create a long text by repeating a sentence multiple times
    smaller_then_model_context_text = " ".join(["a"] * 100)  # 50 repetitions
    equal_to_model_context_text = " ".join(["a"] * analyzer.max_context_length)
    larger_then_model_context_text = " ".join(
        ["a"] * (analyzer.max_context_length * 2)
    )

    small_result = analyzer._analyze_token_perplexity(
        smaller_then_model_context_text
    )

    equal_result = analyzer._analyze_token_perplexity(
        equal_to_model_context_text
    )

    larger_result = analyzer._analyze_token_perplexity(
        larger_then_model_context_text
    )

    assert small_result >= 1.0
    assert equal_result >= 1.0
    assert larger_result >= 1.0

    assert pytest.approx(equal_result, rel=1e-1) == larger_result


# Additional tests for abstract class implementations


class TestAnalyzerAbstractBase:
    """Test the abstract Analyzer base class."""

    def test_analyzer_is_abstract(self):
        """Test that Analyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Analyzer()

    def test_create_analyzer_default_params(self):
        """Test creating analyzer with default parameters."""
        analyzer = Analyzer.create_analyzer(AnalyzerType.SIMILARITY)
        assert isinstance(analyzer, SimilarityAnalyzer)
        assert analyzer.analyze_window == 20  # default value

    def test_create_analyzer_with_kwargs(self):
        """Test creating analyzer with various keyword arguments."""
        analyzer = Analyzer.create_analyzer(
            AnalyzerType.SIMILARITY, analyze_window=30
        )
        assert isinstance(analyzer, SimilarityAnalyzer)
        assert analyzer.analyze_window == 30


def test_analyze_rolling_window_vs_direct(analyzer):
    """Test that rolling window and direct similarity are different."""
    outputs = [
        "This is a completely different sentence.",
        "Another totally unrelated text here.",
        "Some random words about nothing.",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy cat.",
        # Current - similar to [-2]
    ]

    result = analyzer.analyze(outputs)

    # Both should be calculated
    assert result.lexical_similarity is not None
    assert result.semantic_similarity is not None
    assert result.lexical_similarity_window is not None
    assert result.semantic_similarity_window is not None

    # Window similarity should generally be lower than direct similarity
    # since it includes comparison with more dissimilar texts
    assert result.lexical_similarity_window <= result.lexical_similarity
    assert result.semantic_similarity_window <= result.semantic_similarity


def test_analyze_window_size_one_equivalent_to_direct():
    """Test that window_size=1 gives same result as direct similarity."""
    analyzer = SimilarityAnalyzer(analyze_window=1)

    outputs = ["The quick brown fox", "A fast brown fox", "A lazy dog"]

    result = analyzer.analyze(outputs)

    # When analyze_window=1, window similarity should equal direct similarity
    assert result.lexical_similarity == result.lexical_similarity_window
    assert result.semantic_similarity == result.semantic_similarity_window
