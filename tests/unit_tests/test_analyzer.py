"""Tests for the analyzer module."""

import logging

import pytest

from babel_ai.analyzer import Analyzer, SimilarityAnalyzer
from babel_ai.enums import AnalyzerType
from babel_ai.models import AnalysisResult


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
    # Test similar text case
    outputs = [
        "Other sentence",
        "The quick brown super fox",
        "The quick brown super good fox",
    ]

    result = analyzer._analyze_lexical_similarity(outputs)

    assert result == 5 / 6  # 5 intersection words and 6 union words

    # Test non-similar text case
    outputs = ["Other sentence", "The unsimilar fox", "The quick brown fox"]

    result = analyzer._analyze_lexical_similarity(outputs)

    assert result == 2 / 5  # 2 intersection words and 5 union words


def test_analyze_semantic_similarity(analyzer):
    """Test semantic similarity analysis."""
    # Test similar but not identical text
    outputs = [
        "A fast brown fox",
        "A slow not brown dog",
        "The quick brown fox",
    ]

    result = analyzer._analyze_semantic_similarity(outputs)
    assert -1.0 <= result <= 1.0  # Ensure within valid range

    # Test identical text
    outputs = ["A lazy dog", "The quick brown fox", "The quick brown fox"]

    result = analyzer._analyze_semantic_similarity(outputs)
    assert result == 1.0


def test_analyze_full(analyzer):
    """Test the full analysis pipeline."""
    outputs = ["The quick brown fox", "A fast brown fox", "A lazy dog"]

    result = analyzer.analyze(outputs)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.token_perplexity, float)
    assert result.token_perplexity >= 1.0


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
