"""Tests for the analyzer module."""

import logging

import pytest

from babel_ai.analyzer import SimilarityAnalyzer
from babel_ai.models import (
    AnalysisResult,
    LexicalMetrics,
    SemanticMetrics,
    TokenPerplexityMetrics,
    WordStats,
)


@pytest.fixture
def analyzer():
    """Create a SimilarityAnalyzer instance for testing."""
    return SimilarityAnalyzer()


def test_analyzer_init():
    """Test SimilarityAnalyzer initialization."""
    # Test default initialization
    default_analyzer = SimilarityAnalyzer()
    assert default_analyzer._model_name == "all-MiniLM-L6-v2"
    assert default_analyzer.analyze_window == 20

    # Test custom initialization
    custom_analyzer = SimilarityAnalyzer(
        semantic_model="all-mpnet-base-v2", analyze_window=10
    )
    assert custom_analyzer._model_name == "all-mpnet-base-v2"
    assert custom_analyzer.analyze_window == 10


def test_analyze_word_stats(analyzer):
    """Test word statistics analysis."""
    text = "The quick brown fox jumps over the lazy dog"
    result = analyzer._analyze_word_stats(text)

    assert isinstance(result, WordStats)
    assert result.word_count == 9
    assert result.unique_word_count == 8
    assert result.coherence_score == 8 / 9


def test_analyze_similarity(analyzer):
    """Test lexical similarity analysis."""
    # Test similar text case
    current_text = "The quick brown super good fox"
    previous_texts = ["Other sentence", "The quick brown super fox"]

    result = analyzer._analyze_similarity(current_text, previous_texts)

    assert isinstance(result, LexicalMetrics)
    assert result.similarity == 5 / 6  # 5 intersection words and 6 union words
    assert result.is_repetitive is True

    # Test non-similar text case
    current_text = "The quick brown fox"
    previous_texts = ["Other sentence", "The unsimilar fox"]

    result = analyzer._analyze_similarity(current_text, previous_texts)

    assert isinstance(result, LexicalMetrics)
    assert result.similarity == 2 / 5  # 2 intersection words and 5 union words
    assert result.is_repetitive is False


def test_analyze_semantic_similarity(analyzer):
    """Test semantic similarity analysis."""
    # Test similar but not identical text
    current_text = "The quick brown fox"
    previous_texts = ["A fast brown fox", "A slow not brown dog"]

    result = analyzer._analyze_semantic_similarity(
        current_text, previous_texts
    )

    assert isinstance(result, SemanticMetrics)
    assert isinstance(result.similarity, float)
    assert -1.0 <= result.similarity <= 1.0  # Ensure within valid range
    assert result.is_repetitive is False

    # Test identical text
    current_text = "The quick brown fox"
    previous_texts = ["A lazy dog", "The quick brown fox"]

    result = analyzer._analyze_semantic_similarity(
        current_text, previous_texts
    )

    assert isinstance(result, SemanticMetrics)
    assert result.similarity == 1.0
    assert result.is_repetitive is True


def test_analyze_full(analyzer):
    """Test the full analysis pipeline."""
    outputs = ["The quick brown fox", "A fast brown fox", "A lazy dog"]

    result = analyzer.analyze(outputs)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.word_stats, WordStats)
    assert isinstance(result.lexical, LexicalMetrics)
    assert isinstance(result.semantic, SemanticMetrics)
    assert isinstance(result.token_perplexity, TokenPerplexityMetrics)
    assert isinstance(result.token_perplexity.avg_token_perplexity, float)
    assert result.token_perplexity.avg_token_perplexity >= 1.0


def test_analyze_empty_text(analyzer):
    """Test analysis with empty text."""
    text = ""
    result = analyzer._analyze_word_stats(text)

    assert result.word_count == 0
    assert result.unique_word_count == 0
    assert result.coherence_score == 0.0


def test_analyze_single_output(analyzer):
    """Test analysis with only one output."""
    outputs = ["The quick brown fox"]
    result = analyzer.analyze(outputs)

    assert isinstance(result, AnalysisResult)
    assert result.lexical is None
    assert result.semantic is None
    assert isinstance(result.word_stats, WordStats)
    assert isinstance(result.token_perplexity, TokenPerplexityMetrics)
    assert isinstance(result.token_perplexity.avg_token_perplexity, float)
    assert result.token_perplexity.avg_token_perplexity >= 1.0


def test_token_perplexity_basic(analyzer):
    """Test basic token perplexity analysis."""
    text = "The quick brown fox"
    result = analyzer._analyze_token_perplexity(text)

    assert isinstance(result, TokenPerplexityMetrics)
    assert isinstance(result.avg_token_perplexity, float)
    assert result.avg_token_perplexity >= 1.0


def test_token_perplexity_empty_text(analyzer, caplog):
    """Test token perplexity analysis with empty text."""
    text = ""
    with caplog.at_level(logging.WARNING):
        result = analyzer._analyze_token_perplexity(text)
        assert "Input text is empty. Returning max perplexity." in caplog.text

    assert isinstance(result, TokenPerplexityMetrics)
    assert result.avg_token_perplexity == float("inf")


def test_token_perplexity_single_token(analyzer, caplog):
    """Test token perplexity analysis with single token input."""
    text = "hello"
    with caplog.at_level(logging.WARNING):
        result = analyzer._analyze_token_perplexity(text)
        assert (
            "Input text has only one token. Perplexity calculation "
            "requires at least two tokens. Returning max perplexity."
        ) in caplog.text

    assert isinstance(result, TokenPerplexityMetrics)
    assert result.avg_token_perplexity == float("inf")


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

    assert small_result.avg_token_perplexity >= 1.0
    assert equal_result.avg_token_perplexity >= 1.0
    assert larger_result.avg_token_perplexity >= 1.0

    assert (
        equal_result.avg_token_perplexity == larger_result.avg_token_perplexity
    )
