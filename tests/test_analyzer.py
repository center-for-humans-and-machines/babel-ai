"""Tests for the analyzer module."""

import numpy as np
import pytest

from babel_ai.analyzer import SimilarityAnalyzer
from babel_ai.models import (
    AnalysisResult,
    LexicalMetrics,
    SemanticMetrics,
    SurpriseMetrics,
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


def test_get_semantic_distribution(analyzer):
    """Test semantic distribution generation."""
    text = "The quick brown fox"
    dist = analyzer._get_semantic_distribution(text)

    assert isinstance(dist, np.ndarray)
    assert np.all(dist >= 0)
    assert np.isclose(np.sum(dist), 1.0)


def test_analyze_semantic_surprise(analyzer):
    """Test semantic surprise analysis."""
    current_text = "The quick brown fox"
    previous_texts = ["A fast brown fox", "A lazy dog"]

    result = analyzer._analyze_semantic_surprise(current_text, previous_texts)

    assert isinstance(result, SurpriseMetrics)
    assert isinstance(result.semantic_surprise, float)
    assert isinstance(result.max_semantic_surprise, float)
    assert isinstance(result.is_surprising, bool)

    assert result.semantic_surprise >= 0
    assert result.max_semantic_surprise >= 0
    assert result.is_surprising is False


def test_analyze_full(analyzer):
    """Test the full analysis pipeline."""
    outputs = ["The quick brown fox", "A fast brown fox", "A lazy dog"]

    result = analyzer.analyze(outputs)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.word_stats, WordStats)
    assert isinstance(result.lexical, LexicalMetrics)
    assert isinstance(result.semantic, SemanticMetrics)
    assert isinstance(result.surprise, SurpriseMetrics)


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
    assert result.surprise is None
    assert isinstance(result.word_stats, WordStats)
