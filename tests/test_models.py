"""Tests for the models module."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from babel_ai.models import (
    AnalysisResult,
    ExperimentConfig,
    LexicalMetrics,
    Metric,
    SemanticMetrics,
    TokenPerplexityMetrics,
    WordStats,
)


class TestWordStats:
    """Test the WordStats model."""

    def test_valid_word_stats(self):
        """Test creating a valid WordStats instance."""
        word_stats = WordStats(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        assert word_stats.word_count == 100
        assert word_stats.unique_word_count == 80
        assert word_stats.coherence_score == 0.8

    def test_invalid_coherence_score(self):
        """Test that coherence_score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            WordStats(
                word_count=100,
                unique_word_count=80,
                coherence_score=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            WordStats(
                word_count=100,
                unique_word_count=80,
                coherence_score=-0.1,  # Invalid: < 0.0
            )


class TestLexicalMetrics:
    """Test the LexicalMetrics model."""

    def test_valid_lexical_metrics(self):
        """Test creating a valid LexicalMetrics instance."""
        metrics = LexicalMetrics(
            similarity=0.75,
            is_repetitive=True,
        )
        assert metrics.similarity == 0.75
        assert metrics.is_repetitive is True

    def test_optional_similarity(self):
        """Test that similarity can be None."""
        metrics = LexicalMetrics(is_repetitive=False)
        assert metrics.similarity is None
        assert metrics.is_repetitive is False

    def test_invalid_similarity(self):
        """Test that similarity must be between 0 and 1 if provided."""
        with pytest.raises(ValidationError):
            LexicalMetrics(
                similarity=1.5,  # Invalid: > 1.0
                is_repetitive=True,
            )

        with pytest.raises(ValidationError):
            LexicalMetrics(
                similarity=-0.1,  # Invalid: < 0.0
                is_repetitive=True,
            )


class TestSemanticMetrics:
    """Test the SemanticMetrics model."""

    def test_valid_semantic_metrics(self):
        """Test creating a valid SemanticMetrics instance."""
        metrics = SemanticMetrics(
            similarity=0.75,
            is_repetitive=True,
        )
        assert metrics.similarity == 0.75
        assert metrics.is_repetitive is True

    def test_optional_similarity(self):
        """Test that similarity can be None."""
        metrics = SemanticMetrics(is_repetitive=False)
        assert metrics.similarity is None
        assert metrics.is_repetitive is False

    def test_invalid_similarity(self):
        """Test that similarity must be between -1 and 1 if provided."""
        with pytest.raises(ValidationError):
            SemanticMetrics(
                similarity=1.5,  # Invalid: > 1.0
                is_repetitive=True,
            )

        with pytest.raises(ValidationError):
            SemanticMetrics(
                similarity=-1.5,  # Invalid: < -1.0
                is_repetitive=True,
            )


class TestTokenPerplexityMetrics:
    """Test the TokenPerplexityMetrics model."""

    def test_valid_token_perplexity_metrics(self):
        """Test creating a valid TokenPerplexityMetrics instance."""
        metrics = TokenPerplexityMetrics(
            avg_token_perplexity=10.0,
        )
        assert metrics.avg_token_perplexity == 10.0

    def test_invalid_perplexity(self):
        """Test that perplexity must be >= 1."""
        with pytest.raises(ValidationError):
            TokenPerplexityMetrics(
                avg_token_perplexity=0.5,  # Invalid: < 1.0
            )


class TestAnalysisResult:
    """Test the AnalysisResult model."""

    def test_valid_analysis_result(self):
        """Test creating a valid AnalysisResult instance."""
        word_stats = WordStats(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        analysis = AnalysisResult(word_stats=word_stats)
        assert analysis.word_stats == word_stats
        assert analysis.lexical is None
        assert analysis.semantic is None
        assert analysis.token_perplexity is None

    def test_analysis_result_with_all_metrics(self):
        """Test creating an AnalysisResult with all metrics."""
        word_stats = WordStats(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        lexical = LexicalMetrics(similarity=0.75, is_repetitive=True)
        semantic = SemanticMetrics(similarity=0.65, is_repetitive=False)
        token_perplexity = TokenPerplexityMetrics(
            avg_token_perplexity=10.0,
        )
        analysis = AnalysisResult(
            word_stats=word_stats,
            lexical=lexical,
            semantic=semantic,
            token_perplexity=token_perplexity,
        )
        assert analysis.word_stats == word_stats
        assert analysis.lexical == lexical
        assert analysis.semantic == semantic
        assert analysis.token_perplexity == token_perplexity


class TestExperimentConfig:
    """Test the ExperimentConfig model."""

    def test_valid_experiment_config(self):
        """Test creating a valid ExperimentConfig instance."""
        config = ExperimentConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.top_p == 1.0
        assert config.max_iterations == 100
        assert config.max_total_characters == 1000000
        assert config.analyze_window == 20

    def test_custom_experiment_config(self):
        """Test creating a custom ExperimentConfig instance."""
        config = ExperimentConfig(
            temperature=0.9,
            max_tokens=200,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            top_p=0.9,
            max_iterations=50,
            max_total_characters=500000,
            analyze_window=10,
        )
        assert config.temperature == 0.9
        assert config.max_tokens == 200
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.5
        assert config.top_p == 0.9
        assert config.max_iterations == 50
        assert config.max_total_characters == 500000
        assert config.analyze_window == 10

    def test_invalid_temperature(self):
        """Test that temperature must be between 0 and 2."""
        with pytest.raises(ValidationError):
            ExperimentConfig(temperature=2.5)  # Invalid: > 2.0

        with pytest.raises(ValidationError):
            ExperimentConfig(temperature=-0.1)  # Invalid: < 0.0

    def test_invalid_max_tokens(self):
        """Test that max_tokens must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(max_tokens=0)  # Invalid: < 1

    def test_invalid_frequency_penalty(self):
        """Test that frequency_penalty must be between -2 and 2."""
        with pytest.raises(ValidationError):
            ExperimentConfig(frequency_penalty=2.5)  # Invalid: > 2.0

        with pytest.raises(ValidationError):
            ExperimentConfig(frequency_penalty=-2.5)  # Invalid: < -2.0

    def test_invalid_presence_penalty(self):
        """Test that presence_penalty must be between -2 and 2."""
        with pytest.raises(ValidationError):
            ExperimentConfig(presence_penalty=2.5)  # Invalid: > 2.0

        with pytest.raises(ValidationError):
            ExperimentConfig(presence_penalty=-2.5)  # Invalid: < -2.0

    def test_invalid_top_p(self):
        """Test that top_p must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(top_p=1.5)  # Invalid: > 1.0

        with pytest.raises(ValidationError):
            ExperimentConfig(top_p=-0.1)  # Invalid: < 0.0

    def test_invalid_max_iterations(self):
        """Test that max_iterations must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(max_iterations=0)  # Invalid: < 1

    def test_invalid_max_total_characters(self):
        """Test that max_total_characters must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(max_total_characters=0)  # Invalid: < 1

    def test_invalid_analyze_window(self):
        """Test that analyze_window must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(analyze_window=0)  # Invalid: < 1


class TestMetric:
    """Test the Metric model."""

    def test_valid_metric(self):
        """Test creating a valid Metric instance."""
        word_stats = WordStats(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        analysis = AnalysisResult(word_stats=word_stats)
        timestamp = datetime.now()

        metric = Metric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            response="This is a test response.",
            analysis=analysis,
        )

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "assistant"
        assert metric.response == "This is a test response."
        assert metric.analysis == analysis
        assert metric.config is None

    def test_metric_with_config(self):
        """Test creating a Metric instance with config."""
        word_stats = WordStats(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        analysis = AnalysisResult(word_stats=word_stats)
        timestamp = datetime.now()
        config = ExperimentConfig(temperature=0.9)

        metric = Metric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            response="This is a test response.",
            analysis=analysis,
            config=config,
        )

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "assistant"
        assert metric.response == "This is a test response."
        assert metric.analysis == analysis
        assert metric.config == config

    def test_metric_to_dict(self):
        """Test converting a Metric to a dictionary."""
        word_stats = WordStats(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        lexical = LexicalMetrics(similarity=0.75, is_repetitive=True)
        semantic = SemanticMetrics(similarity=0.65, is_repetitive=False)
        token_perplexity = TokenPerplexityMetrics(
            avg_token_perplexity=10.0,
        )
        analysis = AnalysisResult(
            word_stats=word_stats,
            lexical=lexical,
            semantic=semantic,
            token_perplexity=token_perplexity,
        )
        timestamp = datetime.now()
        config = ExperimentConfig(temperature=0.9)

        metric = Metric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            response="This is a test response.",
            analysis=analysis,
            config=config,
        )

        result = metric.to_dict()

        assert result["iteration"] == 1
        assert result["timestamp"] == timestamp
        assert result["role"] == "assistant"
        assert result["response"] == "This is a test response."
        assert result["word_count"] == 100
        assert result["unique_word_count"] == 80
        assert result["coherence_score"] == 0.8
        assert result["lexical_similarity"] == 0.75
        assert result["is_repetitive"] is True
        assert result["semantic_similarity"] == 0.65
        assert result["is_semantically_repetitive"] is False
        assert result["avg_token_perplexity"] == 10.0
        assert result["temperature"] == 0.9
