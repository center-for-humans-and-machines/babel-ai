"""Tests for the models module."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from babel_ai.models import AnalysisResult, ExperimentConfig, Metric


class TestAnalysisResult:
    """Test the AnalysisResult model."""

    def test_valid_analysis_result_minimal(self):
        """Test creating a valid AnalysisResult
        with minimal required fields."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        assert analysis.word_count == 100
        assert analysis.unique_word_count == 80
        assert analysis.coherence_score == 0.8
        assert analysis.lexical_similarity is None
        assert analysis.semantic_similarity is None
        assert analysis.token_perplexity is None

    def test_valid_analysis_result_complete(self):
        """Test creating a valid AnalysisResult with all fields."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
            lexical_similarity=0.75,
            semantic_similarity=0.65,
            token_perplexity=10.0,
        )
        assert analysis.word_count == 100
        assert analysis.unique_word_count == 80
        assert analysis.coherence_score == 0.8
        assert analysis.lexical_similarity == 0.75
        assert analysis.semantic_similarity == 0.65
        assert analysis.token_perplexity == 10.0

    def test_invalid_coherence_score(self):
        """Test that coherence_score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=-0.1,  # Invalid: < 0.0
            )

    def test_invalid_lexical_similarity(self):
        """Test that lexical_similarity must be between 0 and 1."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                lexical_similarity=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                lexical_similarity=-0.1,  # Invalid: < 0.0
            )

    def test_invalid_semantic_similarity(self):
        """Test that semantic_similarity must be between -1 and 1."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                semantic_similarity=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                semantic_similarity=-1.5,  # Invalid: < -1.0
            )

    def test_invalid_token_perplexity(self):
        """Test that token_perplexity must be >= 1.0."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                token_perplexity=0.5,  # Invalid: < 1.0
            )


class TestExperimentConfig:
    """Test the ExperimentConfig model."""

    def test_default_config(self):
        """Test creating an ExperimentConfig with default values."""
        config = ExperimentConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.top_p == 1.0
        assert config.max_iterations == 100
        assert config.max_total_characters == 1000000
        assert config.analyze_window == 20

    def test_custom_config(self):
        """Test creating an ExperimentConfig with custom values."""
        config = ExperimentConfig(
            temperature=0.9,
            max_tokens=200,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            top_p=0.9,
            max_iterations=50,
            max_total_characters=500000,
            analyze_window=10,
        )
        assert config.temperature == 0.9
        assert config.max_tokens == 200
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.1
        assert config.top_p == 0.9
        assert config.max_iterations == 50
        assert config.max_total_characters == 500000
        assert config.analyze_window == 10

    def test_invalid_temperature(self):
        """Test that temperature must be between 0 and 2."""
        with pytest.raises(ValidationError):
            ExperimentConfig(temperature=2.5)

        with pytest.raises(ValidationError):
            ExperimentConfig(temperature=-0.1)

    def test_invalid_max_tokens(self):
        """Test that max_tokens must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(max_tokens=0)

    def test_invalid_frequency_penalty(self):
        """Test that frequency_penalty must be between -2 and 2."""
        with pytest.raises(ValidationError):
            ExperimentConfig(frequency_penalty=2.5)

        with pytest.raises(ValidationError):
            ExperimentConfig(frequency_penalty=-2.5)

    def test_invalid_presence_penalty(self):
        """Test that presence_penalty must be between -2 and 2."""
        with pytest.raises(ValidationError):
            ExperimentConfig(presence_penalty=2.5)

        with pytest.raises(ValidationError):
            ExperimentConfig(presence_penalty=-2.5)

    def test_invalid_top_p(self):
        """Test that top_p must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(top_p=1.5)

        with pytest.raises(ValidationError):
            ExperimentConfig(top_p=-0.1)

    def test_invalid_max_iterations(self):
        """Test that max_iterations must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(max_iterations=0)

    def test_invalid_max_total_characters(self):
        """Test that max_total_characters must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(max_total_characters=0)

    def test_invalid_analyze_window(self):
        """Test that analyze_window must be >= 1."""
        with pytest.raises(ValidationError):
            ExperimentConfig(analyze_window=0)


class TestMetric:
    """Test the Metric model."""

    def test_valid_metric(self):
        """Test creating a valid Metric instance."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
            lexical_similarity=0.75,
            semantic_similarity=0.65,
            token_perplexity=10.0,
        )
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
        """Test creating a Metric with config."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
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

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "assistant"
        assert metric.response == "This is a test response."
        assert metric.analysis == analysis
        assert metric.config == config

    def test_metric_to_dict(self):
        """Test converting a Metric to a dictionary."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
            lexical_similarity=0.75,
            semantic_similarity=0.65,
            token_perplexity=10.0,
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
        assert result["semantic_similarity"] == 0.65
        assert result["token_perplexity"] == 10.0
        assert result["temperature"] == 0.9

    def test_metric_to_dict_minimal_analysis(self):
        """Test converting a Metric with minimal analysis to dict."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
        )
        timestamp = datetime.now()

        metric = Metric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            response="This is a test response.",
            analysis=analysis,
        )

        result = metric.to_dict()

        assert result["iteration"] == 1
        assert result["word_count"] == 100
        assert result["unique_word_count"] == 80
        assert result["coherence_score"] == 0.8
        assert result["lexical_similarity"] is None
        assert result["semantic_similarity"] is None
        assert result["token_perplexity"] is None
