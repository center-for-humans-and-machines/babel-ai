"""Tests for the models module."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from api.enums import OpenAIModels, Provider
from babel_ai.enums import AgentSelectionMethod, AnalyzerType, FetcherType
from models import (
    AgentConfig,
    AgentMetric,
    AnalysisResult,
    AnalyzerConfig,
    ExperimentConfig,
    ExperimentMetadata,
    FetcherConfig,
    FetcherMetric,
    Metric,
)
from models.api import LLMResponse


class TestAnalysisResult:
    """Test the AnalysisResult model."""

    def test_valid_analysis_result_minimal(self):
        """Test creating a valid AnalysisResult with minimal fields."""
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
        assert analysis.lexical_similarity_window is None
        assert analysis.semantic_similarity_window is None
        assert analysis.token_perplexity is None

    def test_valid_analysis_result_complete(self):
        """Test creating a valid AnalysisResult with all fields."""
        analysis = AnalysisResult(
            word_count=100,
            unique_word_count=80,
            coherence_score=0.8,
            lexical_similarity=0.75,
            semantic_similarity=0.65,
            lexical_similarity_window=0.70,
            semantic_similarity_window=0.60,
            token_perplexity=10.0,
        )
        assert analysis.word_count == 100
        assert analysis.unique_word_count == 80
        assert analysis.coherence_score == 0.8
        assert analysis.lexical_similarity == 0.75
        assert analysis.semantic_similarity == 0.65
        assert analysis.lexical_similarity_window == 0.70
        assert analysis.semantic_similarity_window == 0.60
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

    def test_invalid_lexical_similarity_window(self):
        """Test that lexical_similarity_window must be between 0 and 1."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                lexical_similarity_window=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                lexical_similarity_window=-0.1,  # Invalid: < 0.0
            )

    def test_invalid_semantic_similarity_window(self):
        """Test that semantic_similarity_window must be between -1 and 1."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                semantic_similarity_window=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            AnalysisResult(
                word_count=100,
                unique_word_count=80,
                coherence_score=0.8,
                semantic_similarity_window=-1.5,  # Invalid: < -1.0
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


class TestAnalyzerConfig:
    """Test the AnalyzerConfig model."""

    def test_valid_analyzer_config(self):
        """Test creating a valid AnalyzerConfig."""
        config = AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=10
        )
        assert config.analyzer == AnalyzerType.SIMILARITY
        assert config.analyze_window == 10

    def test_invalid_analyze_window(self):
        """Test that analyze_window validation works correctly."""
        # Note: The current model doesn't have explicit validation
        # but we can test basic functionality
        with pytest.raises(ValidationError):
            AnalyzerConfig(analyzer=AnalyzerType.SIMILARITY, analyze_window=0)

    def test_invalid_analyzer_type(self):
        """Test that invalid analyzer type is rejected."""
        with pytest.raises(ValidationError):
            AnalyzerConfig(analyzer="invalid_analyzer_type", analyze_window=5)


class TestFetcherConfig:
    """Test the FetcherConfig model."""

    def test_valid_fetcher_config(self):
        """Test creating a valid FetcherConfig."""
        config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        assert config.fetcher == FetcherType.SHAREGPT
        assert config.data_path == "/path/to/data.json"
        assert config.min_messages == 2
        assert config.max_messages == 10

    def test_fetcher_config_edge_cases(self):
        """Test FetcherConfig with edge case values."""
        config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="",
            min_messages=1,
            max_messages=1,
        )
        assert config.fetcher == FetcherType.SHAREGPT
        assert config.data_path == ""
        assert config.min_messages == 1
        assert config.max_messages == 1

    def test_invalid_min_messages(self):
        """Test that min_messages must be >= 1."""
        with pytest.raises(ValidationError):
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="",
                min_messages=0,
                max_messages=1,
            )

    def test_invalid_max_messages(self):
        """Test that max_messages must be >= min_messages."""
        with pytest.raises(ValidationError):
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="",
                min_messages=0,
                max_messages=0,
            )

        with pytest.raises(ValidationError):
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="",
                min_messages=2,
                max_messages=1,
            )

    def test_invalid_fetcher_type(self):
        """Test that invalid fetcher type is rejected."""
        with pytest.raises(ValidationError):
            FetcherConfig(
                fetcher="invalid_fetcher_type",
                data_path="/path/to/data.json",
                min_messages=2,
                max_messages=10,
            )

    # New tests for updated validation logic
    def test_valid_category_values(self):
        """Test that valid category values are accepted."""
        valid_categories = ["creative", "analytical", "conversational"]

        for category in valid_categories:
            config = FetcherConfig(
                fetcher=FetcherType.RANDOM, category=category
            )
            assert config.category == category

    def test_invalid_category_values(self):
        """Test that invalid category values are rejected."""
        invalid_categories = ["invalid", "wrong", "bad_category", "", None]

        for category in invalid_categories:
            with pytest.raises(ValidationError):
                FetcherConfig(fetcher=FetcherType.RANDOM, category=category)

    def test_max_messages_validation_edge_cases(self):
        """Test max_messages validation edge cases."""
        # Equal values should work
        config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=5,
            max_messages=5,
        )
        assert config.min_messages == 5
        assert config.max_messages == 5

    def test_sharegpt_fetcher_valid_params(self):
        """Test SHAREGPT fetcher with valid parameters."""
        config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        assert config.fetcher == FetcherType.SHAREGPT
        assert config.data_path == "/path/to/data.json"
        assert config.min_messages == 2
        assert config.max_messages == 10

    def test_sharegpt_fetcher_missing_required_params(self):
        """Test SHAREGPT fetcher with missing required parameters."""
        # Missing data_path
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT, min_messages=2, max_messages=10
            )
        assert "Parameter data_path is required for fetcher type" in str(
            exc_info.value
        )

        # Missing min_messages
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="/path/to/data.json",
                max_messages=10,
            )
        assert "Parameter min_messages is required for fetcher type" in str(
            exc_info.value
        )

        # Missing max_messages
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="/path/to/data.json",
                min_messages=2,
            )
        assert "Parameter max_messages is required for fetcher type" in str(
            exc_info.value
        )

    def test_sharegpt_fetcher_invalid_params(self):
        """Test SHAREGPT fetcher with invalid parameters."""
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="/path/to/data.json",
                min_messages=2,
                max_messages=10,
                category="creative",  # Not allowed for SHAREGPT
            )
        assert "Parameter category is not allowed for fetcher type" in str(
            exc_info.value
        )

    def test_infinite_conversation_fetcher_valid_params(self):
        """Test INFINITE_CONVERSATION fetcher with valid parameters."""
        config = FetcherConfig(
            fetcher=FetcherType.INFINITE_CONVERSATION,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        assert config.fetcher == FetcherType.INFINITE_CONVERSATION
        assert config.data_path == "/path/to/data.json"
        assert config.min_messages == 2
        assert config.max_messages == 10

    def test_infinite_conversation_fetcher_missing_required_params(self):
        """
        Test INFINITE_CONVERSATION fetcher
        with missing required parameters.
        """
        # Missing data_path
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.INFINITE_CONVERSATION,
                min_messages=2,
                max_messages=10,
            )
        assert "Parameter data_path is required for fetcher type" in str(
            exc_info.value
        )

    def test_infinite_conversation_fetcher_invalid_params(self):
        """Test INFINITE_CONVERSATION fetcher with invalid parameters."""
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.INFINITE_CONVERSATION,
                data_path="/path/to/data.json",
                min_messages=2,
                max_messages=10,
                second_data_path="/another/path",
            )
        assert (
            "Parameter second_data_path is not allowed for fetcher type"
            in str(exc_info.value)
        )

    def test_topical_chat_fetcher_valid_params(self):
        """Test TOPICAL_CHAT fetcher with valid parameters."""
        config = FetcherConfig(
            fetcher=FetcherType.TOPICAL_CHAT,
            data_path="/path/to/test_rare.jsonl",
            second_data_path="/path/to/test_freq.jsonl",
            min_messages=2,
            max_messages=10,
        )
        assert config.fetcher == FetcherType.TOPICAL_CHAT
        assert config.data_path == "/path/to/test_rare.jsonl"
        assert config.second_data_path == "/path/to/test_freq.jsonl"
        assert config.min_messages == 2
        assert config.max_messages == 10

    def test_topical_chat_fetcher_missing_required_params(self):
        """Test TOPICAL_CHAT fetcher with missing required parameters."""
        # Missing data_path
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.TOPICAL_CHAT,
                second_data_path="/path/to/test_freq.jsonl",
                min_messages=2,
                max_messages=10,
            )
        assert "Parameter data_path is required for fetcher type" in str(
            exc_info.value
        )

        # Missing second_data_path
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.TOPICAL_CHAT,
                data_path="/path/to/test_rare.jsonl",
                min_messages=2,
                max_messages=10,
            )
        assert (
            "Parameter second_data_path is required for fetcher type"
            in str(exc_info.value)
        )

    def test_topical_chat_fetcher_invalid_params(self):
        """Test TOPICAL_CHAT fetcher with invalid parameters."""
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.TOPICAL_CHAT,
                data_path="/path/to/test_rare.jsonl",
                second_data_path="/path/to/test_freq.jsonl",
                min_messages=2,
                max_messages=10,
                category="creative",  # Not allowed for TOPICAL_CHAT
            )
        assert "Parameter category is not allowed for fetcher type" in str(
            exc_info.value
        )

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are caught."""
        # Test with both invalid category and missing required parameter
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.RANDOM, category="invalid_category"
            )
        # Should fail on category validation first
        assert "category must be one of" in str(exc_info.value)

    def test_fetcher_parameter_validation_with_none_values(self):
        """Test fetcher parameter validation handles None values correctly."""
        # This should work - None values are treated as missing
        config = FetcherConfig(
            fetcher=FetcherType.RANDOM,
            category="creative",
            data_path=None,  # None is fine for non-required params
            min_messages=None,
            max_messages=None,
        )
        assert config.category == "creative"
        assert config.data_path is None
        assert config.min_messages is None
        assert config.max_messages is None

    def test_fetcher_parameter_validation_order(self):
        """
        Test that fetcher parameter validation
        runs after field validation.
        """
        # This should fail on field validation first (invalid category)
        # before getting to fetcher parameter validation
        with pytest.raises(ValidationError) as exc_info:
            FetcherConfig(
                fetcher=FetcherType.RANDOM, category="invalid_category"
            )
        assert "category must be one of" in str(exc_info.value)
        # Should not contain fetcher parameter validation message
        assert "Parameter category is required for fetcher type" not in str(
            exc_info.value
        )


class TestAgentConfig:
    """Test the AgentConfig model."""

    def test_valid_agent_config_minimal(self):
        """Test creating a valid AgentConfig with minimal fields."""
        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
        )
        assert config.provider == Provider.OPENAI
        assert config.model == OpenAIModels.GPT4_1106_PREVIEW
        assert config.system_prompt is None
        assert config.temperature == 1.0
        assert config.max_tokens is None
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.top_p == 1.0

    def test_valid_agent_config_complete(self):
        """Test creating a valid AgentConfig with all fields."""
        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            system_prompt="You are a helpful assistant",
            temperature=0.8,
            max_tokens=150,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            top_p=0.9,
        )
        assert config.provider == Provider.OPENAI
        assert config.model == OpenAIModels.GPT4_1106_PREVIEW
        assert config.system_prompt == "You are a helpful assistant"
        assert config.temperature == 0.8
        assert config.max_tokens == 150
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.top_p == 0.9

    def test_model_provider_compatibility_validation(self):
        """Test that model-provider compatibility validation works."""
        from api.enums import OllamaModels

        # This should raise a ValidationError because OllamaModel is not
        # compatible with OpenAI provider
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(
                provider=Provider.OPENAI,
                model=OllamaModels.MISTRAL_7B,  # Incompatible model
            )

        # Check that the error message is informative
        error_msg = str(exc_info.value)
        assert "not compatible with provider" in error_msg
        assert "OpenAI" in error_msg

    def test_invalid_temperature(self):
        """Test that temperature validation works correctly."""
        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                temperature=2.5,  # Invalid: > 2.0
            )

        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                temperature=-0.1,  # Invalid: < 0.0
            )

    def test_invalid_max_tokens(self):
        """Test that max_tokens validation works correctly."""
        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                max_tokens=0,  # Invalid: < 1
            )

    def test_invalid_frequency_penalty(self):
        """Test that frequency_penalty validation works correctly."""
        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                frequency_penalty=2.5,  # Invalid: > 2.0
            )

        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                frequency_penalty=-2.5,  # Invalid: < -2.0
            )

    def test_invalid_presence_penalty(self):
        """Test that presence_penalty validation works correctly."""
        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                presence_penalty=2.5,  # Invalid: > 2.0
            )

        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                presence_penalty=-2.5,  # Invalid: < -2.0
            )

    def test_invalid_top_p(self):
        """Test that top_p validation works correctly."""
        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                top_p=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                top_p=-0.1,  # Invalid: < 0.0
            )


class TestExperimentConfig:
    """Test the ExperimentConfig model."""

    def test_valid_experiment_config(self):
        """Test creating a valid ExperimentConfig."""
        fetcher_config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        analyzer_config = AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=5
        )
        agent_config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
        )

        config = ExperimentConfig(
            fetcher_config=fetcher_config,
            analyzer_config=analyzer_config,
            agent_configs=[agent_config],
            agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
            max_iterations=50,
            max_total_characters=500000,
        )

        assert config.fetcher_config == fetcher_config
        assert config.fetcher_config.fetcher == FetcherType.SHAREGPT
        assert config.analyzer_config == analyzer_config
        assert config.analyzer_config.analyzer == AnalyzerType.SIMILARITY
        assert config.agent_configs == [agent_config]
        assert config.agent_selection_method == (
            AgentSelectionMethod.ROUND_ROBIN
        )
        assert config.max_iterations == 50
        assert config.max_total_characters == 500000

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig default values."""
        fetcher_config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        analyzer_config = AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=5
        )
        agent_config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
        )

        config = ExperimentConfig(
            fetcher_config=fetcher_config,
            analyzer_config=analyzer_config,
            agent_configs=[agent_config],
            agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        )

        assert config.max_iterations == 100  # default
        assert config.max_total_characters == 1000000  # default

    def test_invalid_max_iterations(self):
        """Test that max_iterations must be >= 1."""
        fetcher_config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        analyzer_config = AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=5
        )
        agent_config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
        )

        with pytest.raises(ValidationError):
            ExperimentConfig(
                fetcher_config=fetcher_config,
                analyzer_config=analyzer_config,
                agent_configs=[agent_config],
                agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
                max_iterations=0,
            )

    def test_invalid_max_total_characters(self):
        """Test that max_total_characters must be >= 1."""
        fetcher_config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        analyzer_config = AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=5
        )
        agent_config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
        )

        with pytest.raises(ValidationError):
            ExperimentConfig(
                fetcher_config=fetcher_config,
                analyzer_config=analyzer_config,
                agent_configs=[agent_config],
                agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
                max_total_characters=0,
            )


class TestMetric:
    """Test the Metric model."""

    def test_valid_metric_minimal(self):
        """Test creating a valid Metric with minimal fields."""
        timestamp = datetime.now()
        metric = Metric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            content="This is a test response.",
        )

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "assistant"
        assert metric.content == "This is a test response."
        assert metric.analysis is None

    def test_valid_metric_with_analysis(self):
        """Test creating a valid Metric with analysis."""
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
            content="This is a test response.",
            analysis=analysis,
        )

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "assistant"
        assert metric.content == "This is a test response."
        assert metric.analysis == analysis

    def test_metric_to_dict(self):
        """Test Metric to_dict method."""
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
            content="This is a test response.",
            analysis=analysis,
        )

        result = metric.to_dict()

        assert result["iteration"] == 1
        assert result["timestamp"] == timestamp
        assert result["role"] == "assistant"
        assert result["content"] == "This is a test response."
        assert result["analysis"] == {
            "word_count": 100,
            "unique_word_count": 80,
            "coherence_score": 0.8,
            "lexical_similarity": None,
            "semantic_similarity": None,
            "lexical_similarity_window": None,
            "semantic_similarity_window": None,
            "token_perplexity": None,
        }

    def test_metric_to_dict_no_analysis(self):
        """Test Metric to_dict method without analysis."""
        timestamp = datetime.now()

        metric = Metric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            content="This is a test response.",
        )

        result = metric.to_dict()

        assert result["iteration"] == 1
        assert result["timestamp"] == timestamp
        assert result["role"] == "assistant"
        assert result["content"] == "This is a test response."
        assert result["analysis"] is None


class TestFetcherMetric:
    """Test the FetcherMetric model."""

    def test_valid_fetcher_metric(self):
        """Test creating a valid FetcherMetric."""
        fetcher_config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        timestamp = datetime.now()

        metric = FetcherMetric(
            iteration=1,
            timestamp=timestamp,
            role="user",
            content="This is a user message.",
            fetcher_type=FetcherType.SHAREGPT,
            fetcher_config=fetcher_config,
        )

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "user"
        assert metric.content == "This is a user message."
        assert metric.fetcher_config == fetcher_config

    def test_fetcher_metric_to_dict(self):
        """Test FetcherMetric to_dict method."""
        fetcher_config = FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="/path/to/data.json",
            min_messages=2,
            max_messages=10,
        )
        timestamp = datetime.now()

        metric = FetcherMetric(
            iteration=1,
            timestamp=timestamp,
            role="user",
            content="This is a user message.",
            fetcher_config=fetcher_config,
        )

        result = metric.to_dict()

        assert result["iteration"] == 1
        assert result["timestamp"] == timestamp
        assert result["role"] == "user"
        assert result["content"] == "This is a user message."
        assert result["fetcher_config"] == {
            "fetcher": FetcherType.SHAREGPT,
            "data_path": "/path/to/data.json",
            "second_data_path": None,
            "category": None,
            "min_messages": 2,
            "max_messages": 10,
        }


class TestAgentMetric:
    """Test the AgentMetric model."""

    def test_valid_agent_metric(self):
        """Test creating a valid AgentMetric."""
        agent_config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            temperature=0.8,
        )
        analysis = AnalysisResult(
            word_count=50,
            unique_word_count=45,
            coherence_score=0.9,
        )
        timestamp = datetime.now()

        metric = AgentMetric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            content="This is an agent response.",
            agent_id="agent_1",
            agent_config=agent_config,
            analysis=analysis,
        )

        assert metric.iteration == 1
        assert metric.timestamp == timestamp
        assert metric.role == "assistant"
        assert metric.content == "This is an agent response."
        assert metric.agent_id == "agent_1"
        assert metric.agent_config == agent_config
        assert metric.analysis == analysis

    def test_agent_metric_to_dict(self):
        """Test AgentMetric to_dict method."""
        agent_config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            temperature=0.8,
        )
        timestamp = datetime.now()

        metric = AgentMetric(
            iteration=1,
            timestamp=timestamp,
            role="assistant",
            content="This is an agent response.",
            agent_id="agent_1",
            agent_config=agent_config,
        )

        result = metric.to_dict()

        assert result["iteration"] == 1
        assert result["timestamp"] == timestamp
        assert result["role"] == "assistant"
        assert result["content"] == "This is an agent response."
        assert result["agent_id"] == "agent_1"
        assert result["agent_config"] == {
            "provider": Provider.OPENAI,
            "model": OpenAIModels.GPT4_1106_PREVIEW,
            "system_prompt": None,
            "temperature": 0.8,
            "max_tokens": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "top_p": 1.0,
        }


class TestExperimentMetadata:
    """Test the ExperimentMetadata model."""

    @pytest.fixture
    def mock_config(self):
        """Mock ExperimentConfig for testing."""
        return ExperimentConfig(
            fetcher_config=FetcherConfig(
                fetcher=FetcherType.SHAREGPT,
                data_path="/test/data.json",
                min_messages=2,
                max_messages=10,
            ),
            analyzer_config=AnalyzerConfig(
                analyzer=AnalyzerType.SIMILARITY, analyze_window=5
            ),
            agent_configs=[
                AgentConfig(
                    provider=Provider.OPENAI,
                    model=OpenAIModels.GPT4_1106_PREVIEW,
                )
            ],
            agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        )

    def test_experiment_metadata_minimal(self, mock_config):
        """Test creating ExperimentMetadata with minimal fields."""
        timestamp = datetime.now()
        metadata = ExperimentMetadata(
            timestamp=timestamp,
            config=mock_config,
        )

        assert metadata.timestamp == timestamp
        assert metadata.config == mock_config
        assert metadata.num_iterations_total is None
        assert metadata.num_fetcher_messages is None
        assert metadata.total_characters is None

    def test_experiment_metadata_complete(self, mock_config):
        """Test creating ExperimentMetadata with all fields."""
        timestamp = datetime.now()
        metadata = ExperimentMetadata(
            timestamp=timestamp,
            config=mock_config,
            num_iterations_total=50,
            num_fetcher_messages=3,
            total_characters=1000,
        )

        assert metadata.timestamp == timestamp
        assert metadata.config == mock_config
        assert metadata.num_iterations_total == 50
        assert metadata.num_fetcher_messages == 3
        assert metadata.total_characters == 1000


class TestLLMResponse:
    """Test cases for LLMResponse model."""

    def test_valid_llm_response(self):
        """Test creating a valid LLMResponse with various valid inputs."""
        response = LLMResponse(
            content="Test response content",
            input_token_count=25,
            output_token_count=10,
        )

        assert response.content == "Test response content"
        assert response.input_token_count == 25
        assert response.output_token_count == 10

    def test_llm_response_validation_errors(self):
        """Test validation errors for invalid inputs."""
        # Missing required fields
        with pytest.raises(ValidationError):
            LLMResponse(input_token_count=10, output_token_count=5)

        # Negative token counts
        with pytest.raises(ValidationError):
            LLMResponse(
                content="Test", input_token_count=-1, output_token_count=5
            )
