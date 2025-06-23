"""Tests for the DriftExperiment class."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from babel_ai.analyzer import SimilarityAnalyzer
from babel_ai.llm_drift import DriftExperiment
from babel_ai.models import AnalysisResult, ExperimentConfig, Metric
from babel_ai.prompt_fetcher import RandomPromptFetcher


@pytest.fixture
def mock_analyzer():
    """Create a mock SimilarityAnalyzer instance."""
    analyzer = MagicMock(spec=SimilarityAnalyzer)
    analyzer.analyze.return_value = AnalysisResult(
        word_count=10,
        unique_word_count=8,
        coherence_score=0.8,
        lexical_similarity=0.7,
        semantic_similarity=0.6,
        token_perplexity=10.0,
    )
    return analyzer


@pytest.fixture
def mock_prompt_fetcher():
    """Create a mock PromptFetcher instance."""
    fetcher = MagicMock(spec=RandomPromptFetcher)
    fetcher.get_random_prompt.return_value = "Test prompt"
    return fetcher


@pytest.fixture
def experiment_config():
    """Create a test experiment configuration."""
    return ExperimentConfig(
        provider="openai",
        model="gpt-4o",
        max_iterations=5,
        max_total_characters=1000,
        temperature=0.7,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=1.0,
        analyze_window=10,
    )


@pytest.fixture
def drift_experiment(mock_analyzer, mock_prompt_fetcher, experiment_config):
    """Create a DriftExperiment instance with mocked dependencies."""
    return DriftExperiment(
        analyzer=mock_analyzer,
        prompt_fetcher=mock_prompt_fetcher,
        config=experiment_config,
    )


def test_drift_experiment_init(
    drift_experiment,
    mock_analyzer,
    mock_prompt_fetcher,
    experiment_config,
):
    """Test DriftExperiment initialization."""
    assert drift_experiment.analyzer == mock_analyzer
    assert drift_experiment.prompt_fetcher == mock_prompt_fetcher
    assert drift_experiment.config == experiment_config
    assert isinstance(drift_experiment.results, list)
    assert len(drift_experiment.results) == 0
    assert isinstance(drift_experiment.messages, list)
    assert len(drift_experiment.messages) == 0


def test_drift_experiment_init_with_deprecated_llm_provider(
    mock_analyzer, mock_prompt_fetcher, experiment_config
):
    """Test DriftExperiment initialization with deprecated llm_provider."""
    mock_llm = MagicMock()

    with patch("babel_ai.llm_drift.logger") as mock_logger:
        DriftExperiment(
            llm_provider=mock_llm,
            analyzer=mock_analyzer,
            prompt_fetcher=mock_prompt_fetcher,
            config=experiment_config,
        )

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "llm_provider parameter is deprecated and will be ignored"
        )


def test_validate_messages_valid(drift_experiment):
    """Test message validation with valid messages."""
    valid_messages = [
        {"role": "system", "content": "Test system message"},
        {"role": "user", "content": "Test user message"},
        {"role": "assistant", "content": "Test assistant message"},
    ]
    drift_experiment._validate_messages(valid_messages)  # Should not raise


def test_validate_messages_invalid(drift_experiment):
    """Test message validation with invalid messages."""
    invalid_messages = [{"role": "invalid_role", "content": "Test message"}]
    with pytest.raises(ValueError):
        drift_experiment._validate_messages(invalid_messages)


@patch("babel_ai.llm_drift.generate_response")
@patch("babel_ai.llm_drift.datetime")
def test_run_experiment(
    mock_datetime,
    mock_generate_response,
    drift_experiment,
    mock_analyzer,
    tmp_path,
):
    """Test running the drift experiment."""
    # Setup
    mock_datetime.now.return_value = datetime(2024, 1, 1)
    initial_messages = [
        {"role": "system", "content": "Test system message"},
        {"role": "user", "content": "Test user message"},
    ]
    mock_generate_response.return_value = "Test response"

    # Run experiment within tmp_path
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        results = drift_experiment.run(initial_messages)

    # Verify results
    assert len(results) > 0
    assert all(isinstance(result, Metric) for result in results)
    assert (
        mock_generate_response.call_count
        == drift_experiment.config.max_iterations
    )
    assert mock_analyzer.analyze.call_count == len(results)


@patch("babel_ai.llm_drift.datetime")
def test_save_results_to_csv(
    mock_datetime, drift_experiment, mock_analyzer, tmp_path
):
    """Test saving experiment results to CSV."""
    # Setup
    mock_datetime.now.return_value = datetime(2024, 1, 1)
    metrics = [
        Metric(
            iteration=0,
            timestamp=datetime(2024, 1, 1),
            role="assistant",
            response="Test response",
            analysis=mock_analyzer.analyze.return_value,
            config=drift_experiment.config,
        )
    ]
    timestamp = datetime(2024, 1, 1)

    # Call method
    drift_experiment._save_results_to_csv(
        metrics, timestamp, output_dir=tmp_path
    )

    # Verify files were created
    base_filename = "drift_experiment_20240101_000000"
    csv_path = tmp_path / f"{base_filename}.csv"
    meta_path = tmp_path / f"{base_filename}_meta.json"

    assert csv_path.exists()
    assert meta_path.exists()

    # Verify CSV contents
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["role"] == "assistant"
    assert df.iloc[0]["response"] == "Test response"

    # Verify metadata contents
    with open(meta_path) as f:
        metadata = json.load(f)
    assert metadata["timestamp"] == "2024-01-01T00:00:00"
    assert metadata["num_iterations"] == 1
    assert metadata["total_tokens"] == len("Test response")
    assert metadata["csv_filename"] == f"{base_filename}.csv"


def test_get_progress_range_with_tqdm(drift_experiment):
    """Test progress range creation with tqdm available."""
    drift_experiment.tqdm = MagicMock()
    drift_experiment._get_progress_range(5, "Test")
    assert drift_experiment.tqdm.called


def test_get_progress_range_without_tqdm(drift_experiment):
    """Test progress range creation without tqdm."""
    drift_experiment.tqdm = None
    range_obj = drift_experiment._get_progress_range(5, "Test")
    assert isinstance(range_obj, range)
    assert len(list(range_obj)) == 5
