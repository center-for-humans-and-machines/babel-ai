"""Tests for the LLM Drift Experiment class."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from api.llm_interface import Provider
from api.openai import OpenAIModel
from babel_ai.agent import Agent
from babel_ai.analyzer import Analyzer
from babel_ai.enums import AgentSelectionMethod, AnalyzerType, FetcherType
from babel_ai.experiment import Experiment
from babel_ai.prompt_fetcher import BasePromptFetcher
from models import (
    AgentConfig,
    AgentMetric,
    AnalyzerConfig,
    ExperimentConfig,
    ExperimentMetadata,
    FetcherConfig,
    FetcherMetric,
)


@pytest.fixture
def sample_agent_config():
    """Create a sample AgentConfig for testing."""
    return AgentConfig(
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT4_1106_PREVIEW,
        temperature=0.8,
        max_tokens=150,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        top_p=0.9,
    )


@pytest.fixture
def sample_fetcher_config():
    """Create a sample FetcherConfig for testing."""
    return FetcherConfig(
        fetcher=FetcherType.SHAREGPT,
        data_path="/fake/path/data.json",
        min_messages=2,
        max_messages=10,
    )


@pytest.fixture
def sample_analyzer_config():
    """Create a sample AnalyzerConfig for testing."""
    return AnalyzerConfig(analyzer=AnalyzerType.SIMILARITY, analyze_window=5)


@pytest.fixture
def sample_experiment_config(
    sample_agent_config, sample_fetcher_config, sample_analyzer_config
):
    """Create a sample ExperimentConfig for testing."""
    # The fetcher and analyzer types are now already included in the fixtures

    return ExperimentConfig(
        fetcher_config=sample_fetcher_config,
        analyzer_config=sample_analyzer_config,
        agent_configs=[sample_agent_config],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=5,
        max_total_characters=1000,
        output_dir=None,
    )


@pytest.fixture
def sample_messages():
    """Create sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer."""
    analyzer = Mock(spec=Analyzer)
    return analyzer


@pytest.fixture
def mock_prompt_fetcher(sample_messages):
    """Create a mock prompt fetcher."""
    fetcher = Mock(spec=BasePromptFetcher)
    fetcher.get_conversation.return_value = sample_messages
    return fetcher


@pytest.fixture
def mock_agent(sample_agent_config):
    """Create a mock agent."""
    agent = Mock(spec=Agent)
    agent.id = "test_agent_1"
    agent.config = sample_agent_config
    agent.generate_response.return_value = "Test response from agent"
    return agent


class TestExperiment:
    """Test the Experiment class."""

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_experiment_initialization(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
        sample_messages,
    ):
        """Test that Experiment initializes correctly with config."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(sample_experiment_config)

        # Verify basic attributes
        assert experiment.config == sample_experiment_config
        assert experiment.max_iterations == 5
        assert experiment.max_total_characters == 1000
        assert experiment.use_notebook_tqdm is False

        # Verify output directory setup
        assert experiment.output_dir == Path.cwd() / "results"

        # Verify metadata initialization
        assert isinstance(experiment.metadata, ExperimentMetadata)
        assert experiment.metadata.config == sample_experiment_config
        assert experiment.metadata.num_fetcher_messages == len(sample_messages)

        # Verify components were created correctly
        mock_analyzer_create.assert_called_once_with(
            analyzer_type=AnalyzerType.SIMILARITY,
            analyze_window=5,
        )
        mock_fetcher_create.assert_called_once_with(
            fetcher_type=FetcherType.SHAREGPT,
            data_path="/fake/path/data.json",
            min_messages=2,
            max_messages=10,
        )
        mock_agent_class.assert_called_once()

        # Verify messages and results initialization
        assert experiment.messages == sample_messages
        assert experiment.result_metrics == []

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_experiment_initialization_with_custom_output_dir(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test experiment initialization with custom output directory."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        # Set custom output dir
        sample_experiment_config.output_dir = "/custom/output/path"

        experiment = Experiment(sample_experiment_config)

        assert experiment.output_dir == Path("/custom/output/path")

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_should_continue_generation(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test _should_continue_generation method."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(sample_experiment_config)
        experiment.total_characters = 500

        # Should continue when under limits
        assert experiment._should_continue_generation() is True

        # Should stop when max iterations reached
        experiment.messages = ["msg"] * 5  # Reach max_iterations
        assert experiment._should_continue_generation() is False

        # Reset messages, test character limit
        experiment.messages = ["msg"] * 3
        experiment.total_characters = 1001  # Exceed max_total_characters
        assert experiment._should_continue_generation() is False

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_generate_conversation(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
        sample_messages,
    ):
        """Test the generate_conversation method."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        # Create experiment with very low limits to control execution
        sample_experiment_config.max_iterations = 4  # 3 fetcher + 1 agent
        sample_experiment_config.max_total_characters = 500

        experiment = Experiment(sample_experiment_config)

        # Mock agent selection generator
        def mock_agent_generator():
            while True:
                yield mock_agent

        experiment.agent_selection_method = mock_agent_generator()

        results = experiment.run_interaction_loop()

        # Should have 3 FetcherMetrics + 1 AgentMetric
        assert len(results) == 4
        assert len(experiment.result_metrics) == 4

        # Check FetcherMetric results
        for i in range(3):
            metric = results[i]
            assert isinstance(metric, FetcherMetric)
            assert metric.iteration == i
            assert metric.role == sample_messages[i]["role"]
            assert metric.content == sample_messages[i]["content"]
            assert metric.fetcher_config.fetcher == FetcherType.SHAREGPT

        # Check AgentMetric result
        agent_metric = results[3]
        assert isinstance(agent_metric, AgentMetric)
        assert agent_metric.iteration == 3
        assert agent_metric.role == "test_agent_1"
        assert agent_metric.content == "Test response from agent"
        assert agent_metric.agent_id == "test_agent_1"

        # Verify agent was called correctly
        mock_agent.generate_response.assert_called_once()

        # Check metadata updates
        assert (
            experiment.metadata.total_characters == experiment.total_characters
        )
        assert experiment.metadata.num_iterations_total == 4

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pandas.DataFrame.to_csv")
    @patch("pathlib.Path.mkdir")
    @patch("json.dump")
    def test_save_results_to_csv(
        self,
        mock_json_dump,
        mock_mkdir,
        mock_to_csv,
        mock_file_open,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test the _save_results_to_csv method."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(sample_experiment_config)

        # Create some test metrics
        test_metrics = [
            FetcherMetric(
                iteration=0,
                timestamp=datetime.now(),
                role="user",
                content="test content",
                fetcher_type=FetcherType.SHAREGPT,
                fetcher_config=sample_experiment_config.fetcher_config,
            )
        ]

        experiment._save_results_to_csv(
            metrics=test_metrics,
            metadata=experiment.metadata,
        )

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify CSV saving
        mock_to_csv.assert_called_once()

        # Verify JSON metadata saving
        mock_file_open.assert_called()
        mock_json_dump.assert_called_once()

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_run_method(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test the run method integrates
        generate_conversation and _save_results_to_csv."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(sample_experiment_config)

        # Mock the methods called by run
        with patch.object(
            experiment, "generate_conversation"
        ) as mock_generate, patch.object(
            experiment, "_save_results_to_csv"
        ) as mock_save:

            mock_generate.return_value = []
            results = experiment.run()

            # Verify methods were called
            mock_generate.assert_called_once()
            mock_save.assert_called_once_with(
                metrics=experiment.result_metrics,
                metadata=experiment.metadata,
                output_dir=None,
            )

            assert results == []

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_run_method_with_custom_output_dir(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test the run method with custom output directory."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(sample_experiment_config)
        custom_output_dir = Path("/custom/test/path")

        # Mock the methods called by run
        with patch.object(
            experiment, "generate_conversation"
        ) as mock_generate, patch.object(
            experiment, "_save_results_to_csv"
        ) as mock_save:

            mock_generate.return_value = []
            experiment.run(output_dir=custom_output_dir)

            # Verify save was called with custom output dir
            mock_save.assert_called_once_with(
                metrics=experiment.result_metrics,
                metadata=experiment.metadata,
                output_dir=custom_output_dir,
            )

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_multiple_agents_selection(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        sample_agent_config,
        mock_analyzer,
        mock_prompt_fetcher,
    ):
        """Test that multiple agents are selected correctly."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher

        # Create two different mock agents
        mock_agent1 = Mock(spec=Agent)
        mock_agent1.id = "agent_1"
        mock_agent1.config = sample_agent_config
        mock_agent1.generate_response.return_value = "Response from agent 1"

        mock_agent2 = Mock(spec=Agent)
        mock_agent2.id = "agent_2"
        mock_agent2.config = sample_agent_config
        mock_agent2.generate_response.return_value = "Response from agent 2"

        mock_agent_class.side_effect = [mock_agent1, mock_agent2]

        # Configure experiment with two agents
        sample_experiment_config.agent_configs = [
            sample_agent_config,
            sample_agent_config,
        ]
        sample_experiment_config.max_iterations = 5  # 3 fetcher + 2 agent

        experiment = Experiment(sample_experiment_config)

        # Mock round-robin agent selection
        def mock_round_robin():
            agents = [mock_agent1, mock_agent2]
            i = 0
            while True:
                yield agents[i % len(agents)]
                i += 1

        experiment.agent_selection_method = mock_round_robin()

        results = experiment.run_interaction_loop()

        # Should have 3 FetcherMetrics + 2 AgentMetrics
        assert len(results) == 5

        # Check that both agents were called
        mock_agent1.generate_response.assert_called_once()
        mock_agent2.generate_response.assert_called_once()

        # Check that agent metrics have correct agent IDs
        agent_metrics = [r for r in results if isinstance(r, AgentMetric)]
        assert len(agent_metrics) == 2
        assert agent_metrics[0].agent_id == "agent_1"
        assert agent_metrics[1].agent_id == "agent_2"

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_experiment_stops_at_character_limit(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test that experiment stops when character limit is reached."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        # Set very low character limit
        sample_experiment_config.max_total_characters = 100
        sample_experiment_config.max_iterations = 100  # High iteration limit

        # Mock agent to return long responses
        mock_agent.generate_response.return_value = "X" * 50

        experiment = Experiment(sample_experiment_config)

        # Mock agent selection generator
        def mock_agent_generator():
            while True:
                yield mock_agent

        experiment.agent_selection_method = mock_agent_generator()

        results = experiment.run_interaction_loop()

        # Should stop due to character limit, not iteration limit
        # 3 fetcher messages + some agent messages until character limit
        assert len(results) >= 3
        assert experiment.total_characters >= 100

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_experiment_with_empty_messages(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_agent,
    ):
        """Test experiment behavior with empty initial messages."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_prompt_fetcher = Mock(spec=BasePromptFetcher)
        mock_prompt_fetcher.get_conversation.return_value = []
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(sample_experiment_config)

        # Mock agent selection generator
        def mock_agent_generator():
            while True:
                yield mock_agent

        experiment.agent_selection_method = mock_agent_generator()

        results = experiment.run_interaction_loop()

        # Should only have agent metrics, no fetcher metrics
        agent_metrics = [r for r in results if isinstance(r, AgentMetric)]
        fetcher_metrics = [r for r in results if isinstance(r, FetcherMetric)]

        assert len(fetcher_metrics) == 0
        assert len(agent_metrics) > 0
        assert experiment.metadata.num_fetcher_messages == 0

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_experiment_with_notebook_tqdm_flag(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
    ):
        """Test experiment initialization with notebook_tqdm flag."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(
            sample_experiment_config, use_notebook_tqdm=True
        )

        assert experiment.use_notebook_tqdm is True

    @patch("babel_ai.experiment.Analyzer.create_analyzer")
    @patch("babel_ai.experiment.BasePromptFetcher.create_fetcher")
    @patch("babel_ai.experiment.Agent")
    def test_experiment_messages_accumulate_correctly(
        self,
        mock_agent_class,
        mock_fetcher_create,
        mock_analyzer_create,
        sample_experiment_config,
        mock_analyzer,
        mock_prompt_fetcher,
        mock_agent,
        sample_messages,
    ):
        """Test that messages accumulate correctly during conversation."""
        # Setup mocks
        mock_analyzer_create.return_value = mock_analyzer
        mock_fetcher_create.return_value = mock_prompt_fetcher
        mock_agent_class.return_value = mock_agent

        # Set low limits to control execution
        sample_experiment_config.max_iterations = 5
        sample_experiment_config.max_total_characters = 500

        experiment = Experiment(sample_experiment_config)

        # Track messages before and after
        initial_message_count = len(experiment.messages)

        # Mock agent selection generator
        def mock_agent_generator():
            while True:
                yield mock_agent

        experiment.agent_selection_method = mock_agent_generator()

        experiment.run_interaction_loop()

        # Messages should have grown
        assert len(experiment.messages) > initial_message_count

        # Last message should be from agent
        last_message = experiment.messages[-1]
        assert last_message["role"] == mock_agent.id
        assert (
            last_message["content"]
            == mock_agent.generate_response.return_value
        )
