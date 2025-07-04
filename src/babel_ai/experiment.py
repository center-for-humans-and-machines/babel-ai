"""LLM Drift Experiment.

This module implements an experiment to analyze long-term
behavior of Large Language Models when they operate in a
self-loop without external input.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from babel_ai.agent import Agent
from babel_ai.analyzer import Analyzer
from babel_ai.enums import AgentSelectionMethod
from babel_ai.prompt_fetcher import BasePromptFetcher
from models import (
    AgentMetric,
    AnalysisResult,
    ExperimentConfig,
    ExperimentMetadata,
    FetcherMetric,
    Metric,
)

logger = logging.getLogger(__name__)


class Experiment:
    """Main class for running LLM drift experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        use_notebook_tqdm: bool = False,
    ):
        # save configs
        self.config = config
        self.max_iterations = config.max_iterations
        self.max_total_characters = config.max_total_characters
        self.use_notebook_tqdm = use_notebook_tqdm

        # create output directory
        if config.output_dir is None:
            self.output_dir = Path.cwd() / "results"
        else:
            self.output_dir = Path(config.output_dir)

        # create metadata
        self.metadata = ExperimentMetadata(
            timestamp=datetime.now(),
            config=config,
        )

        # create analyzer
        self.analyzer = Analyzer.create_analyzer(
            analyzer_type=self.config.analyzer,
            analyze_window=self.config.analyzer_config.analyze_window,
        )

        # create prompt fetcher
        self.prompt_fetcher = BasePromptFetcher.create_fetcher(
            fetcher_type=self.config.fetcher,
            **self.config.fetcher_config.model_dump(),
        )

        # create agents
        self.agents = [
            Agent(agent_config) for agent_config in self.config.agent_configs
        ]

        # create agent selection method
        self.agent_selection_method = AgentSelectionMethod(
            self.config.agent_selection_method
        ).get_generator(self.agents)

        # set up results list
        self.results: List[Metric] = []

        # keep track of message history for the experiment
        self.messages: List[
            Dict[str, str]
        ] = self.prompt_fetcher.get_conversation()

        # update metadata
        self.metadata.num_fetcher_messages = len(self.messages)

    def run(
        self,
        output_dir: Optional[Path] = None,
    ) -> List[Metric]:
        """Run the drift experiment."""
        self.generate_conversation()
        self._save_results_to_csv(
            metrics=self.results,
            metadata=self.metadata,
            output_dir=output_dir,
        )
        return self.results

    def generate_conversation(
        self,
    ) -> List[Metric]:
        """Run the drift experiment with the given initial messages.

        Args:
            initial_messages: List of message dictionaries to start with

        Returns:
            List of Metric objects containing experiment results

        Raises:
            ValueError: If messages are not properly formatted
        """

        for i, message in enumerate(self.messages):
            self.results.append(
                FetcherMetric(
                    iteration=i,
                    timestamp=datetime.now(),
                    role=message["role"],
                    content=message["content"],
                    fetcher_type=self.config.fetcher,
                    fetcher_config=self.config.fetcher_config,
                )
            )

        # keep track of total characters
        self.total_characters = sum(
            len(msg["content"]) for msg in self.messages
        )

        # continue until max iterations or max total characters is reached
        iteration = len(self.results)  # Start from after fetcher metrics
        while self._should_continue_generation():

            # select next agent
            agent = next(self.agent_selection_method)

            # generate response
            response = agent.generate_response(self.messages)

            # add response to conversation history and update total characters
            self.total_characters += len(response)
            self.messages.append(
                {
                    "role": str(agent.id),
                    "content": response,
                }
            )

            # add response to results
            self.results.append(
                AgentMetric(
                    iteration=iteration,
                    timestamp=datetime.now(),
                    role=str(agent.id),
                    content=response,
                    agent_id=str(agent.id),
                    agent_config=agent.config,
                )
            )

            iteration += 1

        # save total iterations/characters to metadata
        self.metadata.total_characters = self.total_characters
        self.metadata.num_iterations_total = iteration

        return self.results

    def _analyze_response(self, metrics: List[Metric]) -> AnalysisResult:
        """Analyze the response of the last agent."""

        content = [metric.content for metric in metrics]

        for i, metric in enumerate(metrics):
            metric.analysis = self.analyzer.analyze(content[: i + 1])

        return metrics

    def _should_continue_generation(self) -> bool:
        """Check if generation should continue based on configured limits."""
        # Check iteration limit
        if len(self.messages) >= self.max_iterations:
            return False
        # Check character limit
        if self.total_characters >= self.max_total_characters:
            return False
        return True

    def _save_results_to_csv(
        self,
        metrics: List[Metric],
        metadata: ExperimentMetadata,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save experiment results to a CSV file and metadata to JSON.

        This method handles different types of metrics (Metric, FetcherMetric,
        AgentMetric) and flattens all nested structures into a single CSV file.
        It also saves comprehensive metadata to a JSON file.

        Args:
            metrics: List of Metric objects from the experiment
            metadata: ExperimentMetadata object containing experiment info
        """

        # Convert metrics to DataFrame
        df = pd.DataFrame([metric.to_dict() for metric in metrics])

        # Use current working directory if no output directory specified
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        base_filename = (
            f"drift_experiment_{metadata.timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        csv_filename = f"{base_filename}.csv"
        meta_filename = f"{base_filename}_meta.json"
        csv_path = output_dir / csv_filename
        meta_path = output_dir / meta_filename

        df.to_csv(csv_path, index=False)

        # Save metadata
        metadata_dict = metadata.model_dump()

        with open(meta_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        logger.info(f"Saved experiment results to {csv_path}")
        logger.info(f"Saved experiment metadata to {meta_path}")
