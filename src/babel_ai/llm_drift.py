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

from api.llm_interface import Provider, generate_response
from babel_ai.agent import Agent
from babel_ai.analyzer import Analyzer
from babel_ai.enums import AgentSelectionMethod
from babel_ai.models import ExperimentConfig, Metric
from babel_ai.prompt_fetcher import PromptFetcher
from utils import progress_range

logger = logging.getLogger(__name__)


class DriftExperiment:
    """Main class for running LLM drift experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        use_notebook_tqdm: bool = False,
    ):
        # save config
        self.config = config

        # create analyzer
        self.analyzer = Analyzer.create_analyzer(
            analyzer_type=self.config.analyzer,
            analyze_window=self.config.analyze_window,
        )

        # create prompt fetcher
        self.prompt_fetcher = PromptFetcher.create_fetcher(
            fetcher_type=self.config.fetcher,
            **self.config.fetcher_config,
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

    def run(
        self,
        initial_messages: List[Dict[str, str]],
    ) -> List[Metric]:
        """Run the drift experiment with the given initial messages.

        Args:
            initial_messages: List of message dictionaries to start with

        Returns:
            List of Metric objects containing experiment results

        Raises:
            ValueError: If messages are not properly formatted
        """
        # Initialize message history with all but the last message
        self._validate_messages(initial_messages)
        self.messages = initial_messages[:-1].copy()

        # Analyze initial messages
        msgs_text = [message["content"] for message in initial_messages]
        metrics = []
        for i in progress_range(
            len(initial_messages), desc="Analyzing initial messages"
        ):
            role = initial_messages[i]["role"]
            content = initial_messages[i]["content"]

            # Analyze content
            analysis = self.analyzer.analyze(
                [message["content"] for message in initial_messages[: i + 1]]
            )

            # Store results
            metrics.append(
                Metric(
                    iteration=i,
                    timestamp=datetime.now(),
                    role=role,
                    response=content,
                    analysis=analysis,
                    config=self.config,
                )
            )

        # Run experiment
        for i in progress_range(
            self.config.max_iterations, desc="Running drift experiment"
        ):
            # Generate next response using the function
            response = generate_response(
                messages=self.messages,
                provider=Provider(self.config.provider),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                top_p=self.config.top_p,
            )

            # Add the response to message history
            self.messages.append({"role": "assistant", "content": response})

            # Store output
            msgs_text.append(response)

            # Analyze all outputs
            analysis = self.analyzer.analyze(msgs_text)

            # Store results
            metrics.append(
                Metric(
                    iteration=len(msgs_text),
                    timestamp=datetime.now(),
                    role="assistant",
                    response=response,
                    analysis=analysis,
                    config=self.config,
                )
            )

            # Check total length of outputs
            if (
                sum(len(output) for output in msgs_text)
                > self.config.max_total_characters
            ):
                break

        # Save results to CSV
        self._save_results_to_csv(metrics, metrics[0].timestamp)

        return metrics

    def _save_results_to_csv(
        self,
        metrics: List[Metric],
        timestamp: datetime,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save experiment results to a CSV file and metadata to JSON.

        Args:
            metrics: List of Metric objects from the experiment
            timestamp: Timestamp to use in filename
            output_dir: Directory to save output files. If None, uses current
                       working directory.
        """
        # Convert metrics to DataFrame
        df = pd.DataFrame([metric.to_dict() for metric in metrics])

        # Use current working directory if no output directory specified
        output_dir = output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        base_filename = (
            f"drift_experiment_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        csv_filename = f"{base_filename}.csv"
        meta_filename = f"{base_filename}_meta.json"
        csv_path = output_dir / csv_filename
        meta_path = output_dir / meta_filename

        df.to_csv(csv_path, index=False)

        # Save metadata
        metadata = {
            "timestamp": timestamp.isoformat(),
            "config": self.config.model_dump(),
            "num_iterations": len(metrics),
            "total_tokens": sum(len(m.response) for m in metrics),
            "csv_filename": csv_filename,
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved experiment results to {csv_path}")
        logger.info(f"Saved experiment metadata to {meta_path}")
