"""LLM Drift Experiment.

This module implements an experiment to analyze long-term
behavior of Large Language Models when they operate in a
self-loop without external input.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm as terminal_tqdm
    from tqdm.notebook import tqdm as notebook_tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from src.babel_ai.analyzer import SimilarityAnalyzer
from src.babel_ai.llm_interface import LLMInterface
from src.babel_ai.models import ExperimentConfig, Metric
from src.babel_ai.prompt_fetcher import BasePromptFetcher, PromptFetcher

logger = logging.getLogger(__name__)


class DriftExperiment:
    """Main class for running LLM drift experiments."""

    def __init__(
        self,
        llm_provider: Optional[LLMInterface] = None,
        config: Optional[ExperimentConfig] = None,
        analyzer: Optional[SimilarityAnalyzer] = None,
        prompt_fetcher: Optional[BasePromptFetcher] = None,
        use_notebook_tqdm: bool = False,
    ):
        self.llm = llm_provider or LLMInterface()
        self.config = config or ExperimentConfig()
        self.analyzer = analyzer or SimilarityAnalyzer(
            analyze_window=self.config.analyze_window
        )
        self.prompt_fetcher = prompt_fetcher or PromptFetcher()
        self.results: List[Metric] = []

        # Set up progress tracking
        self.range_func = range
        if TQDM_AVAILABLE:
            tqdm_class = notebook_tqdm if use_notebook_tqdm else terminal_tqdm
            self.tqdm = tqdm_class
        else:
            self.tqdm = None

    def _get_progress_range(self, total: int, desc: str = "Progress") -> range:
        """Create a progress-tracked range.

        Args:
            total: Total number of iterations
            desc: Description for the progress bar

        Returns:
            Range object, optionally wrapped in tqdm
        """
        if self.tqdm is not None:
            return self.tqdm(range(total), desc=desc)
        return range(total)

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format for OpenAI API.

        Args:
            messages: List of message dictionaries to validate

        Raises:
            ValueError: If messages are not properly formatted
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        valid_roles = {"system", "user", "assistant"}
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise ValueError(f"Message {i} must have a 'role' key")

            if message["role"] not in valid_roles:
                raise ValueError(
                    f"Message {i} has invalid role '{message['role']}'. "
                    f"Must be one of: {valid_roles}"
                )

            if "content" not in message:
                raise ValueError(f"Message {i} must have a 'content' key")

            if not isinstance(message["content"], str):
                raise ValueError(f"Message {i} content must be a string")

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
        # Initialize LLM's message history with all but the last message
        self._validate_messages(initial_messages)
        self.llm.messages = initial_messages[:-1]

        # Analyze initial messages
        msgs_text = [message["content"] for message in initial_messages]
        metrics = []
        for i in self._get_progress_range(
            len(initial_messages), desc="Analyzing initial messages"
        ):
            role = initial_messages[i]["role"]
            content = initial_messages[i]["content"]
            metrics.append(
                Metric(
                    iteration=i,
                    timestamp=datetime.now(),
                    role=role,
                    response=content,
                    analysis=self.analyzer.analyze(content),
                    config=self.config,
                )
            )

        # Initialize prompt
        prompt = msgs_text[-1]

        # Run experiment
        for i in self._get_progress_range(
            self.config.max_iterations, desc="Running drift experiment"
        ):
            # Generate next response
            response = self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                top_p=self.config.top_p,
            )

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

            # Update prompt for next iteration
            prompt = response

        # Save results to CSV
        self._save_results_to_csv(metrics, metrics[0].timestamp)

        return metrics

    def _save_results_to_csv(
        self, metrics: List[Metric], timestamp: datetime
    ) -> None:
        """Save experiment results to a CSV file and metadata to JSON.

        Args:
            metrics: List of Metric objects from the experiment
            timestamp: Timestamp to use in filename
        """
        # Convert metrics to DataFrame
        df = pd.DataFrame([metric.to_dict() for metric in metrics])

        # Save CSV
        base_filename = (
            f"drift_experiment_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        csv_filename = f"{base_filename}.csv"
        meta_filename = f"{base_filename}_meta.json"

        df.to_csv(csv_filename, index=False)

        # Save metadata
        metadata = {
            "timestamp": timestamp.isoformat(),
            "config": self.config.model_dump(),
            "num_iterations": len(metrics),
            "total_tokens": sum(len(m.response) for m in metrics),
            "csv_filename": csv_filename,
        }

        with open(meta_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved experiment results to {csv_filename}")
        logger.info(f"Saved experiment metadata to {meta_filename}")
