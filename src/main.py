import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import List

from babel_ai.experiment import Experiment, ExperimentConfig
from utils import load_yaml_config


def setup_logging(log_file: str):
    """Set up logging to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


async def run_experiment(config: ExperimentConfig):
    """Run a single experiment and log its results."""
    experiment = Experiment(config)
    await asyncio.to_thread(experiment.run)


async def run_experiment_batch(configs: List[ExperimentConfig]):
    """Run multiple experiments in parallel using asyncio."""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Set up logging with a more meaningful file name
    log_file = os.path.join(
        "logs",
        "experiment_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    setup_logging(log_file)

    # Run experiments in parallel
    await asyncio.gather(*(run_experiment(config) for config in configs))

    logging.info("All experiments completed.")


if __name__ == "__main__":
    # Example usage: replace with actual ExperimentConfig list
    example_configs = []  # Populate with actual ExperimentConfig instances
    if len(sys.argv) > 1:
        # Load configs from provided file paths
        for config_path in sys.argv[1:]:
            # Load and append each config
            # This is a placeholder for actual config loading logic
            config = load_yaml_config(ExperimentConfig, config_path)
            example_configs.append(config)
    else:
        raise ValueError("No config files provided")

    asyncio.run(run_experiment_batch(example_configs))
