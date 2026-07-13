import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import List

from experiment import Experiment, ExperimentConfig
from utils import load_yaml_config

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, debug: bool = False):
    """Configure quiet console output and detailed file logging."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_level = logging.DEBUG if debug else logging.WARNING
    file_level = logging.DEBUG if debug else logging.INFO

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.DEBUG)

    logger.debug(
        "Logging configured - console: %s, file: %s",
        logging.getLevelName(console_level),
        logging.getLevelName(file_level) if log_file else "disabled",
    )
    if log_file:
        logger.debug("Log file: %s", log_file)


async def run_experiment(config: ExperimentConfig):
    """Run a single experiment and log its results."""
    experiment = Experiment(config)
    logger.debug("Starting experiment thread.")
    await asyncio.to_thread(experiment.run)


async def run_experiment_batch(
    configs: List[ExperimentConfig],
    parallel: bool = True,
    debug: bool = False,
):
    """Run multiple experiments in parallel using asyncio."""

    os.makedirs("logs", exist_ok=True)

    setup_logging(
        log_file=os.path.join(
            "logs",
            "experiment_{}.log".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        ),
        debug=debug,
    )

    if parallel:
        logger.debug("Running %s experiments in parallel.", len(configs))
        for i, config in enumerate(configs):
            logger.debug(
                "Running experiment %s with config: %s",
                i,
                config.model_dump(),
            )
        await asyncio.gather(*(run_experiment(config) for config in configs))
    else:
        logger.debug(
            "Running %s experiments sequentially.",
            len(configs),
        )
        for i, config in enumerate(configs):
            logger.debug(
                "Running experiment %s with config: %s",
                i,
                config.model_dump(),
            )
            await run_experiment(config)
            logger.debug("Experiment %s completed", i)

    logger.debug("All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run babel_ai experiments")
    parser.add_argument(
        "config_files", nargs="+", help="Path to experiment config files"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of parallel",
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)

    example_configs = []
    for config_path in args.config_files:
        config = load_yaml_config(ExperimentConfig, config_path)
        example_configs.append(config)

    asyncio.run(
        run_experiment_batch(
            example_configs, parallel=not args.sequential, debug=args.debug
        )
    )
