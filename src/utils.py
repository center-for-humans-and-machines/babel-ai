import logging
from pathlib import Path
from typing import Iterator, Type, TypeVar

import yaml
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def progress_range(*args, desc: str = "Processing", **kwargs) -> Iterator[int]:
    """
    A range function with progress bar using tqdm.

    Args:
        *args: Arguments passed to range() (start, stop, step)
        desc: Description for the progress bar
        **kwargs: Additional keyword arguments passed to tqdm

    Returns:
        Iterator yielding integers with progress display

    Examples:
        # Simple range with progress
        for i in progress_range(100):
            # do something
            pass

        # Range with start, stop, step
        for i in progress_range(0, 100, 2, desc="Even numbers"):
            # do something
            pass
    """
    range_obj = range(*args)
    return tqdm(range_obj, desc=desc, **kwargs)


def load_yaml_config(config_type: Type[T], config_path: str | Path) -> T:
    """
    Load a YAML configuration file into a specified Pydantic model type.

    This function reads a YAML file and validates it against the provided
    Pydantic model type, ensuring type safety and data validation.

    Args:
        config_type: The Pydantic model class to load the config into
        config_path: Path to the YAML configuration file

    Returns:
        An instance of the specified config type with loaded data

    Raises:
        FileNotFoundError: If the configuration file doesn't exist

    Examples:
        # Load an experiment configuration
        config = load_yaml_config(ExperimentConfig, "config.yaml")

        # Load an agent configuration
        agent_config = load_yaml_config(AgentConfig, "agent.yaml")
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(
        f"Loading configuration of type {config_type.__name__} "
        f"from {config_path}"
    )

    with open(config_path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    logger.debug(f"Parsed YAML data: {yaml_data}")

    # Create and validate the configuration object
    config = config_type(**yaml_data)

    logger.info(
        f"Successfully loaded {config_type.__name__} from {config_path}"
    )
    return config
