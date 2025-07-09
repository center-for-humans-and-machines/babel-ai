import os
import tempfile
from subprocess import run

import pytest


@pytest.fixture
def temp_config_file():
    """
    Fixture to create a temporary YAML
    config file derived from test_config.yaml.
    """
    # Hardcoded config content
    config_content = """# Test configuration for LLM Drift Experiment
# This file defines a complete ExperimentConfig in YAML format

# Configuration for the fetcher
fetcher_config:
  fetcher: "random"

# Configuration for the analyzer
analyzer_config:
  analyzer: "similarity"
  analyze_window: 1

# List of agent configurations to use in the experiment
agent_configs:
  - provider: "openai"
    model: "gpt-4-1106-preview"
    system_prompt: "You are a helpful assistant participating in a
      conversation."
    temperature: 0.7
    max_tokens: 150
    frequency_penalty: 0.0
    presence_penalty: 0.0
    top_p: 1.0

# Method for selecting which agent responds next
agent_selection_method: "round_robin"

# Maximum number of conversation turns to run
max_iterations: 1

# Maximum total characters across all responses
max_total_characters: 100000

# Directory to save results (optional)
output_dir: "results"
"""

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
    with open(temp_file.name, "w") as f:
        f.write(config_content)
    yield temp_file.name

    # Clean up temporary file
    os.remove(temp_file.name)


def test_main_terminal_execution(temp_config_file):
    # Run the main.py script with the temporary config file
    result = run(
        ["python", "src/main.py", temp_config_file, temp_config_file],
        capture_output=True,
        text=True,
    )

    # Check if the script executed successfully
    assert result.returncode == 0
    assert "All experiments completed." in result.stdout
