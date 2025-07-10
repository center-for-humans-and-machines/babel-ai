import os
import shutil
import tempfile
from subprocess import run

import pytest


@pytest.fixture
def temp_config_file():
    """
    Fixture to create a temporary YAML
    config file derived from test_config.yaml.
    """
    # Create a temporary directory for output
    temp_output_dir = tempfile.mkdtemp()

    # Hardcoded config content with temporary output directory
    config_content = f"""# Test configuration for LLM Drift Experiment
# This file defines a complete ExperimentConfig in YAML format

# Configuration for the fetcher
fetcher_config:
  fetcher: "random"
  category: "creative"

# Configuration for the analyzer
analyzer_config:
  analyzer: "similarity"
  analyze_window: 1

# List of agent configurations to use in the experiment
agent_configs:
  - provider: "openai"
    model: "gpt-4-1106-preview"
    system_prompt: "You are a helpful assistant participating in a conversation." # noqa: E501
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

# Directory to save results (temporary directory)
output_dir: "{temp_output_dir}"
"""

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
    with open(temp_file.name, "w") as f:
        f.write(config_content)
    yield temp_file.name, temp_output_dir

    # Clean up temporary file and directory
    os.remove(temp_file.name)
    shutil.rmtree(temp_output_dir, ignore_errors=True)


def test_main_terminal_execution(temp_config_file):
    config_file, temp_output_dir = temp_config_file

    # Run the main.py script with the temporary config file
    result = run(
        ["python", "src/main.py", config_file, config_file],
        capture_output=True,
        text=True,
    )

    # Check if the script executed successfully
    assert result.returncode == 0
    assert "All experiments completed." in result.stderr

    # Check here that the correct output files are created

    # Check that output directory exists and contains the expected files
    assert os.path.exists(temp_output_dir)

    # List all files in the output directory
    output_files = os.listdir(temp_output_dir)

    # Check that we have exactly 2 files (CSV and JSON)
    assert len(output_files) == 2

    # Check for CSV file (results)
    csv_files = [f for f in output_files if f.endswith(".csv")]
    assert len(csv_files) == 1
    assert csv_files[0].startswith("drift_experiment_")

    # Check for JSON file (metadata)
    json_files = [f for f in output_files if f.endswith(".json")]
    assert len(json_files) == 1
    assert json_files[0].startswith("drift_experiment_")
    assert json_files[0].endswith("_meta.json")

    # Verify that both files have the same base timestamp
    csv_basename = csv_files[0].replace(".csv", "")
    json_basename = json_files[0].replace("_meta.json", "")
    assert csv_basename == json_basename
