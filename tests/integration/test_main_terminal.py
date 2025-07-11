import os
import shutil
import tempfile
from subprocess import run

import pytest


@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file for testing."""
    # Create a temporary directory for output
    temp_output_dir = tempfile.mkdtemp()

    # Minimal config content
    config_content = f"""
fetcher_config:
  fetcher: "random"
  category: "creative"

analyzer_config:
  analyzer: "similarity"
  analyze_window: 1

agent_configs:
  - provider: "openai"
    model: "gpt-4-1106-preview"
    system_prompt: "You are a helpful assistant."
    temperature: 0.7
    max_tokens: 150

agent_selection_method: "round_robin"
max_iterations: 1
max_total_characters: 100000
output_dir: "{temp_output_dir}"
"""

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
    with open(temp_file.name, "w") as f:
        f.write(config_content)
    yield temp_file.name, temp_output_dir

    # Cleanup
    os.remove(temp_file.name)
    shutil.rmtree(temp_output_dir, ignore_errors=True)


def test_main_execution(temp_config_file):
    """Test basic main.py execution."""
    config_file, temp_output_dir = temp_config_file

    result = run(
        ["python", "src/main.py", config_file],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "All experiments completed." in result.stderr
    assert os.path.exists(temp_output_dir)
    assert len(os.listdir(temp_output_dir)) == 2  # CSV and JSON files


def test_command_line_flags(temp_config_file):
    """Test command-line flags (debug and sequential)."""
    config_file, _ = temp_config_file

    # Test debug flag
    result = run(
        ["python", "src/main.py", "--debug", config_file],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Logging configured - Level: DEBUG" in result.stderr

    # Test sequential flag
    result = run(
        ["python", "src/main.py", "--sequential", config_file],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "experiments sequentially" in result.stderr
