"""Tests for budget tracking functionality."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from api.budget import BudgetTracker
from api.enums import (
    AnthropicModels,
    AzureModels,
    OllamaModels,
    OpenAIModels,
    Provider,
)


class TestBudgetTracker:
    """Test cases for BudgetTracker singleton."""

    def setup_method(self):
        """Reset singleton for each test."""
        BudgetTracker._instance = None

    @patch("api.budget.Path")
    def test_singleton_pattern(self, mock_path):
        """Test that BudgetTracker follows singleton pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_path.return_value = Path(tmpdir) / "budget_tracking.json"

            tracker1 = BudgetTracker()
            tracker2 = BudgetTracker()

            assert tracker1 is tracker2

    @patch("api.budget.Path")
    def test_add_usage_openai(self, mock_path):
        """Test adding usage for OpenAI provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            tracker = BudgetTracker()
            result = tracker.add_usage(
                Provider.OPENAI, OpenAIModels.GPT4_1106_PREVIEW, 1000, 500
            )

            expected_input_cost = (1000 / 1000.0) * 0.001
            expected_output_cost = (500 / 1000.0) * 0.003
            expected_total = expected_input_cost + expected_output_cost

            assert result["provider"] == "openai"
            assert result["model"] == "gpt-4-1106-preview"
            assert result["input_tokens"] == 1000
            assert result["output_tokens"] == 500
            assert result["total_cost"] == round(expected_total, 6)

    @patch("api.budget.Path")
    def test_add_usage_ollama_free(self, mock_path):
        """Test adding usage for free Ollama provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            tracker = BudgetTracker()
            result = tracker.add_usage(
                Provider.OLLAMA, OllamaModels.LLAMA3_70B, 2000, 1000
            )

            assert result["total_cost"] == 0.0

    @patch("api.budget.Path")
    def test_add_usage_anthropic(self, mock_path):
        """Test adding usage for Anthropic provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            tracker = BudgetTracker()
            result = tracker.add_usage(
                Provider.ANTHROPIC,
                AnthropicModels.CLAUDE_SONNET_4_20250514,
                1000,
                500,
            )

            expected_input_cost = (1000 / 1000.0) * 0.003  # $0.003
            expected_output_cost = (500 / 1000.0) * 0.015  # $0.0075
            expected_total = expected_input_cost + expected_output_cost

            assert result["total_cost"] == round(expected_total, 6)
            assert result["provider"] == "anthropic"
            assert result["model"] == "claude-sonnet-4-20250514"
            assert result["input_tokens"] == 1000
            assert result["output_tokens"] == 500
            assert result["total_cost"] == round(expected_total, 6)

    @patch("api.budget.Path")
    def test_add_usage_azure(self, mock_path):
        """Test adding usage for Azure provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            tracker = BudgetTracker()
            result = tracker.add_usage(
                Provider.AZURE, AzureModels.GPT4O_2024_08_06, 1000, 500
            )

            expected_input_cost = (1000 / 1000.0) * 0.00234883  # $0.00234883
            expected_output_cost = (500 / 1000.0) * 0.0093953  # $0.00469765
            expected_total = expected_input_cost + expected_output_cost
            assert result["provider"] == "azure"
            assert result["model"] == "azure-gpt-4o-2024-08-06"
            assert result["input_tokens"] == 1000
            assert result["output_tokens"] == 500
            assert result["total_cost"] == round(expected_total, 6)

    @patch("api.budget.Path")
    def test_load_corrupted_data(self, mock_path):
        """Test handling of corrupted JSON data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            # Create corrupted JSON file
            with open(budget_file, "w") as f:
                f.write('{"invalid": json}')  # Invalid JSON

            # Should handle corrupted file gracefully
            tracker = BudgetTracker()
            result = tracker.add_usage(
                Provider.OLLAMA, OllamaModels.LLAMA3_70B, 1000, 500
            )

            assert result["total_cost"] == 0.0
            assert result["input_cost"] == 0.0
            assert result["output_cost"] == 0.0

    @patch("api.budget.Path")
    def test_budget_persistence(self, mock_path):
        """Test that budget data is saved and loaded from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            # First tracker instance
            tracker1 = BudgetTracker()
            tracker1.add_usage(
                Provider.AZURE, AzureModels.GPT4O_2024_08_06, 1000, 1000
            )

            # Verify file was created
            assert budget_file.exists()

            # Reset singleton and create new instance
            BudgetTracker._instance = None
            tracker2 = BudgetTracker()

            # Check that data was loaded
            summary = tracker2.budget_data
            assert summary["total_cost"] > 0
            assert "azure" in summary["providers"]

    @patch("api.budget.Path")
    def test_invalid_model_pricing(self, mock_path):
        """Test handling of unknown model pricing."""
        mock_provider = MagicMock()
        mock_provider.value = "imaginary_provider"
        mock_model = MagicMock()
        mock_model.value = "imaginary_model"

        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            tracker = BudgetTracker()

            result = tracker.add_usage(mock_provider, mock_model, 1000, 500)

            assert result["total_cost"] == 0.0

    @patch("api.budget.Path")
    @patch("builtins.open", side_effect=IOError("Cannot write file"))
    def test_save_error_handling(self, mock_open, mock_path):
        """Test handling of file save errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            budget_file = Path(tmpdir) / "budget_tracking.json"
            mock_path.return_value = budget_file

            tracker = BudgetTracker()
            # This should not raise an exception despite save error
            result = tracker.add_usage(
                Provider.OLLAMA, OllamaModels.LLAMA3_70B, 1000, 500
            )

            assert result["total_cost"] == 0.0
