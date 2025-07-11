import logging
from unittest.mock import MagicMock, patch

import pytest

from babel_ai.experiment import ExperimentConfig
from main import run_experiment, run_experiment_batch, setup_logging


class TestSetupLogging:
    """Test logging configuration."""

    def test_setup_logging_debug_vs_info(self):
        """Test debug vs info logging levels."""
        with patch("main.logging.basicConfig") as mock_basic_config:
            # Test debug mode
            setup_logging(debug=True)
            assert mock_basic_config.call_args[1]["level"] == logging.DEBUG

            # Test info mode
            setup_logging(debug=False)
            assert mock_basic_config.call_args[1]["level"] == logging.INFO


class TestExperimentExecution:
    """Test experiment execution."""

    @pytest.mark.asyncio
    async def test_run_experiment(self):
        """Test single experiment execution."""
        mock_config = MagicMock(spec=ExperimentConfig)
        mock_experiment = MagicMock()

        with patch("main.Experiment", return_value=mock_experiment):
            await run_experiment(mock_config)
            mock_experiment.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_experiment_batch_modes(self):
        """Test parallel vs sequential execution."""
        mock_config = MagicMock(spec=ExperimentConfig)

        with patch("main.run_experiment") as mock_run, patch(
            "main.setup_logging"
        ), patch("main.os.makedirs"):

            # Test parallel (default)
            await run_experiment_batch([mock_config, mock_config])
            assert mock_run.call_count == 2

            mock_run.reset_mock()

            # Test sequential
            await run_experiment_batch([mock_config], parallel=False)
            assert mock_run.call_count == 1
