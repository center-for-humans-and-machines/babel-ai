import logging
from unittest.mock import MagicMock, patch

import pytest

from experiment import ExperimentConfig
from main import run_experiment, run_experiment_batch, setup_logging


class TestSetupLogging:
    """Test logging configuration."""

    def test_setup_logging_sets_console_and_file_levels(self, tmp_path):
        """Console stays quiet unless debug mode is enabled."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=str(log_file), debug=False)
        root = logging.getLogger()
        console = root.handlers[0]
        file_handler = root.handlers[1]
        assert console.level == logging.WARNING
        assert file_handler.level == logging.INFO
        assert root.level == logging.DEBUG

    def test_setup_logging_debug_enables_console_debug(self):
        """Debug mode mirrors verbose output to the console too."""
        setup_logging(debug=True)
        root = logging.getLogger()
        console = root.handlers[0]
        assert console.level == logging.DEBUG


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

            await run_experiment_batch([mock_config, mock_config])
            assert mock_run.call_count == 2

            mock_run.reset_mock()

            await run_experiment_batch([mock_config], parallel=False)
            assert mock_run.call_count == 1
