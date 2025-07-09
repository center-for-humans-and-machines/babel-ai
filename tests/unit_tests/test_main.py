from unittest.mock import MagicMock, patch

import pytest

from babel_ai.experiment import ExperimentConfig
from main import run_experiment, run_experiment_batch


@pytest.mark.asyncio
async def test_run_experiment():
    mock_config = MagicMock(spec=ExperimentConfig)
    mock_experiment = MagicMock()
    with patch("main.Experiment", return_value=mock_experiment):
        await run_experiment(mock_config)
        mock_experiment.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_experiment_batch():
    mock_config1 = MagicMock(spec=ExperimentConfig)
    mock_config2 = MagicMock(spec=ExperimentConfig)
    with patch(
        "main.run_experiment", return_value=MagicMock()
    ) as mock_run_experiment:
        await run_experiment_batch([mock_config1, mock_config2])
        assert mock_run_experiment.call_count == 2
        mock_run_experiment.assert_any_call(mock_config1)
        mock_run_experiment.assert_any_call(mock_config2)
