"""Tests for the t-SNE artifact backfill command."""

import runpy
import sys
from pathlib import Path

import pandas as pd


def test_cli_backfills_results_root(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "turn_index": [0, 1],
            "speaker": ["agent_0", "agent_0"],
            "content": ["first", "second"],
            "agent_config": ["{}", "{}"],
        }
    ).to_parquet(run_dir / "turns.parquet", index=False)
    script = Path(__file__).parents[3] / "scripts" / "analyze_tsne.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    assert "run-1: unavailable" in capsys.readouterr().out
