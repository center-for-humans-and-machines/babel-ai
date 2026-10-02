"""Tests for t-SNE artifact persistence and backfill."""

import json

import numpy as np
import pandas as pd

from trajectory.artifacts import (
    META_FILENAME,
    TRAJECTORY_FILENAME,
    analyze_results_root,
    analyze_run,
    analyze_run_safely,
    load_projection,
)


class FakeEmbedder:
    """Small deterministic embedder for artifact tests."""

    model_name = "fake"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[index, len(text), index + 1] for index, text in enumerate(texts)],
            dtype=float,
        )


def _write_turns(run_dir, count: int = 3) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "turn_index": range(count),
            "speaker": ["agent_0"] * count,
            "content": [f"turn {index}" for index in range(count)],
            "agent_config": ["{}"] * count,
        }
    ).to_parquet(run_dir / "turns.parquet", index=False)


def test_analyze_run_persists_and_loads_projection(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_turns(run_dir)
    created = analyze_run(run_dir, embedder=FakeEmbedder())
    loaded = load_projection(run_dir)
    assert (run_dir / TRAJECTORY_FILENAME).is_file()
    assert (run_dir / META_FILENAME).is_file()
    assert loaded is not None
    assert loaded.meta == created.meta
    pd.testing.assert_frame_equal(loaded.trajectory, created.trajectory)


def test_load_projection_returns_none_when_artifact_is_absent(tmp_path):
    assert load_projection(tmp_path) is None


def test_safe_analysis_records_failure(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    def fail(*args, **kwargs):
        raise RuntimeError("projection broke")

    monkeypatch.setattr("trajectory.artifacts.analyze_run", fail)
    assert analyze_run_safely(run_dir) is None
    meta = json.loads((run_dir / META_FILENAME).read_text())
    assert meta["status"] == "failed"
    assert meta["reason"] == "projection broke"


def test_results_root_backfills_completed_runs(tmp_path):
    complete = tmp_path / "complete"
    _write_turns(complete, count=2)
    (tmp_path / "incomplete").mkdir()
    statuses = analyze_results_root(tmp_path)
    assert statuses == {"complete": "unavailable"}
    assert (complete / TRAJECTORY_FILENAME).is_file()


def test_results_root_handles_missing_directory(tmp_path):
    assert analyze_results_root(tmp_path / "missing") == {}
