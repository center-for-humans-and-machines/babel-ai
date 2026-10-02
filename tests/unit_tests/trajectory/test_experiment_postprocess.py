"""Test automatic t-SNE post-processing after canonical save."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

from experiment import Experiment


def test_experiment_runs_tsne_after_saving(tmp_path, monkeypatch):
    experiment = Experiment.__new__(Experiment)
    experiment.uuid = UUID("00000000-0000-0000-0000-000000000001")
    experiment.output_dir = tmp_path
    experiment._manager = SimpleNamespace(run_id="run-1")
    experiment.config = Mock()
    metadata = SimpleNamespace(
        config=Mock(),
        num_iterations_total=3,
        num_fetcher_messages=1,
        total_characters=20,
        timestamp=datetime(2026, 7, 13),
    )
    saved = []
    analyzed = []

    def save(run_dir, metrics, meta, manifest):
        run_dir.mkdir(parents=True)
        saved.append(run_dir)

    monkeypatch.setattr("experiment.build_run_slug", lambda config: "slug")
    monkeypatch.setattr(
        "experiment.enrich_run_meta",
        lambda meta, timestamp: meta,
    )
    monkeypatch.setattr("experiment.save_run", save)
    monkeypatch.setattr(
        "experiment.analyze_run_safely",
        lambda run_dir: analyzed.append(run_dir),
    )
    run_dir = experiment._save_results([], metadata)
    assert saved == [run_dir]
    assert analyzed == [run_dir]
