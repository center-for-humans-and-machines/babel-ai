"""Tests for canonical run persistence and collapse signals."""

from datetime import datetime

import pandas as pd

from collapse import detect_similarity_spikes, mark_generic_fallback_turns
from models.metrics import AgentMetric
from persistence.run_store import RunManifest, list_runs, load_run, save_run


class AgentMetricStub:
    """Minimal AgentMetric-compatible fixture."""

    def to_dict(self) -> dict[str, object]:
        """Return the AgentMetric serialization shape."""
        return {
            "iteration": 2,
            "timestamp": datetime(2026, 7, 13, 12, 0),
            "role": "assistant",
            "content": "Hello",
            "agent_id": "eliza",
            "agent_config": {"model": "rule-based"},
            "analysis": {
                "word_count": 1,
                "unique_word_count": 1,
                "coherence_score": 1.0,
                "semantic_similarity_window": 0.8,
            },
        }


def test_save_load_and_list_run_flattens_metric_analysis(tmp_path):
    run_dir = tmp_path / "results" / "run-1"

    save_run(
        run_dir,
        [AgentMetricStub()],
        {"run_id": "run-1", "seed": 3},
        manifest=RunManifest(extra={"source": "test"}),
    )

    run = load_run(run_dir)
    assert (run_dir / "turns.parquet").is_file()
    assert run.turns.loc[0, "run_id"] == "run-1"
    assert run.turns.loc[0, "turn_index"] == 2
    assert run.turns.loc[0, "word_count"] == 1
    assert run.turns.loc[0, "agent_config"] == '{"model": "rule-based"}'
    assert run.manifest == RunManifest(extra={"source": "test"})
    assert [record.run_id for record in list_runs(run_dir.parent)] == ["run-1"]


def test_mark_generic_fallback_turns():
    turns = pd.DataFrame({"used_generic_fallback": [False, True, None]})
    mask = mark_generic_fallback_turns(turns)
    assert mask.tolist() == [False, True, False]


def test_detect_similarity_spikes_flags_sharp_drops():
    turns = pd.DataFrame({"semantic_similarity_window": [0.8, 0.7, 0.5]})
    assert detect_similarity_spikes(turns).tolist() == [False, False, True]


def test_scaffolder_trace_is_persisted(tmp_path):
    metric = AgentMetric(
        iteration=0,
        timestamp=datetime(2026, 7, 13, 12, 0),
        role="scaffolder",
        content="Add one concrete detail.",
        agent_id="scaffolder",
        scaffolder_action="thrive_protection",
        scaffolder_informative=True,
        scaffolder_novelty=0.4,
        scaffolder_continuity=0.2,
        scaffolder_content_tokens=12,
        scaffolder_meta_detected=False,
        scaffolder_memory_size=2,
        scaffolder_topic_source_turn=3,
    )
    run_dir = tmp_path / "trace-run"
    save_run(run_dir, [metric], {"run_id": "trace-run"})
    row = load_run(run_dir).turns.iloc[0]
    assert row["scaffolder_action"] == "thrive_protection"
    assert row["scaffolder_memory_size"] == 2
    assert row["scaffolder_topic_source_turn"] == 3
