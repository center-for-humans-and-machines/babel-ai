"""Tests for legacy CSV migration."""

import pandas as pd

from persistence.legacy_adapter import load_legacy_csv


def test_load_legacy_csv_flattens_analysis(tmp_path):
    path = tmp_path / "drift_experiment_20260713.csv"
    pd.DataFrame(
        {
            "iteration": [4],
            "role": ["assistant"],
            "content": ["Hello"],
            "analysis": [
                "{'word_count': 1, 'analysis_extra': {'novel_metric': 2}}"
            ],
        }
    ).to_csv(path, index=False)

    turns = load_legacy_csv(path)

    assert turns.loc[0, "run_id"] == path.stem
    assert turns.loc[0, "turn_index"] == 4
    assert turns.loc[0, "word_count"] == 1
    assert turns.loc[0, "analysis_extra_novel_metric"] == 2
