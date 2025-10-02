import os
from pathlib import Path

import pandas as pd

from graphical_analysis.data_utils import (
    ANALYSIS_METRICS,
    aggregate_metric_across_experiments,
    list_experiment_csvs,
    load_experiment_rows,
    merge_metric_across_experiments,
)


def _write_csv(tmpdir: Path, name: str, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    p = tmpdir / name
    df.to_csv(p, index=False)
    return str(p)


def test_load_and_merge_and_aggregate_metrics(tmp_path: Path) -> None:
    root = tmp_path / "structured_analysis"
    exp1 = root / "expA"
    exp2 = root / "expB"
    exp1.mkdir(parents=True)
    exp2.mkdir(parents=True)

    rows_common = [
        {
            "iteration": 0,
            "timestamp": "t0",
            "role": "human",
            "content": "hi",
            "analysis": "{'word_count': 1, 'semantic_similarity': 0.1}",
        },
        {
            "iteration": 1,
            "timestamp": "t1",
            "role": "gpt",
            "content": "bye",
            "analysis": "{'word_count': 1, 'semantic_similarity': 0.2}",
        },
    ]

    _write_csv(exp1, "drift_experiment_1.csv", rows_common)
    _write_csv(exp2, "drift_experiment_2.csv", rows_common)

    found = list_experiment_csvs(str(root))
    assert len(found) == 2

    rows = load_experiment_rows(found[0])
    assert rows[0].word_count == 1
    assert rows[1].semantic_similarity == 0.2

    merged = merge_metric_across_experiments(found, "semantic_similarity")
    assert set(merged["experiment"]) == {"expA", "expB"}
    assert merged["metric"].unique().tolist() == ["semantic_similarity"]

    agg = aggregate_metric_across_experiments(merged)
    assert set(agg["iteration"]) == {0, 1}
    assert {
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "count",
    }.issubset(set(agg.columns))

