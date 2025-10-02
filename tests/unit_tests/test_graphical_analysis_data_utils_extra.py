from pathlib import Path

import pandas as pd

from graphical_analysis.data_utils import (
    ANALYSIS_METRICS,
    aggregate_metric_across_experiments,
    compute_normalized_metrics,
    list_experiment_csvs,
    load_experiment_rows,
    merge_metric_across_experiments,
)


def _mkcsv(tmp: Path, d: str, name: str, rows: list[dict]) -> str:
    p = tmp / d
    p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    file = p / name
    df.to_csv(file, index=False)
    return str(file)


def test_parsing_edge_cases_and_rolling(tmp_path: Path) -> None:
    r1 = {
        "iteration": 0,
        "timestamp": "t0",
        "role": "gpt",
        "content": "a",
        "analysis": "{'semantic_similarity': '0.3', 'token_perplexity': 'inf'}",
    }
    r2 = {
        "iteration": 1,
        "timestamp": "t1",
        "role": "gpt",
        "content": "b",
        "analysis": "{'semantic_similarity': 0.5, 'token_perplexity': 2.0}",
    }

    f1 = _mkcsv(
        tmp_path,
        "expX",
        "drift_experiment_x.csv",
        [r1, r2],
    )

    rows = load_experiment_rows(f1)
    assert rows[0].token_perplexity == float("inf")
    assert rows[1].semantic_similarity == 0.5

    merged = merge_metric_across_experiments(
        [f1], "semantic_similarity"
    )
    # rolling window = 2 should alter the second value to mean(0.3,0.5)=0.4
    agg = aggregate_metric_across_experiments(merged, window=2)
    it1 = agg[agg["iteration"] == 1]["mean"].iloc[0]
    assert abs(it1 - 0.4) < 1e-6

    # normalize helper
    df = pd.DataFrame(
        {
            "experiment": ["expX", "expX", "expX"],
            "metric": ["m", "m", "m"],
            "iteration": [0, 1, 2],
            "value": [1.0, 2.0, 3.0],
        }
    )
    normed = compute_normalized_metrics(
        df.rename(columns={"value": "m"}), "m"
    )
    assert "m_z" in normed.columns

