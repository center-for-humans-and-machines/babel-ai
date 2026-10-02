"""Tests for trajectory chart helpers."""

import pandas as pd

from viz.charts import (
    available_metrics,
    similarity_metrics,
    trajectory_chart,
    trajectory_charts_for_run,
    trajectory_metrics,
    trajectory_panel_chart,
    _TRAJECTORY_PANELS,
)


def _turns(**metrics: float | None) -> pd.DataFrame:
    row = {
        "turn_index": [2],
        "speaker": ["agent_0"],
        "content": ["reply"],
        "agent_config": ['{"model": "gpt-4o"}'],
    }
    for name, value in metrics.items():
        row[name] = [value]
    return pd.DataFrame(row)


def test_similarity_metrics_returns_all_present_columns():
    turns = _turns(
        semantic_similarity_window=0.8,
        semantic_similarity=0.7,
        lexical_similarity_window=0.6,
        lexical_similarity=0.5,
    )
    assert similarity_metrics(turns) == [
        "semantic_similarity_window",
        "semantic_similarity",
        "lexical_similarity_window",
        "lexical_similarity",
    ]


def test_trajectory_metrics_includes_perplexity():
    turns = _turns(token_perplexity=12.5)
    assert trajectory_metrics(turns) == ["token_perplexity"]


def test_available_metrics_falls_back_to_turn_index():
    turns = pd.DataFrame(
        {
            "turn_index": [0, 1],
            "speaker": ["seed", "agent_0"],
            "content": ["hi", "there"],
            "agent_config": [None, "{}"],
        }
    )
    assert available_metrics(turns) == ["turn_index"]


def test_trajectory_panel_chart_combines_window_and_direct():
    turns = _turns(
        semantic_similarity_window=0.8,
        semantic_similarity=0.7,
    )
    chart = trajectory_panel_chart(
        turns,
        _TRAJECTORY_PANELS[0],
        {},
        include_plotlyjs=False,
    )
    assert "Semantic similarity (window)" in chart
    assert "Semantic similarity (direct)" in chart
    assert '"range":[0,1]' in chart.replace(" ", "")


def test_trajectory_charts_for_run_groups_metrics_into_panels():
    turns = _turns(
        semantic_similarity_window=0.8,
        semantic_similarity=0.7,
        lexical_similarity_window=0.6,
        lexical_similarity=0.5,
        token_perplexity=11.0,
    )
    charts = trajectory_charts_for_run(turns, {})
    assert len(charts) == 3
    assert charts[0]["title"] == "Semantic similarity"
    assert charts[1]["title"] == "Lexical similarity"
    assert charts[2]["title"] == "Token perplexity"
    assert all(item["chart"] for item in charts)


def test_perplexity_panel_keeps_autoscale_y_axis():
    turns = _turns(token_perplexity=15.0)
    chart = trajectory_panel_chart(
        turns,
        _TRAJECTORY_PANELS[2],
        {},
        include_plotlyjs=False,
    )
    assert '"range":[0,1]' not in chart.replace(" ", "")


def test_similarity_charts_use_zero_to_one_y_axis():
    turns = _turns(semantic_similarity_window=0.2)
    chart = trajectory_chart(
        turns,
        "semantic_similarity_window",
        "Semantic similarity (window)",
        {},
        include_plotlyjs=False,
    )
    assert '"range":[0,1]' in chart.replace(" ", "")


def test_turn_index_chart_keeps_autoscale_y_axis():
    turns = pd.DataFrame(
        {
            "turn_index": [0, 3],
            "speaker": ["seed", "agent_0"],
            "content": ["hi", "there"],
            "agent_config": [None, "{}"],
        }
    )
    chart = trajectory_chart(turns, "turn_index", "Turn index", {})
    assert '"range":[0,1]' not in chart.replace(" ", "")
