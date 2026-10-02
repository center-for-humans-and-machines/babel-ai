"""Tests for the t-SNE topic-cloud display."""

import pandas as pd

from trajectory.tsne import ProjectionResult
from viz.tsne_display import tsne_status, tsne_trajectory_chart


def _result(status: str = "complete") -> ProjectionResult:
    frame = pd.DataFrame(
        {
            "turn_index": [2, 4, 6],
            "trajectory_index": [0, 1, 2],
            "speaker": ["agent_0"] * 3,
            "content": ["first topic", "middle topic", "final topic"],
            "tsne_x": [0.0, 1.0, 0.5],
            "tsne_y": [0.0, 0.5, 1.0],
        }
    )
    return ProjectionResult(
        trajectory=frame,
        meta={
            "status": status,
            "point_count": 3,
            "reason": None if status == "complete" else "not enough points",
        },
    )


def test_tsne_chart_contains_trajectory_endpoints_and_hover_data():
    chart = tsne_trajectory_chart(_result())
    assert "LLM topic cloud and trajectory" in chart
    assert "Start" in chart
    assert "End" in chart
    assert "middle topic" in chart
    compact = chart.replace(" ", "").lower()
    assert '"showgrid":true' in compact
    assert '"showticklabels":false' not in compact


def test_tsne_chart_uses_equal_axis_scaling():
    chart = tsne_trajectory_chart(_result())
    compact = chart.replace(" ", "")
    assert '"scaleanchor":"x"' in compact
    assert '"scaleratio":1' in compact


def test_tsne_chart_is_empty_when_projection_unavailable():
    assert tsne_trajectory_chart(None) == ""
    assert tsne_trajectory_chart(_result("unavailable")) == ""
    empty = ProjectionResult(
        trajectory=pd.DataFrame(),
        meta={"status": "complete"},
    )
    assert tsne_trajectory_chart(empty) == ""


def test_tsne_status_describes_complete_missing_and_unavailable():
    assert tsne_status(_result()) == "Projected 3 generated LLM turns."
    assert "backfill" in tsne_status(None)
    assert "not enough points" in tsne_status(_result("unavailable"))
