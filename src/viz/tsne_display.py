"""Plotly display helpers for saved t-SNE trajectories."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from trajectory.tsne import ProjectionResult


def tsne_status(result: ProjectionResult | None) -> str:
    """Describe projection availability for the run detail page."""
    if result is None:
        return "No t-SNE artifact. Run the t-SNE backfill command."
    status = str(result.meta.get("status", "unknown"))
    if status == "complete":
        count = int(result.meta.get("point_count", len(result.trajectory)))
        return f"Projected {count} generated LLM turns."
    reason = result.meta.get("reason") or "projection unavailable"
    return f"t-SNE {status}: {reason}"


def tsne_trajectory_chart(result: ProjectionResult | None) -> str:
    """Render a topic cloud with the ordered LLM trajectory."""
    if result is None or result.meta.get("status") != "complete":
        return ""
    if result.trajectory.empty:
        return ""
    frame = result.trajectory.sort_values("trajectory_index").copy()
    frame["hover_content"] = frame["content"].map(_truncate)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["tsne_x"],
            y=frame["tsne_y"],
            mode="lines",
            name="Trajectory",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["tsne_x"],
            y=frame["tsne_y"],
            mode="markers",
            name="LLM turns",
            marker={
                "color": frame["trajectory_index"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Order"},
                "size": 9,
            },
            customdata=frame[
                ["turn_index", "speaker", "hover_content"]
            ].to_numpy(),
            hovertemplate=(
                "Turn %{customdata[0]}<br>"
                "Speaker: %{customdata[1]}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        )
    )
    _add_endpoint(figure, frame.iloc[0], "Start", "star")
    if len(frame) > 1:
        _add_endpoint(figure, frame.iloc[-1], "End", "x")
    figure.update_layout(
        title="LLM topic cloud and trajectory",
        xaxis={
            "title": "t-SNE 1",
            "showgrid": True,
            "zeroline": True,
        },
        yaxis={
            "title": "t-SNE 2",
            "showgrid": True,
            "zeroline": True,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h"},
    )
    return figure.to_html(full_html=False, include_plotlyjs=False)


def _add_endpoint(
    figure: go.Figure,
    row: pd.Series,
    label: str,
    symbol: str,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=[row["tsne_x"]],
            y=[row["tsne_y"]],
            mode="markers+text",
            name=label,
            text=[label],
            textposition="top center",
            marker={"symbol": symbol, "size": 14},
            hoverinfo="skip",
        )
    )


def _truncate(value: object, limit: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[:limit - 1].rstrip()}…"
