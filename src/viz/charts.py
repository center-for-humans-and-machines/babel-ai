"""Plot helpers for multi-run trajectory visualization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis_scope import (
    AnalysisScope,
    resolve_analysis_scope,
    trajectory_plot_frame,
)
from persistence.run_store import RunRecord

_DEFAULT_METRIC = "semantic_similarity_window"
_SIMILARITY_METRICS = (
    "semantic_similarity_window",
    "semantic_similarity",
    "lexical_similarity_window",
    "lexical_similarity",
)
_TRAJECTORY_METRICS = _SIMILARITY_METRICS + ("token_perplexity",)
_TRAJECTORY_PANELS = (
    {
        "title": "Semantic similarity",
        "metrics": ("semantic_similarity_window", "semantic_similarity"),
        "fixed_range": (0, 1),
    },
    {
        "title": "Lexical similarity",
        "metrics": ("lexical_similarity_window", "lexical_similarity"),
        "fixed_range": (0, 1),
    },
    {
        "title": "Token perplexity",
        "metrics": ("token_perplexity",),
        "fixed_range": None,
    },
)
_METRIC_LABELS = {
    "semantic_similarity_window": "Semantic similarity (window)",
    "semantic_similarity": "Semantic similarity (direct)",
    "lexical_similarity_window": "Lexical similarity (window)",
    "lexical_similarity": "Lexical similarity (direct)",
    "token_perplexity": "Token perplexity",
    "turn_index": "Turn index",
}


def _usable_metrics(
    turns: pd.DataFrame, metrics: tuple[str, ...]
) -> list[str]:
    """Return panel metric columns that contain at least one value."""
    present = [metric for metric in metrics if metric in turns.columns]
    return [metric for metric in present if turns[metric].notna().any()]


def similarity_metrics(turns: pd.DataFrame) -> list[str]:
    """Return usable similarity columns in stable display order."""
    return _usable_metrics(turns, _SIMILARITY_METRICS)


def trajectory_metrics(turns: pd.DataFrame) -> list[str]:
    """Return usable trajectory metric columns in stable display order."""
    return _usable_metrics(turns, _TRAJECTORY_METRICS)


def available_metrics(turns: pd.DataFrame) -> list[str]:
    """Return metric columns that contain at least one non-null value."""
    usable = trajectory_metrics(turns)
    return usable or ["turn_index"]


def metric_label(metric: str) -> str:
    """Return a human-readable metric title."""
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _apply_similarity_y_axis(figure: go.Figure, metric: str) -> None:
    """Pin similarity charts to a shared 0–1 scale."""
    if metric in _SIMILARITY_METRICS:
        figure.update_layout(yaxis={"range": [0, 1]})


def metric_frame_for_plot(turns: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return turn_index plus one metric column without duplicate labels."""
    frame = turns.loc[:, ["turn_index"]].copy()
    if metric != "turn_index" and metric in turns.columns:
        frame[metric] = turns[metric]
    elif metric == "turn_index":
        frame["metric_value"] = turns["turn_index"]
    return frame


def trajectory_panel_chart(
    turns: pd.DataFrame,
    panel: dict[str, object],
    meta: dict,
    *,
    include_plotlyjs: bool | str = False,
) -> str:
    """Return one chart with windowed and direct series in the same axes."""
    frame = trajectory_plot_frame(turns, meta)
    if frame.empty:
        return ""
    metrics = _usable_metrics(turns, tuple(panel["metrics"]))
    if not metrics:
        return ""

    x_label = _x_axis_label(meta)
    figure = go.Figure()
    for metric in metrics:
        plot = frame.merge(
            metric_frame_for_plot(turns, metric),
            on="turn_index",
            how="left",
        )
        figure.add_trace(
            go.Scatter(
                x=plot["plot_index"],
                y=plot[metric],
                mode="lines+markers",
                name=metric_label(metric),
            )
        )
    title = str(panel["title"])
    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=title,
    )
    fixed_range = panel.get("fixed_range")
    if fixed_range is not None:
        figure.update_layout(yaxis={"range": list(fixed_range)})
    plotly_js = "cdn" if include_plotlyjs else False
    return figure.to_html(full_html=False, include_plotlyjs=plotly_js)


def trajectory_chart(
    turns: pd.DataFrame,
    metric: str,
    title: str,
    meta: dict,
    *,
    include_plotlyjs: bool | str = False,
) -> str:
    """Return one embeddable similarity trajectory chart."""
    frame = trajectory_plot_frame(turns, meta)
    if frame.empty:
        return ""
    if metric != "turn_index" and metric not in turns.columns:
        return ""
    plot = frame.merge(
        metric_frame_for_plot(turns, metric),
        on="turn_index",
        how="left",
    )
    y_column = "metric_value" if metric == "turn_index" else metric
    x_label = _x_axis_label(meta)
    figure = px.line(
        plot,
        x="plot_index",
        y=y_column,
        markers=True,
        title=title,
        labels={"plot_index": x_label, y_column: title},
    )
    _apply_similarity_y_axis(figure, metric)
    plotly_js = "cdn" if include_plotlyjs else False
    return figure.to_html(full_html=False, include_plotlyjs=plotly_js)


def trajectory_charts_for_run(
    turns: pd.DataFrame,
    meta: dict,
) -> list[dict[str, str]]:
    """Build grouped analysis charts for one run detail page."""
    charts: list[dict[str, str]] = []
    include_plotly = True
    for panel in _TRAJECTORY_PANELS:
        title = str(panel["title"])
        chart = trajectory_panel_chart(
            turns,
            panel,
            meta,
            include_plotlyjs=include_plotly,
        )
        if not chart:
            continue
        charts.append({"title": title, "chart": chart})
        include_plotly = False

    if charts:
        return charts

    fallback = trajectory_chart(
        turns,
        "turn_index",
        "Turn index",
        meta,
        include_plotlyjs=True,
    )
    if fallback:
        charts.append({"title": "Turn index", "chart": fallback})
    return charts


def overlay_chart(
    records: list[RunRecord],
    metric: str,
    *,
    baseline_id: str | None = None,
) -> str:
    """Plot multiple runs on one trajectory chart."""
    figure = go.Figure()
    x_label = _x_axis_label(records[0].meta) if records else "LLM turn"
    for record in records:
        plot_frame = trajectory_plot_frame(record.turns, record.meta)
        if plot_frame.empty:
            continue
        if metric != "turn_index" and metric not in record.turns.columns:
            continue
        plot = plot_frame.merge(
            metric_frame_for_plot(record.turns, metric),
            on="turn_index",
            how="left",
        )
        y_column = "metric_value" if metric == "turn_index" else metric
        run_id = record.run_id
        width = 3 if run_id == baseline_id else 1.5
        dash = "solid" if run_id == baseline_id else "dash"
        figure.add_trace(
            go.Scatter(
                x=plot["plot_index"],
                y=plot[y_column],
                mode="lines+markers",
                name=run_id,
                line={"width": width, "dash": dash},
            )
        )
    figure.update_layout(
        title=metric_label(metric),
        xaxis_title=x_label,
        yaxis_title=metric_label(metric),
    )
    _apply_similarity_y_axis(figure, metric)
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def aggregate_chart(records: list[RunRecord], metric: str) -> str:
    """Plot mean trajectory with a 95% confidence band."""
    frames = []
    for record in records:
        plot_frame = trajectory_plot_frame(record.turns, record.meta)
        if plot_frame.empty:
            continue
        if metric != "turn_index" and metric not in record.turns.columns:
            continue
        part = plot_frame.merge(
            metric_frame_for_plot(record.turns, metric),
            on="turn_index",
            how="left",
        )
        y_column = "metric_value" if metric == "turn_index" else metric
        part["run_id"] = record.run_id
        frames.append(
            part[["plot_index", y_column, "run_id"]].rename(
                columns={y_column: metric}
            )
        )
    if not frames:
        return ""
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("plot_index")[metric].agg(
        ["mean", "std", "count"]
    )
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"].clip(1))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grouped.index,
            y=grouped["mean"] + grouped["ci95"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grouped.index,
            y=grouped["mean"] - grouped["ci95"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            name="95% CI",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grouped.index,
            y=grouped["mean"],
            mode="lines+markers",
            name="Mean",
        )
    )
    figure.update_layout(
        title=f"Mean {metric_label(metric)}",
        xaxis_title=_x_axis_label(records[0].meta),
        yaxis_title=metric_label(metric),
    )
    _apply_similarity_y_axis(figure, metric)
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def _x_axis_label(meta: dict) -> str:
    if resolve_analysis_scope(meta) is AnalysisScope.ALL_TURNS:
        return "Turn"
    return "LLM turn"


def eliza_branch_chart(turns: pd.DataFrame) -> str:
    """Plot ELIZA branch usage across ELIZA turns."""
    if "eliza_branch" not in turns.columns:
        return ""
    frame = turns[turns["eliza_branch"].notna()].copy()
    if frame.empty:
        return ""
    frame = frame.sort_values("turn_index")
    figure = px.scatter(
        frame,
        x="turn_index",
        y="eliza_branch",
        color="eliza_branch",
        title="ELIZA decision branches by turn",
        labels={
            "turn_index": "Turn",
            "eliza_branch": "Branch",
        },
    )
    figure.update_layout(showlegend=False, height=320)
    return figure.to_html(full_html=False, include_plotlyjs=False)


def eliza_branch_summary(turns: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize how often each ELIZA branch fired."""
    if "eliza_branch" not in turns.columns:
        return []
    frame = turns[turns["eliza_branch"].notna()]
    if frame.empty:
        return []
    counts = frame["eliza_branch"].value_counts().reset_index()
    counts.columns = ["branch", "count"]
    return counts.to_dict(orient="records")


def eliza_branch_bar_chart(turns: pd.DataFrame) -> str:
    """Bar chart of ELIZA branch counts."""
    summary = eliza_branch_summary(turns)
    if not summary:
        return ""
    frame = pd.DataFrame(summary)
    figure = px.bar(
        frame,
        x="branch",
        y="count",
        title="ELIZA branch counts",
        labels={"branch": "Branch", "count": "Turns"},
    )
    figure.update_layout(height=320)
    return figure.to_html(full_html=False, include_plotlyjs=False)


def config_diff_rows(records: list[RunRecord]) -> list[dict[str, str]]:
    """Flatten config keys across selected runs for side-by-side compare."""
    keys: set[str] = set()
    configs: dict[str, dict] = {}
    for record in records:
        config = record.meta.get("config", {})
        configs[record.run_id] = config
        keys.update(_flatten_keys(config))
    rows = []
    for key in sorted(keys):
        row = {"key": key}
        for record in records:
            row[record.run_id] = _stringify(
                _lookup(configs[record.run_id], key)
            )
        rows.append(row)
    return rows


def _flatten_keys(value: dict, prefix: str = "") -> list[str]:
    """Collect dotted paths for nested config dictionaries."""
    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            paths.extend(_flatten_keys(item, path))
        else:
            paths.append(path)
    return paths


def _lookup(config: dict, dotted: str):
    """Read a dotted path from a nested config dictionary."""
    current: object = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _stringify(value: object) -> str:
    """Render config values for HTML tables."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)
