"""Plot helpers for multi-run trajectory visualization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from persistence.run_store import RunRecord

_DEFAULT_METRIC = "semantic_similarity_window"
_METRIC_LABELS = {
    "semantic_similarity_window": "Semantic similarity (window)",
    "lexical_similarity_window": "Lexical similarity (window)",
    "turn_index": "Turn index",
}


def available_metrics(turns: pd.DataFrame) -> list[str]:
    """Return metric columns that contain at least one non-null value."""
    candidates = list(_METRIC_LABELS.keys())
    present = [column for column in candidates if column in turns.columns]
    usable = [
        column
        for column in present
        if turns[column].notna().any()
    ]
    return usable or ["turn_index"]


def metric_label(metric: str) -> str:
    """Return a human-readable metric title."""
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def overlay_chart(
    records: list[RunRecord],
    metric: str,
    *,
    baseline_id: str | None = None,
) -> str:
    """Plot multiple runs on one trajectory chart."""
    figure = go.Figure()
    for record in records:
        frame = record.turns.sort_values("turn_index")
        if metric not in frame.columns:
            continue
        run_id = record.run_id
        width = 3 if run_id == baseline_id else 1.5
        dash = "solid" if run_id == baseline_id else "dash"
        figure.add_trace(
            go.Scatter(
                x=frame["turn_index"],
                y=frame[metric],
                mode="lines+markers",
                name=run_id,
                line={"width": width, "dash": dash},
            )
        )
    figure.update_layout(
        title=metric_label(metric),
        xaxis_title="Turn",
        yaxis_title=metric_label(metric),
    )
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def aggregate_chart(records: list[RunRecord], metric: str) -> str:
    """Plot mean trajectory with a 95% confidence band."""
    frames = []
    for record in records:
        part = record.turns[["turn_index", metric]].copy()
        part["run_id"] = record.run_id
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("turn_index")[metric].agg(["mean", "std", "count"])
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
        xaxis_title="Turn",
        yaxis_title=metric_label(metric),
    )
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


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
            row[record.run_id] = _stringify(_lookup(configs[record.run_id], key))
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
