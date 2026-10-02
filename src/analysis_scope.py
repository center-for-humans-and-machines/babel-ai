"""Turn selection for drift analysis and trajectory visualization."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

import pandas as pd


class AnalysisScope(str, Enum):
    """Which transcript turns feed similarity and trajectory analysis."""

    LLM_ONLY = "llm_only"
    ALL_TURNS = "all_turns"


_TRAJECTORY_COLUMNS = (
    "turn_index",
    "trajectory_index",
    "speaker",
    "content",
)


def is_llm_turn_value(agent_config: Any) -> bool:
    """Return True when a persisted or runtime config marks an LLM turn."""
    if agent_config is None:
        return False
    if isinstance(agent_config, float) and pd.isna(agent_config):
        return False
    if isinstance(agent_config, Mapping) and not agent_config:
        return True
    text = str(agent_config).strip()
    return text not in ("", "null", "None", "nan")


def resolve_analysis_scope(meta: Mapping[str, Any] | None) -> AnalysisScope:
    """Read analysis scope from run metadata, defaulting to LLM-only."""
    if not meta:
        return AnalysisScope.LLM_ONLY
    config = meta.get("config", {})
    if not isinstance(config, Mapping):
        return AnalysisScope.LLM_ONLY
    analyzer = config.get("analyzer_config", {})
    if not isinstance(analyzer, Mapping):
        return AnalysisScope.LLM_ONLY
    raw = analyzer.get("analysis_scope", AnalysisScope.LLM_ONLY.value)
    try:
        return AnalysisScope(str(raw))
    except ValueError:
        return AnalysisScope.LLM_ONLY


def select_analysis_turns(
    turns: pd.DataFrame,
    scope: AnalysisScope = AnalysisScope.LLM_ONLY,
) -> pd.DataFrame:
    """Return turns included in analysis, ordered with trajectory_index."""
    if turns.empty:
        return pd.DataFrame(columns=list(_TRAJECTORY_COLUMNS))
    if scope is AnalysisScope.ALL_TURNS:
        if "turn_index" not in turns.columns:
            return pd.DataFrame(columns=list(_TRAJECTORY_COLUMNS))
        frame = turns.sort_values("turn_index").reset_index(drop=True)
        frame = frame.copy()
        frame.insert(1, "trajectory_index", range(len(frame)))
        return frame

    required = {"turn_index", "speaker", "content", "agent_config"}
    missing = required - set(turns.columns)
    if missing:
        return pd.DataFrame(columns=list(_TRAJECTORY_COLUMNS))

    mask = turns["agent_config"].map(is_llm_turn_value)
    selected = turns.loc[
        mask,
        ["turn_index", "speaker", "content"],
    ].copy()
    selected = selected.sort_values("turn_index").reset_index(drop=True)
    selected.insert(1, "trajectory_index", range(len(selected)))
    return selected


def trajectory_plot_frame(
    turns: pd.DataFrame,
    meta: Mapping[str, Any] | None = None,
    *,
    scope: AnalysisScope | None = None,
) -> pd.DataFrame:
    """Filter turns for trajectory charts and pick the x-axis column."""
    resolved = scope or resolve_analysis_scope(meta)
    frame = select_analysis_turns(turns, resolved)
    if frame.empty:
        return frame
    x_column = (
        "trajectory_index"
        if resolved is AnalysisScope.LLM_ONLY
        else "turn_index"
    )
    return frame.assign(plot_index=frame[x_column])


def analysis_contents_for_metrics(
    metrics: Sequence[Any],
    *,
    scope: AnalysisScope,
    through_index: int,
) -> list[str] | None:
    """Build analyzer input for one metric index, or skip when excluded."""
    if through_index < 0 or through_index >= len(metrics):
        return None
    if scope is AnalysisScope.ALL_TURNS:
        return [metric.content for metric in metrics[: through_index + 1]]

    current = metrics[through_index]
    if not _metric_is_llm_turn(current):
        return None
    return [
        metric.content
        for metric in metrics[: through_index + 1]
        if _metric_is_llm_turn(metric)
    ]


def _metric_is_llm_turn(metric: Any) -> bool:
    if isinstance(metric, Mapping):
        return is_llm_turn_value(metric.get("agent_config"))
    return is_llm_turn_value(getattr(metric, "agent_config", None))
