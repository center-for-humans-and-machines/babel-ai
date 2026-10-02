"""Helpers for rendering RAG scaffolder data in the viewer."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

_RAG_COLUMNS = (
    "rag_used",
    "rag_mode",
    "rag_words",
    "rag_query",
    "rag_source_url",
    "rag_source_title",
    "rag_fallback_reason",
)


def extract_rag_agent_config(meta: dict[str, Any]) -> dict[str, str]:
    """Return RAG scaffolder settings stored in run metadata."""
    config = meta.get("config", {})
    agents = config.get("agents", [])
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if agent.get("type") == "rag_scaffolder":
            return {
                "word_model": str(agent.get("word_model", "")),
                "word_model_path": str(agent.get("word_model_path") or ""),
                "num_words": str(agent.get("num_words", "")),
                "search_backend": str(agent.get("search_backend", "")),
                "search_results": str(agent.get("search_results", "")),
                "novelty_nudge_rate": str(agent.get("novelty_nudge_rate", "")),
            }
    return {}


def has_rag_data(turns: pd.DataFrame) -> bool:
    """Return True when at least one RAG trace value was persisted."""
    rag = _rag_frame(turns)
    if rag.empty:
        return False
    present = [column for column in _RAG_COLUMNS if column in rag.columns]
    if not present:
        return False
    return bool(rag[present].notna().any().any())


def rag_status(turns: pd.DataFrame) -> str:
    """Describe grounded vs fallback RAG turns for a run."""
    rag = _rag_frame(turns)
    if rag is None or rag.empty:
        return "No RAG scaffolder turns in this run."
    if not has_rag_data(turns):
        return (
            "RAG trace metadata is unavailable for this run. Re-run with a "
            "current build to capture rag_* columns."
        )
    total = len(rag)
    grounded = int(_bool_series(rag, "rag_used").sum())
    fallbacks = (
        int(rag["rag_fallback_reason"].notna().sum())
        if "rag_fallback_reason" in rag.columns
        else 0
    )
    text = f"Grounded {grounded} of {total} RAG turns; {fallbacks} fallbacks."
    reason = _top_fallback_reason(rag)
    if reason:
        text += f" Most common: {reason}"
    return text


def rag_turn_rows(turns: pd.DataFrame) -> list[dict[str, object]]:
    """Return RAG-scaffolder turns with parsed provenance fields."""
    rag = _rag_frame(turns)
    if rag is None or rag.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, row in rag.sort_values("turn_index").iterrows():
        rows.append(
            {
                "turn_index": _clean(row.get("turn_index")),
                "rag_mode": _clean(row.get("rag_mode")),
                "status": _row_status(row),
                "rag_words": ", ".join(_parse_words(row.get("rag_words"))),
                "rag_query": _clean(row.get("rag_query")),
                "rag_source_url": _clean(row.get("rag_source_url")),
                "rag_source_title": _clean(row.get("rag_source_title")),
                "rag_fallback_reason": _clean(row.get("rag_fallback_reason")),
                "content": _clean(row.get("content")),
            }
        )
    return rows


def rag_source_domains(turns: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize grounded sources by URL domain."""
    rag = _rag_frame(turns)
    if rag is None or "rag_source_url" not in rag.columns:
        return []
    urls = rag["rag_source_url"].dropna()
    counts: dict[str, int] = {}
    for url in urls:
        domain = str(url).split("//")[-1].split("/")[0]
        if domain:
            counts[domain] = counts.get(domain, 0) + 1
    return [
        {"domain": domain, "count": count}
        for domain, count in sorted(counts.items())
    ]


def _rag_frame(turns: pd.DataFrame) -> pd.DataFrame:
    if "speaker" not in turns.columns:
        return turns.iloc[0:0]
    return turns[turns["speaker"] == "rag_scaffolder"]


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].eq(True)


def _row_status(row: pd.Series) -> str:
    reason = row.get("rag_fallback_reason")
    if isinstance(reason, str) and reason.strip():
        return "fallback"
    used = row.get("rag_used")
    if used is None:
        return "not_used"
    try:
        is_missing = pd.isna(used)
    except (TypeError, ValueError):
        is_missing = False
    if bool(is_missing):
        return "not_used"
    return "grounded" if bool(used) else "not_used"


def _top_fallback_reason(rag: pd.DataFrame) -> str:
    if "rag_fallback_reason" not in rag.columns:
        return ""
    reasons = rag["rag_fallback_reason"].dropna()
    if reasons.empty:
        return ""
    return str(reasons.value_counts().index[0])


def _parse_words(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return value
