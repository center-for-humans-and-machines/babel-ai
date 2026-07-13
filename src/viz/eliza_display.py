"""Helpers for rendering ELIZA branch data in the viewer."""

from __future__ import annotations

from typing import Any

import pandas as pd


def extract_eliza_agent_config(meta: dict[str, Any]) -> dict[str, str]:
    """Return ELIZA partner settings stored in run metadata."""
    config = meta.get("config", {})
    agents = config.get("agents", [])
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if (
            agent.get("type") == "rule_based"
            or agent.get("partner") == "eliza"
        ):
            return {
                "partner": str(agent.get("partner", "eliza")),
                "generic_intervention": str(
                    agent.get("generic_intervention", "passthrough")
                ),
                "topic_switch_probability": str(
                    agent.get("topic_switch_probability", "0.5")
                ),
                "feed_sources": ", ".join(
                    agent.get("feed_sources", ["topic_bank"])
                ),
            }
    return {}


def has_eliza_branch_data(turns: pd.DataFrame) -> bool:
    """Return True when at least one ELIZA branch was persisted."""
    if "eliza_branch" not in turns.columns:
        return False
    eliza = turns[turns["speaker"] == "eliza"]
    if eliza.empty:
        return False
    return eliza["eliza_branch"].notna().any()


def eliza_turn_rows(turns: pd.DataFrame) -> list[dict[str, object]]:
    """Return ELIZA-only transcript rows with branch metadata."""
    if "speaker" not in turns.columns:
        return []
    frame = turns[turns["speaker"] == "eliza"].sort_values("turn_index")
    fields = (
        "turn_index",
        "eliza_branch",
        "eliza_keyword",
        "eliza_reassembly",
        "used_generic_fallback",
        "content",
    )
    present = [field for field in fields if field in frame.columns]
    rows = frame.reindex(columns=present).fillna("").to_dict(orient="records")
    for row in rows:
        for field in fields:
            row.setdefault(field, "")
        if row.get("used_generic_fallback") in ("", None):
            row["used_generic_fallback"] = ""
        else:
            row["used_generic_fallback"] = str(row["used_generic_fallback"])
    return rows


def eliza_branch_status(turns: pd.DataFrame) -> str:
    """Describe whether branch metadata is available for this run."""
    eliza_count = (
        len(turns[turns["speaker"] == "eliza"])
        if "speaker" in turns.columns
        else 0
    )
    if eliza_count == 0:
        return "No ELIZA turns in this run."
    if not has_eliza_branch_data(turns):
        return (
            "ELIZA branch metadata is unavailable for this run. "
            "Re-run the experiment with a current build to capture "
            "keyword, memory, and generic fallback paths."
        )
    tracked = (
        turns.loc[turns["speaker"] == "eliza", "eliza_branch"].notna().sum()
    )
    return f"Tracked {tracked} of {eliza_count} ELIZA turns."
