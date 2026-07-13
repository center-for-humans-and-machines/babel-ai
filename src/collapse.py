"""Collapse signal helpers over flat turn tables."""

import pandas as pd


def detect_similarity_spikes(
    turns: pd.DataFrame,
    *,
    column: str = "semantic_similarity_window",
    threshold: float = 0.15,
) -> pd.Series:
    """Flag similarity drops of at least ``threshold`` from prior turn."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if column not in turns.columns:
        return pd.Series(False, index=turns.index, dtype=bool)
    similarities = pd.to_numeric(turns[column], errors="coerce")
    return similarities.diff().le(-threshold).fillna(False).astype(bool)


def mark_generic_fallback_turns(turns: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for ELIZA generic-fallback turns."""
    if "used_generic_fallback" not in turns.columns:
        return pd.Series(False, index=turns.index, dtype=bool)
    return turns["used_generic_fallback"].astype("boolean").fillna(False).astype(
        bool
    )
