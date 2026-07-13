"""Flat, persistable analysis metrics for experiment turns."""

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Analysis metrics stored alongside each turn."""

    word_count: int | None = None
    unique_word_count: int | None = None
    coherence_score: float | None = None
    lexical_similarity: float | None = None
    semantic_similarity: float | None = None
    lexical_similarity_window: float | None = None
    semantic_similarity_window: float | None = None
    token_perplexity: float | None = None
    used_generic_fallback: bool | None = None
    analysis_extra: dict[str, Any] = Field(default_factory=dict)


class MetricComputer(Protocol):
    """Pluggable analyzer invoked for a single turn."""

    def compute(
        self,
        content: str,
        history: list[str],
        *,
        turn_index: int,
    ) -> dict[str, Any]:
        """Return analysis fields for one turn."""
        ...


def parquet_columns() -> list[str]:
    """Return the stable analysis columns written to parquet."""
    return [
        name
        for name in AnalysisResult.model_fields
        if name != "analysis_extra"
    ]


def flatten_analysis(
    analysis: Mapping[str, Any] | BaseModel | None,
) -> dict[str, Any]:
    """Flatten analysis fields and namespace unknown metrics."""
    if analysis is None:
        values: dict[str, Any] = {}
    elif isinstance(analysis, BaseModel):
        values = analysis.model_dump()
    else:
        values = dict(analysis)

    extra = values.pop("analysis_extra", {})
    flattened = {
        column: values.pop(column, None) for column in parquet_columns()
    }
    for key, value in values.items():
        flattened[f"analysis_extra_{key}"] = value
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            flattened[f"analysis_extra_{key}"] = value
    return flattened
