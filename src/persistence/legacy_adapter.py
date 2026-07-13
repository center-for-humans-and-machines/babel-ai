"""Read legacy ``drift_experiment_*.csv`` files during migration."""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_schema import flatten_analysis, parquet_columns


def load_legacy_csv(path: Path) -> pd.DataFrame:
    """Load a legacy CSV and flatten its serialized analysis column."""
    path = Path(path)
    if not path.name.startswith("drift_experiment_") or path.suffix != ".csv":
        raise ValueError("expected a drift_experiment_*.csv file")

    turns = pd.read_csv(path)
    if "analysis" not in turns.columns:
        raise ValueError("legacy CSV is missing the analysis column")

    rows = []
    for index, record in enumerate(turns.to_dict(orient="records")):
        analysis = _parse_analysis(record.pop("analysis"))
        record.update(flatten_analysis(analysis))
        record["turn_index"] = record.pop("iteration", index)
        record["run_id"] = path.stem
        rows.append(record)
    if not rows:
        return pd.DataFrame(
            columns=["run_id", "turn_index", *parquet_columns()]
        )
    return pd.DataFrame(rows)


def _parse_analysis(value: Any) -> dict[str, Any] | None:
    """Parse JSON or Python dict serialization without evaluating code."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None
