"""Canonical per-run artifact I/O under ``results/{run_id}/``."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_schema import flatten_analysis, parquet_columns

_CORE_COLUMNS = (
    "run_id",
    "turn_index",
    "timestamp",
    "role",
    "speaker",
    "content",
    "agent_id",
    "eliza_branch",
    "eliza_keyword",
    "eliza_reassembly",
    "used_generic_fallback",
    "scaffolder_action",
    "scaffolder_informative",
    "scaffolder_novelty",
    "scaffolder_continuity",
    "scaffolder_content_tokens",
    "scaffolder_meta_detected",
    "scaffolder_memory_size",
    "scaffolder_topic_source_turn",
)


@dataclass
class RunManifest:
    """Schema marker for a persisted run."""

    schema_version: str = "1"
    extra: dict[str, Any] | None = None


@dataclass
class RunRecord:
    """Loaded run bundle."""

    run_id: str
    run_dir: Path
    turns: pd.DataFrame
    meta: dict[str, Any]
    manifest: RunManifest | None = None


def save_run(
    run_dir: Path,
    turns: pd.DataFrame | Iterable[Mapping[str, Any] | Any],
    meta: Mapping[str, Any],
    *,
    manifest: RunManifest | None = None,
) -> Path:
    """Write flat turns, metadata, and an optional manifest to ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    flattened_turns = _prepare_turns(turns, run_dir.name)
    flattened_turns.to_parquet(run_dir / "turns.parquet", index=False)
    _write_json(run_dir / "meta.json", dict(meta))
    if manifest is not None:
        _write_json(run_dir / "manifest.json", asdict(manifest))
    return run_dir


def load_run(run_dir: Path) -> RunRecord:
    """Load a completed run directory."""
    run_dir = Path(run_dir)
    turns = pd.read_parquet(run_dir / "turns.parquet")
    with (run_dir / "meta.json").open(encoding="utf-8") as file:
        meta = json.load(file)
    manifest_path = run_dir / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as file:
            manifest = RunManifest(**json.load(file))
    run_id = str(meta.get("run_id", run_dir.name))
    return RunRecord(
        run_id=run_id,
        run_dir=run_dir,
        turns=turns,
        meta=meta,
        manifest=manifest,
    )


def list_runs(results_root: Path) -> list[RunRecord]:
    """Return completed runs below ``results_root``, sorted by run ID."""
    results_root = Path(results_root)
    if not results_root.is_dir():
        return []
    run_dirs = sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and (path / "turns.parquet").is_file()
        and (path / "meta.json").is_file()
    )
    return [load_run(run_dir) for run_dir in run_dirs]


def _prepare_turns(
    turns: pd.DataFrame | Iterable[Mapping[str, Any] | Any],
    run_id: str,
) -> pd.DataFrame:
    """Convert metrics and nested analysis values into flat turn rows."""
    if isinstance(turns, pd.DataFrame):
        records = turns.to_dict(orient="records")
    else:
        records = [_metric_record(turn) for turn in turns]
    flattened = [
        _flatten_turn(record, run_id, index)
        for index, record in enumerate(records)
    ]
    frame = pd.DataFrame(flattened)
    if frame.empty:
        frame = pd.DataFrame(columns=[*_CORE_COLUMNS, *parquet_columns()])
    return _ordered_columns(frame)


def _metric_record(metric: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert an AgentMetric-like value to its serialization dictionary."""
    if isinstance(metric, Mapping):
        return dict(metric)
    to_dict = getattr(metric, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            "turns must contain mappings or objects with to_dict()"
        )
    return dict(to_dict())


def _flatten_turn(
    record: Mapping[str, Any],
    run_id: str,
    index: int,
) -> dict[str, Any]:
    """Flatten one metric record while preserving transcript fields."""
    row = dict(record)
    analysis = row.pop("analysis", None)
    row.update(flatten_analysis(analysis))
    row["turn_index"] = row.pop("turn_index", row.pop("iteration", index))
    record_run_id = row.get("run_id")
    if record_run_id is not None and record_run_id != run_id:
        raise ValueError("turn run_id must match the run directory name")
    row["run_id"] = run_id
    for column in _CORE_COLUMNS:
        row.setdefault(column, None)
    return {key: _serialize_nested(value) for key, value in row.items()}


def _ordered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Place identity and known analysis fields before extra columns."""
    preferred = [*_CORE_COLUMNS, *parquet_columns()]
    present = [column for column in preferred if column in frame.columns]
    remaining = sorted(
        column for column in frame.columns if column not in present
    )
    return frame.loc[:, [*present, *remaining]]


def _serialize_nested(value: Any) -> Any:
    """Encode nested metadata so every turn column remains scalar."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write readable metadata with datetime-safe serialization."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, default=str, sort_keys=True)
