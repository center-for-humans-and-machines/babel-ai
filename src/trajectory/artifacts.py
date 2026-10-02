"""Persistence for independent t-SNE trajectory artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from analysis_scope import AnalysisScope, resolve_analysis_scope
from trajectory.tsne import (
    EmbeddingProvider,
    ProjectionResult,
    analyze_tsne,
)

logger = logging.getLogger(__name__)

TRAJECTORY_FILENAME = "tsne_trajectory.parquet"
META_FILENAME = "tsne_meta.json"


def analyze_run(
    run_dir: Path,
    *,
    embedder: EmbeddingProvider | None = None,
    random_state: int = 0,
    scope: AnalysisScope | None = None,
) -> ProjectionResult:
    """Analyze a completed run and persist its t-SNE artifacts."""
    run_dir = Path(run_dir)
    turns = pd.read_parquet(run_dir / "turns.parquet")
    resolved_scope = scope or _scope_from_run_dir(run_dir)
    result = analyze_tsne(
        turns,
        embedder=embedder,
        random_state=random_state,
        scope=resolved_scope,
    )
    save_projection(run_dir, result)
    return result


def analyze_run_safely(
    run_dir: Path,
    *,
    random_state: int = 0,
) -> ProjectionResult | None:
    """Run optional post-processing without invalidating a completed run."""
    try:
        return analyze_run(run_dir, random_state=random_state)
    except Exception as exc:
        logger.warning("t-SNE post-processing failed for %s: %s", run_dir, exc)
        _write_meta(
            Path(run_dir) / META_FILENAME,
            {
                "status": "failed",
                "reason": str(exc),
                "random_state": random_state,
            },
        )
        return None


def save_projection(run_dir: Path, result: ProjectionResult) -> None:
    """Persist coordinates and metadata beside canonical run artifacts."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result.trajectory.to_parquet(
        run_dir / TRAJECTORY_FILENAME,
        index=False,
    )
    _write_meta(run_dir / META_FILENAME, result.meta)


def load_projection(run_dir: Path) -> ProjectionResult | None:
    """Load saved t-SNE artifacts, or return no result when absent."""
    run_dir = Path(run_dir)
    trajectory_path = run_dir / TRAJECTORY_FILENAME
    meta_path = run_dir / META_FILENAME
    if not trajectory_path.is_file() or not meta_path.is_file():
        return None
    trajectory = pd.read_parquet(trajectory_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ProjectionResult(trajectory=trajectory, meta=meta)


def analyze_results_root(
    results_root: Path,
    *,
    random_state: int = 0,
) -> dict[str, str]:
    """Backfill every completed run below a results directory."""
    root = Path(results_root)
    statuses: dict[str, str] = {}
    if not root.is_dir():
        return statuses
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (run_dir / "turns.parquet").is_file():
            continue
        result = analyze_run_safely(
            run_dir,
            random_state=random_state,
        )
        statuses[run_dir.name] = (
            str(result.meta["status"]) if result else "failed"
        )
    return statuses


def _write_meta(path: Path, meta: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _scope_from_run_dir(run_dir: Path) -> AnalysisScope:
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.is_file():
        return AnalysisScope.LLM_ONLY
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return resolve_analysis_scope(meta)
