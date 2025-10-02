"""Data utilities for graphical analysis.

These functions prepare experiment CSV data produced by
`babel_ai.analyzer.SimilarityAnalyzer` for plotting. Plotting code should live
in notebooks or dedicated plotting modules; this file avoids any plotting
dependencies beyond pandas and numpy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# All metric keys present in the CSV "analysis" column dictionary
ANALYSIS_METRICS: Tuple[str, ...] = (
    "word_count",
    "unique_word_count",
    "coherence_score",
    "token_perplexity",
    "lexical_similarity",
    "semantic_similarity",
    "lexical_similarity_window",
    "semantic_similarity_window",
)


@dataclass(frozen=True)
class ExperimentRow:
    """One parsed CSV row with flattened metrics."""

    experiment_dir: str
    iteration: int
    timestamp: str
    role: str
    content: str
    agent_id: Optional[str]
    # Flattened metrics
    word_count: Optional[float]
    unique_word_count: Optional[float]
    coherence_score: Optional[float]
    token_perplexity: Optional[float]
    lexical_similarity: Optional[float]
    semantic_similarity: Optional[float]
    lexical_similarity_window: Optional[float]
    semantic_similarity_window: Optional[float]


def _safe_parse_analysis(raw: str) -> Dict[str, float]:
    """Parse the "analysis" cell which contains a Python-like dict string.

    The saved CSV uses single-quoted Python dict repr. Attempt JSON repair.
    """

    if not isinstance(raw, str) or not raw:
        return {}

    # Try fast path: replace single quotes with double quotes for JSON
    # and convert special float values.
    text = raw.strip()
    text = text.replace("None", "null")
    text = text.replace("inf", "Infinity")
    # heuristic: only replace quotes around keys/strings, not inside numbers
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {k: _to_optional_float(v) for k, v in data.items()}
        return {}
    except Exception:
        # Fallback: very permissive eval-safe parse
        try:
            # literal_eval would be ideal, but we avoid importing ast here to
            # keep surface small; a minimal guarded eval with replacements.
            # We keep a strict local scope.
            from ast import literal_eval

            data = literal_eval(raw)
            if isinstance(data, dict):
                return {k: _to_optional_float(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _to_optional_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "none", "nan"}:
                return None
            if v in {"inf", "+inf", "infinity", "+infinity"}:
                return float("inf")
            if v in {"-inf", "-infinity"}:
                return float("-inf")
            return float(value)
    except Exception:
        return None
    return None


def list_experiment_csvs(root_dir: str) -> List[str]:
    """Return paths to all CSVs in a structured_analysis-like directory tree.

    The expected shape is `root_dir/**/drift_experiment_*.csv`.
    """

    matches: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".csv") and fn.startswith("drift_experiment_"):
                matches.append(os.path.join(dirpath, fn))
    matches.sort()
    return matches


def load_experiment_rows(csv_path: str) -> List[ExperimentRow]:
    """Load a single experiment CSV and return flattened rows.

    Only rows with a valid integer `iteration` are included.
    """

    df = pd.read_csv(csv_path)
    required = {"iteration", "timestamp", "role", "content", "analysis"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {csv_path}")

    experiment_dir = os.path.dirname(csv_path)
    rows: List[ExperimentRow] = []
    for _, r in df.iterrows():
        try:
            iteration = int(r["iteration"])  # type: ignore[index]
        except Exception:
            continue

        metrics = _safe_parse_analysis(str(r.get("analysis", "")))
        values = {
            m: _to_optional_float(metrics.get(m)) for m in ANALYSIS_METRICS
        }

        rows.append(
            ExperimentRow(
                experiment_dir=experiment_dir,
                iteration=iteration,
                timestamp=str(r.get("timestamp", "")),
                role=str(r.get("role", "")),
                content=str(r.get("content", "")),
                agent_id=(
                    None
                    if "agent_id" not in df.columns
                    else _as_optional_str(r.get("agent_id"))
                ),
                **values,  # type: ignore[arg-type]
            )
        )
    rows.sort(key=lambda x: x.iteration)
    return rows


def _as_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _rows_to_dataframe(rows: Sequence[ExperimentRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def merge_metric_across_experiments(
    csv_paths: Sequence[str], metric: str
) -> pd.DataFrame:
    """Create a long DataFrame with columns:
    ["experiment", "iteration", "metric", "value"].
    """

    if metric not in ANALYSIS_METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        rows = load_experiment_rows(p)
        df = _rows_to_dataframe(rows)
        if metric not in df.columns:
            continue
        exp_name = os.path.basename(os.path.dirname(p))
        frames.append(
            pd.DataFrame(
                {
                    "experiment": exp_name,
                    "iteration": df["iteration"].to_numpy(),
                    "metric": metric,
                    "value": df[metric].to_numpy(dtype=float),
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["experiment", "iteration", "metric", "value"]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(["metric", "experiment", "iteration"], inplace=True)
    return merged


def aggregate_metric_across_experiments(
    merged: pd.DataFrame, window: int = 1
) -> pd.DataFrame:
    """Aggregate a merged metric DataFrame into mean and ci columns.

    Returns columns: [metric, iteration, mean, std, ci_low, ci_high, count]
    where CI is normal approx 95% based on std/sqrt(n).
    """

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "iteration",
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "count",
            ]
        )

    def _compute_ci(s: pd.Series) -> Tuple[float, float, float, int]:
        arr = s.to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            return np.nan, np.nan, np.nan, 0
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else np.nan
        ci = 1.96 * se if n > 0 else np.nan
        return mean, std, ci, n

    # Optional rolling by iteration per experiment before aggregating
    df = merged.copy()
    if window and window > 1:
        df = df.sort_values(["experiment", "iteration"]).reset_index(drop=True)
        df["value"] = (
            df.groupby("experiment", group_keys=False)["value"]
            .apply(lambda s: s.rolling(window, min_periods=1).mean())
            .reset_index(drop=True)
        )

    agg = (
        df.groupby(["metric", "iteration"])["value"]  # type: ignore[arg-type]
        .apply(_compute_ci)
        .reset_index()
    )

    # split tuple columns
    agg[["mean", "std", "ci", "count"]] = pd.DataFrame(
        agg["value"].tolist(), index=agg.index
    )
    agg.drop(columns=["value"], inplace=True)
    agg["ci_low"] = agg["mean"] - agg["ci"]
    agg["ci_high"] = agg["mean"] + agg["ci"]
    agg.drop(columns=["ci"], inplace=True)
    agg.sort_values(["metric", "iteration"], inplace=True)
    return agg


# ---- Alignment from first LLM-only generation (uuid4 agent_id) ----


def _is_uuid4_like(text: Optional[str]) -> bool:
    if text is None:
        return False
    s = str(text).strip()
    if len(s) < 36:
        return False
    # very light-weight check for uuid4 pattern 8-4-4-4-12 hex
    import re

    return bool(
        re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",  # noqa: E501
            s,
        )
    )


def compute_relative_iteration_from_first_llm(
    rows: Sequence[ExperimentRow],
) -> pd.DataFrame:
    """Return a DataFrame with columns of
    `_rows_to_dataframe` plus `rel_iter_from_llm`.

    The `rel_iter_from_llm` is 0 at the first row
    whose `agent_id` looks like a uuid4
    (assumed to be LLM-only generation), negative before, positive after.
    If no such row exists, the column will be NaN.
    """

    df = _rows_to_dataframe(rows)
    if df.empty:
        df["rel_iter_from_llm"] = np.nan
        return df

    # Locate first uuid4-like agent_id
    first_idx: Optional[int] = None
    for i, val in enumerate(df.get("agent_id", [])):
        if _is_uuid4_like(val):
            first_idx = i
            break

    if first_idx is None:
        df["rel_iter_from_llm"] = np.nan
        return df

    first_iter = int(df.iloc[first_idx]["iteration"])  # type: ignore[index]
    df["rel_iter_from_llm"] = df["iteration"].astype(int) - first_iter
    return df


def merge_metric_aligned_from_first_llm(
    csv_paths: Sequence[str], metric: str
) -> pd.DataFrame:
    """Create long DataFrame using relative iteration from first LLM-only gen.

    Columns: ["experiment", "rel_iter_from_llm", "metric", "value"].
    """

    if metric not in ANALYSIS_METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        rows = load_experiment_rows(p)
        df = compute_relative_iteration_from_first_llm(rows)
        if metric not in df.columns:
            continue
        if df["rel_iter_from_llm"].isna().all():
            continue
        exp_name = os.path.basename(os.path.dirname(p))
        frames.append(
            pd.DataFrame(
                {
                    "experiment": exp_name,
                    "rel_iter_from_llm": df["rel_iter_from_llm"].to_numpy(),
                    "metric": metric,
                    "value": df[metric].to_numpy(dtype=float),
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["experiment", "rel_iter_from_llm", "metric", "value"]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(
        ["metric", "experiment", "rel_iter_from_llm"], inplace=True
    )
    return merged


def compute_normalized_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add a z-normalized version of the metric per experiment.

    Returns a copy with extra column f"{metric}_z".
    """

    if metric not in df.columns:
        return df.copy()

    result = df.copy()
    col = metric
    zcol = f"{metric}_z"
    result[zcol] = result.groupby("experiment")[col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) or np.nan)
    )
    return result
