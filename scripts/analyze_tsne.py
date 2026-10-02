"""Generate t-SNE trajectory artifacts for completed runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trajectory.artifacts import (
    analyze_results_root,
    analyze_run_safely,
)  # noqa: E402


def main() -> None:
    """Analyze one run directory or every run below a results root."""
    parser = argparse.ArgumentParser(
        description="Generate LLM t-SNE trajectory artifacts",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="results",
        help="Run directory or results root",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Deterministic t-SNE seed",
    )
    args = parser.parse_args()
    path = Path(args.path)
    if (path / "turns.parquet").is_file():
        result = analyze_run_safely(
            path,
            random_state=args.random_state,
        )
        status = result.meta["status"] if result else "failed"
        print(f"{path.name}: {status}")
        return

    statuses = analyze_results_root(
        path,
        random_state=args.random_state,
    )
    if not statuses:
        print(f"No completed runs found under {path}")
        return
    for run_id, status in statuses.items():
        print(f"{run_id}: {status}")


if __name__ == "__main__":
    main()
