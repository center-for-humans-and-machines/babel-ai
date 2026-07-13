"""FastAPI app for local trajectory analysis (pillar C)."""

from pathlib import Path
from typing import Any


def create_app(results_root: Path | None = None) -> Any:
    """Build read-only viz app over ``results/{run_id}/``.

    Returns a FastAPI instance once pillar C dependencies land.
    """
    raise NotImplementedError(
        "viz app requires fastapi and uvicorn (pillar C)"
    )
