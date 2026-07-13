"""FastAPI app for local trajectory analysis (pillar C)."""

from pathlib import Path

import pandas as pd
import plotly.express as px
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from persistence import list_runs, load_run

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def create_app(results_root: Path | None = None) -> FastAPI:
    """Build a read-only visualization app over ``results/{run_id}/``."""
    app = FastAPI(title="babel-ai trajectory viewer")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    app.state.results_root = Path(results_root or _DEFAULT_RESULTS_ROOT)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the local viewer is available."""
        return {"status": "ok"}

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        """Render the completed-run list."""
        records = list_runs(app.state.results_root)
        return templates.TemplateResponse(
            request=request,
            name="runs.html",
            context={"runs": records},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        """Render one run's trajectory and transcript."""
        record = _load_record(app.state.results_root, run_id)
        metric, title = _trajectory_metric(record.turns)
        chart = _trajectory_chart(record.turns, metric, title)
        transcript = _transcript_rows(record.turns)
        return templates.TemplateResponse(
            request=request,
            name="run_detail.html",
            context={
                "run": record,
                "chart": chart,
                "metric_title": title,
                "transcript": transcript,
            },
        )

    return app


def _load_record(results_root: Path, run_id: str):
    """Load a run after ensuring its directory remains below the root."""
    root = results_root.resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return load_run(run_dir)
    except (FileNotFoundError, OSError) as error:
        raise HTTPException(status_code=404, detail="Run not found") from error


def _trajectory_metric(turns: pd.DataFrame) -> tuple[str, str]:
    """Select semantic similarity when usable, otherwise turn index."""
    metric = "semantic_similarity_window"
    if metric in turns and turns[metric].notna().any():
        return metric, "Semantic similarity (window)"
    return "turn_index", "Turn index"


def _trajectory_chart(turns: pd.DataFrame, metric: str, title: str) -> str:
    """Return an embeddable Plotly trajectory chart."""
    frame = turns.sort_values("turn_index")
    figure = px.line(
        frame,
        x="turn_index",
        y=metric,
        markers=True,
        title=title,
        labels={"turn_index": "Turn", metric: title},
    )
    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def _transcript_rows(turns: pd.DataFrame) -> list[dict[str, object]]:
    """Extract display-safe transcript fields in turn order."""
    fields = ("turn_index", "role", "speaker", "content")
    frame = turns.sort_values("turn_index")
    return frame.reindex(columns=fields).fillna("").to_dict(orient="records")
