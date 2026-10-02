"""FastAPI app for local trajectory analysis (pillar C)."""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from persistence import list_runs, load_run
from persistence.run_naming import timestamp_from_meta
from trajectory.artifacts import load_projection
from viz.charts import (
    aggregate_chart,
    available_metrics,
    config_diff_rows,
    eliza_branch_bar_chart,
    eliza_branch_chart,
    eliza_branch_summary,
    metric_label,
    overlay_chart,
    rag_usage_chart,
    trajectory_charts_for_run,
)
from viz.eliza_display import (
    eliza_branch_status,
    eliza_turn_rows,
    extract_eliza_agent_config,
    has_eliza_branch_data,
)
from viz.rag_display import (
    extract_rag_agent_config,
    has_rag_data,
    rag_status,
    rag_turn_rows,
)
from viz.tsne_display import tsne_status, tsne_trajectory_chart

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
        for record in records:
            record.eliza_status = eliza_branch_status(record.turns)
            record.timestamp_human = timestamp_from_meta(record.meta)
        return templates.TemplateResponse(
            request=request,
            name="runs.html",
            context={"runs": records},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        """Render one run's trajectory and transcript."""
        record = _load_record(app.state.results_root, run_id)
        trajectory_charts = trajectory_charts_for_run(
            record.turns,
            record.meta,
        )
        eliza_chart = eliza_branch_chart(record.turns)
        eliza_counts = eliza_branch_bar_chart(record.turns)
        transcript = _transcript_rows(record.turns)
        tsne_result = load_projection(record.run_dir)
        run_slug = record.meta.get("run_slug", "")
        timestamp_human = timestamp_from_meta(record.meta)
        return templates.TemplateResponse(
            request=request,
            name="run_detail.html",
            context={
                "run": record,
                "run_slug": run_slug,
                "timestamp_human": timestamp_human,
                "eliza_config": extract_eliza_agent_config(record.meta),
                "eliza_status": eliza_branch_status(record.turns),
                "has_eliza_branches": has_eliza_branch_data(record.turns),
                "eliza_turns": eliza_turn_rows(record.turns),
                "trajectory_charts": trajectory_charts,
                "eliza_chart": eliza_chart,
                "eliza_counts": eliza_counts,
                "eliza_summary": eliza_branch_summary(record.turns),
                "transcript": transcript,
                "tsne_chart": tsne_trajectory_chart(tsne_result),
                "tsne_status": tsne_status(tsne_result),
                "has_rag_data": has_rag_data(record.turns),
                "rag_status": rag_status(record.turns),
                "rag_config": extract_rag_agent_config(record.meta),
                "rag_turns": rag_turn_rows(record.turns),
                "rag_chart": rag_usage_chart(record.turns),
            },
        )

    @app.get("/compare", response_class=HTMLResponse)
    def compare(
        request: Request,
        runs: str = "",
        baseline: str = "",
        metric: str = "semantic_similarity_window",
    ) -> HTMLResponse:
        """Overlay, aggregate, and config-diff views for selected runs."""
        all_runs = list_runs(app.state.results_root)
        selected_ids = [part for part in runs.split(",") if part]
        selected = [
            record for record in all_runs if record.run_id in selected_ids
        ]
        if selected and metric not in available_metrics(selected[0].turns):
            metric = available_metrics(selected[0].turns)[0]
        overlay = overlay_chart(selected, metric, baseline_id=baseline or None)
        aggregate = (
            aggregate_chart(selected, metric) if len(selected) > 1 else ""
        )
        diff_rows = config_diff_rows(selected) if len(selected) > 1 else []
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "runs": all_runs,
                "selected_ids": selected_ids,
                "baseline": baseline,
                "metric": metric,
                "metric_label": metric_label(metric),
                "overlay_chart": overlay,
                "aggregate_chart": aggregate,
                "diff_rows": diff_rows,
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


def _transcript_rows(turns: pd.DataFrame) -> list[dict[str, object]]:
    """Extract display-safe transcript fields in turn order."""
    fields = (
        "turn_index",
        "role",
        "speaker",
        "eliza_branch",
        "eliza_reassembly",
        "content",
    )
    frame = turns.sort_values("turn_index")
    present = [field for field in fields if field in frame.columns]
    rows = frame.reindex(columns=present).fillna("").to_dict(orient="records")
    for row in rows:
        if "eliza_branch" not in row:
            row["eliza_branch"] = ""
        if "eliza_reassembly" not in row:
            row["eliza_reassembly"] = ""
    return rows
