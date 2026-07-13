# Web trajectory viewer

The local viewer reads completed run artifacts from `results/{run_id}/`.

Start it from the repository root:

```bash
poetry run python -m viz
```

Open `http://127.0.0.1:8765/runs` to select a run. Each run page shows
the windowed semantic-similarity trajectory and transcript. If the
metric is unavailable, the chart plots turn index instead.

The viewer is read-only. It uses `persistence.list_runs()` and
`persistence.load_run()`; create artifacts with the canonical
`persistence.save_run()` API.
