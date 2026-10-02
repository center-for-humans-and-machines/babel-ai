# Web trajectory viewer

The local viewer reads completed run artifacts from `results/{run_id}/`.

Start it from the repository root:

```bash
poetry run python -m viz
```

Open `http://127.0.0.1:8765/runs` to select a run. Each run page shows
grouped analysis trajectories: semantic and lexical similarity (windowed
and direct on one chart, fixed 0–1 y-axis), token perplexity, plus the
transcript. If no analysis metric is available, the page plots turn index
instead.

## LLM topic cloud

Completed runs also show a per-run t-SNE map of generated LLM turns.
Markers form the topic cloud; the connecting line and color gradient
show turn order. Start and end points are labeled. Axes show t-SNE
coordinates on an equal-scale grid.

New experiments generate this artifact after saving. Backfill existing
runs without calling an LLM:

```bash
poetry run python scripts/analyze_tsne.py results
```

Pass `results/{run_id}` to analyze one run. The viewer only reads saved
coordinates and never computes embeddings.

## ELIZA run detail

When a run includes a rule-based ELIZA partner, the run-detail page
shows:

- Resolved partner config: `generic_intervention`,
  `topic_switch_probability`, `feed_sources`
- Per-turn branch labels: `keyword:…`, `memory:pop`, `generic:$`,
  `generic:topic_switch`
- Keyword, reassembly, and `used_generic_fallback` columns
- Branch distribution chart and summary counts

Older runs without `eliza_branch` in `turns.parquet` display a warning;
re-run with a current build to capture branch metadata.

## RAG scaffolder detail

When a run includes a `rag_scaffolder` agent, the run-detail page shows:

- Resolved pipeline config: `word_model`, `word_model_path`, `num_words`,
  `search_backend`, `search_results`, `novelty_nudge_rate`
- A per-turn table with mode, status (grounded / fallback / not used),
  sampled words, search query, chosen source link, fallback reason, and
  the emitted nudge
- A grounded-vs-fallback usage chart over turn index

A `fallback` status means the retrieval pipeline could not produce a
grounded nudge this turn (offline, no vectors, no search results, or an
LLM error) and the deterministic scaffolder wording was used instead.

To populate vectors without `gensim.downloader`, pre-cache them:

```bash
poetry run python scripts/download_vectors.py glove-wiki-gigaword-100
```

`search_backend: auto` tries DuckDuckGo then Wikipedia; set
`search_backend: wikipedia` if DuckDuckGo serves an anti-bot challenge.

## Compare view (C2–C4)

Open `http://127.0.0.1:8765/compare` to overlay multiple runs, plot mean
± 95% CI across selected runs, and diff flattened config keys.

Query parameters:

- `runs` — comma-separated run ids
- `baseline` — run id drawn with a bold solid line
- `metric` — parquet column (default `semantic_similarity_window`;
  also `semantic_similarity`, `lexical_similarity_window`,
  `lexical_similarity`, `token_perplexity`)

The viewer is read-only. It uses `persistence.list_runs()` and
`persistence.load_run()`; create artifacts with the canonical
`persistence.save_run()` API.

## Terminal progress

Long runs driven through `Experiment` render a live turn progress bar in
the terminal (`conversation.progress.TurnProgress`), showing turn count,
speaker, and a preview of the latest reply.
