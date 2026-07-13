# Run artifacts

The canonical output contract is one directory per run:

```text
results/{run_id}/
├── checkpoint.json
├── turns.parquet
├── meta.json
└── manifest.json
```

`checkpoint.json` supports recovery while a run is active.
`turns.parquet` is the primary completed-run artifact.
`meta.json` records run metadata and resolved configuration.
`manifest.json` is optional and records the artifact schema version.

Run directory names are config-driven slugs from
`persistence.run_naming` (agents, fetcher, turn limit, short uuid).

## Checkpoint schema

The current checkpoint writer emits:

| Field | Meaning |
| --- | --- |
| `run_id`, `status`, `saved_at` | Run identity and lifecycle state. |
| `messages` | Serialized context stack entries. |
| `metrics` | Metrics accumulated so far. |
| `turn_taking`, `turn_taking_state` | Algorithm snapshot and cursors. |
| `pending_llm_nudge` | Visible-partner nudge pending for scheduling. |
| `agent_states` | Per-agent state (ELIZA memory stack and counters). |
| `agent_turn_count`, `settings` | Loop state and resolved settings. |

Checkpoint writes are atomic within the filesystem: a temporary JSON
file is renamed into place.

## `turns.parquet` schema

Each row represents one transcript turn. The flat design makes it easy
to load into Pandas, notebooks, or the visualization without parsing a
nested analysis blob.

| Column | Type or shape | Meaning |
| --- | --- | --- |
| `run_id` | string | Run directory identifier. |
| `turn_index` | integer | Ordered transcript index. |
| `timestamp` | timestamp | Turn creation time. |
| `role` | string | Chat role used by the context. |
| `speaker` | string | Human-readable agent or seed name. |
| `content` | string | Turn text. |
| `agent_id` | nullable string | Producing agent identifier. |
| `eliza_branch` | nullable string | ELIZA path label (`keyword:…`, `memory:pop`, `generic:$`, `generic:topic_switch`). |
| `eliza_keyword` | nullable string | Matched keyword when applicable. |
| `eliza_reassembly` | nullable string | Reassembly rule or redirect label. |
| `word_count` | integer | Flat lexical metric. |
| `semantic_similarity` | nullable float | Similarity signal. |
| `semantic_similarity_window` | nullable float | Windowed signal. |
| `lexical_similarity` | nullable float | Lexical signal. |
| `lexical_similarity_window` | nullable float | Windowed signal. |
| `coherence_score` | nullable float | Unique-to-total word signal. |
| `token_perplexity` | nullable float | Token-distribution signal. |
| `used_generic_fallback` | boolean | Whether ELIZA used `$`. |
| `analysis_extra_*` | nullable | Future flat analysis fields. |

Column names should be derived from the Pydantic analysis schema, not a
second handwritten registry.

## Current implementation state

`RunManifest`, `RunRecord`, `save_run()`, `load_run()`, and
`list_runs()` are implemented in `persistence/run_store.py`.
`Experiment` writes canonical parquet and metadata for managed
conversations; legacy root-level CSV remains for older experiment paths
during migration.
