# Metrics and run persistence (pillar B)

Collapse metrics are stored as flat columns in `results/{run_id}/turns.parquet`.

## Primary signals

| Column | Meaning |
| --- | --- |
| `semantic_similarity_window` | Mean embedding similarity vs recent turns |
| `lexical_similarity_window` | Mean Jaccard similarity vs recent turns |
| `coherence_score` | Unique words / total words |
| `token_perplexity` | Language-model perplexity |
| `used_generic_fallback` | ELIZA `$` fallback fired |

## API

```python
from persistence import save_run, load_run, list_runs
from analysis_schema import AnalysisResult, flatten_analysis
from collapse import detect_similarity_spikes, mark_generic_fallback_turns
```

## Legacy CSV

Root-level `drift_experiment_*.csv` is no longer written. Use
`persistence.legacy_adapter.load_legacy_csv()` for old runs during
migration.
