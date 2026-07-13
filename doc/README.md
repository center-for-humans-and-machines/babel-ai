# babel-ai architecture documentation

This documentation describes the ELIZA conversation architecture. The
conversation manager, ELIZA partner (including `live_feed`), unified
agent factory, run artifacts, and web viewer are implemented.

## Guides

- [Architecture](architecture.md): system boundaries and pillars A–E.
- [ELIZA partner](eliza_partner.md): deterministic partner and ladder.
- [Conversation manager](conversation_manager.md): agents and recovery.
- [Run artifacts](run_artifacts.md): canonical `results/{run_id}/` data.
- [Configuration](configuration.md): canonical YAML agent configuration.
- [Web trajectory viewer](viz.md): inspect completed run trajectories.
- [Metrics and persistence](metrics.md): parquet schema and analysis API.
- [Quick start](quickstart.md): installation and current runnable checks.
- [LLM providers](llm_providers/): provider setup and API references.

## Source map

| Area | Source |
| --- | --- |
| Managed conversation | `src/conversation/` |
| ELIZA partner | `src/eliza/` |
| Run artifact API | `src/persistence/run_store.py` |
| Metrics and analysis | `src/analyzer.py` |
| Visualization | `src/viz/` |

Run the local smoke conversation with:

```bash
poetry run python scripts/brief_conversation.py
```

It uses flat imports such as `conversation.manager` and writes run
artifacts under `results/`.
