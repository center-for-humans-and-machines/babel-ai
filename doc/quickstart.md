# Quick start

## Install

Run these commands from the repository root:

```bash
poetry install
poetry run pytest tests/unit_tests/conversation/
```

Python 3.13 and Poetry are required. Configure provider credentials in
`.env` only when running LLM-backed experiments.

## Current runnable conversation

Run the local ELIZA and mirror smoke conversation:

```bash
poetry run python scripts/brief_conversation.py
```

It requires no provider credentials and writes a run under `results/`.

## Trajectory viewer

After a run exists under `results/`, launch the local viewer:

```bash
poetry run python -m viz
```

Open `http://127.0.0.1:8765/runs` to browse trajectories and transcripts.

Resume an interrupted run:

```bash
poetry run python scripts/resume_conversation.py results/{run_id}/checkpoint.json
```

## ELIZA conversation configuration

The script uses the canonical agent configuration:

```yaml
agents:
  - type: llm
    provider: ollama
    model: mistral:7b-instruct
  - type: rule_based
    partner: eliza
    generic_intervention: passthrough
conversation_settings:
  turn_taking_method: round_robin
  max_iterations: 6
  analysis_policy: at_end
```

Its expected order is seed → ELIZA → mirror → ELIZA → mirror. ELIZA
replies are deterministic and plain text, with no `Eliza:` or `You:`
framing. See [Configuration](configuration.md) for the full schema.

## Inspect results

The script prints the transcript and the path to its run artifacts.
Modules use flat imports, for example `conversation.manager` and
`persistence.run_store`.
