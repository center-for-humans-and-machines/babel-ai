# Configuration

The target configuration replaces legacy `agent_configs` and
`agent_selection_method` with an ordered `agents` list and
`conversation_settings`. The `type` discriminator selects a
configuration model.

```yaml
agents:
  - type: llm
    provider: ollama
    model: mistral:7b-instruct
    system_prompt: "You are having a conversation."
  - type: rule_based
    partner: eliza
    generic_intervention: live_feed
    topic_switch_probability: 0.5
    feed_sources:
      - topic_bank
      - hackernews
  - type: mirror
  - type: scaffolder
    stuck_turns: 2
    random_seed: 0

conversation_settings:
  turn_taking_method: round_robin
  analysis_policy: on_checkpoint
  checkpoint_enabled: true
  checkpoint_interval_seconds: 120
  max_iterations: 50
  max_total_characters: 1000000
```

## Agent entries

| Type | Required keys | Optional keys |
| --- | --- | --- |
| `llm` | `provider`, `model` | `system_prompt` |
| `rule_based` | None; `partner` defaults to `eliza` | `generic_intervention`, `topic_switch_probability`, `feed_sources` |
| `mirror` | None | None |
| `scaffolder` | None | thresholds, memory policy, `random_seed` |

`generic_intervention` accepts `passthrough`, `llm_nudge`, `live_feed`,
or `custom`. `live_feed` uses `topic_switch_probability` (default
`0.5`) and `feed_sources` (`topic_bank`, `hackernews`). `passthrough`,
`llm_nudge`, and `live_feed` are fully executable; `custom` remains
reserved.

The `scaffolder` uses three branches: protect informative continuation,
consume one remembered topic when stuck, then inject a local topic when
memory is empty. Its lexical thresholds, memory limits, cooldown, and
random seed are configurable. See
`configs/scaffolder_gpt_first_test.yaml` for all fields.

## Conversation settings

| Key | Values or default | Purpose |
| --- | --- | --- |
| `turn_taking_method` | `round_robin` | Select speaker scheduling. |
| `fixed_order` | list of agent indices | Required for `fixed_order`. |
| `analysis_policy` | `on_checkpoint` | `per_turn`, checkpoint, or end. |
| `checkpoint_enabled` | `true` | Enable periodic recovery files. |
| `checkpoint_interval_seconds` | `120`, minimum `1` | Checkpoint cadence. |
| `max_iterations` | `100`, minimum `1` | Stack-message stop limit. |
| `max_total_characters` | `1000000`, minimum `1` | Context-size stop limit. |

## Compatibility note

The Pydantic models for the target agent list exist in
`conversation/agent_config.py`, but `Experiment` currently accepts only
the legacy fields. Existing runnable YAML must therefore retain
`fetcher_config`, `analyzer_config`, `agent_configs`, and
`agent_selection_method` until the factory and Experiment migration
land.
