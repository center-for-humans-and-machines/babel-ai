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
    temperature: 0.7
    max_tokens: 150
    frequency_penalty: 0.0
    presence_penalty: 0.0
    top_p: 1.0
    forgetting: null
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
    novelty_nudge_rate: 0.20
    random_seed: 0
  - type: rag_scaffolder
    novelty_nudge_rate: 0.20
    num_words: 5
    search_results: 5
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
| `llm` | `provider`, `model` | `system_prompt`, `temperature`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `top_p`, `forgetting` |
| `rule_based` | None; `partner` defaults to `eliza` | `generic_intervention`, `topic_switch_probability`, `feed_sources` |
| `mirror` | None | None |
| `scaffolder` | None | thresholds, memory policy, `random_seed` |
| `rag_scaffolder` | an `llm` agent in the same list | scaffolder fields plus `word_model`, `word_model_path`, `num_words`, `search_backend`, `search_results`, `fetch_page`, `max_source_chars`, `search_timeout`, `context_turns`, `nudge_system_prompt`, `nudge_max_tokens` |

LLM generation parameters mirror the legacy `agent_configs` fields.
Defaults: `temperature` `1.0`, `max_tokens` unset, penalties `0.0`,
`top_p` `1.0`, `forgetting` unset (use full history).

`generic_intervention` accepts `passthrough`, `llm_nudge`, `live_feed`,
or `custom`. `live_feed` uses `topic_switch_probability` (default
`0.5`) and `feed_sources` (`topic_bank`, `hackernews`). `passthrough`,
`llm_nudge`, and `live_feed` are fully executable; `custom` remains
reserved.

The `scaffolder` uses three branches: protect informative continuation,
consume one remembered topic when stuck, then inject a local topic when
memory is empty. Its lexical thresholds, memory limits, cooldown, and
novelty-nudge rate are configurable. The default `0.20` rate schedules
one model-generated novelty prompt per five informative turns. See
`configs/scaffolder_gpt_first_test.yaml` for all fields.

The `rag_scaffolder` shares that three-branch policy but replaces both
the novelty nudge and topic injection with a retrieval-augmented
pipeline: draw random vectors in a `word_model` embedding space, take the
nearest words, join them into a search query, pick one result, fetch its
page text, and ask the experiment's `llm` agent for a short grounded
instruction. It falls back to the deterministic wording whenever search
or the LLM call fails. See `configs/rag_scaffolder_gpt_first_test.yaml`.
Each grounded turn records its words, query, and source in the
`rag_*` columns of `turns.parquet`.

`search_backend` selects the search source: `duckduckgo` (default) scrapes
DuckDuckGo HTML, `wikipedia` uses the MediaWiki API, and `auto` tries
DuckDuckGo first and falls back to Wikipedia. DuckDuckGo can serve an
anti-bot challenge on proxied or datacenter networks, in which case
`auto` or `wikipedia` keeps the pipeline working.

Word vectors are fetched lazily with `requests` into `~/gensim-data`
(and cached). Prefetch them, or point `word_model_path` at a local
word2vec/GloVe file:

```bash
poetry run python scripts/download_vectors.py glove-wiki-gigaword-100
```

Because five arbitrary embedding words rarely match a full-text query,
the provider tries the joined words, then the first two, then the first
word, and records whichever query succeeded in `rag_query`.

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

## Analyzer settings

| Key | Values or default | Purpose |
| --- | --- | --- |
| `analyzer` | `similarity` | Drift metric implementation. |
| `analyze_window` | minimum `1` | Rolling similarity window size. |
| `analysis_scope` | `llm_only` | Include only LLM turns in similarity and t-SNE analysis. Set `all_turns` to include partner and seed turns. |

## Compatibility note

The Pydantic models for the target agent list exist in
`conversation/agent_config.py`, but `Experiment` currently accepts only
the legacy fields. Existing runnable YAML must therefore retain
`fetcher_config`, `analyzer_config`, `agent_configs`, and
`agent_selection_method` until the factory and Experiment migration
land.
