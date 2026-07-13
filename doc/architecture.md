# ELIZA scaffolding architecture

babel-ai studies drift and collapse in long-running conversations. A
`ConversationManager` owns the transcript and schedules agents. ELIZA
is a deterministic participant, not an LLM prompt wrapper.

## System overview

```mermaid
flowchart TB
  Seed[Seed messages] --> Manager[ConversationManager]
  Config[YAML configuration] --> Manager
  Config --> Factory[Agent factory]
  Factory --> Manager
  Manager --> Context[ContextStack]
  Manager --> TurnTaking[Turn-taking algorithm]
  TurnTaking --> Agents[Conversation agents]
  Agents --> LLM[LLMConversationAgent]
  Agents --> Rule[RuleBasedConversationAgent]
  Agents --> Mirror[MirrorConversationAgent]
  Rule --> Eliza[PartnerSession]
  Eliza --> Vendor[Vendored rdimaio ELIZA]
  Manager --> Checkpoint[checkpoint.json]
  Manager --> Analyzer[Analyzer]
  Analyzer --> Store[Run store]
  Store --> Run["results/{run_id}/"]
  Run --> Viz[Read-only visualization]
```

All generated turns enter `ContextStack`. This gives each participant
the same ordered history and makes ELIZA interventions visible as normal
conversation turns.

## ELIZA response path

```mermaid
flowchart LR
  Input[Latest non-partner turn] --> Rank[Rank keywords]
  Rank --> Decompose[Decompose input]
  Decompose -->|match| Reassemble[Reassemble reply]
  Decompose -->|no match| Memory{Memory stack?}
  Memory -->|yes| Pop[Pop remembered reply]
  Memory -->|no| Generic["$ generic response"]
  Generic --> Hook[GenericIntervention]
  Reassemble --> Clean[Strip ELIZA and You prefixes]
  Pop --> Clean
  Hook --> Clean
  Clean --> Partner[PartnerTurn]
```

The vendor logic remains stock rdimaio except for the `$` hook. The
wrapper does not call rdimaio's interactive `prepare_response()`.

## Pillars

| Pillar | Scope | Architectural role |
| --- | --- | --- |
| A | LLM access | Provider adapters used by `LLMConversationAgent`. |
| B | Metrics | Flat metric schema and canonical parquet artifacts. |
| C | Web visualization | Read-only exploration of saved runs. |
| D | Repository hygiene | Removes legacy loops and stale documentation. |
| E | Conversation manager | Owns lifecycle, context, scheduling, recovery. |

The ELIZA work is coupled to E3: `RuleBasedConversationAgent` adapts a
`PartnerSession`. Pillar B persists its fallback marker. Pillar C reads
the resulting run directories without controlling experiments.

## Delivery status

E1 supplies the manager, context stack, round-robin scheduling,
analysis policies, and atomic checkpoints. E2–E4, ELIZA execution,
agent factories, run-store I/O, and canonical YAML loading remain
scaffolding or planned work.
