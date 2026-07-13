# ELIZA partner

`PartnerSession` is the stateful adapter around vendored
[rdimaio/eliza-py](https://github.com/rdimaio/eliza-py). It provides
a deterministic, non-LLM participant for managed conversations.

## `PartnerSession`

`PartnerSession.respond(messages)`:

1. Load `general.json` and a per-session deep copy of `doctor.json`.
2. Preserve the rdimaio memory stack across turns.
3. Select the latest non-partner turn as the ELIZA input.
4. Invoke the generic intervention only on the `$` fallback path.
5. Apply topic and self-reference redirects when configured markers
   fire.
6. Remove `Eliza: ` and interactive `You: ` framing.
7. Return `PartnerTurn` with branch metadata.

Copying the doctor script isolates rdimaio's cyclic reassembly counters
between sessions. Prefix removal occurs in the wrapper so vendor logic
can remain otherwise unchanged.

`PartnerTurn` fields:

| Field | Meaning |
| --- | --- |
| `text` | Plain partner reply. |
| `used_generic_fallback` | True when the `$` path fired. |
| `eliza_branch` | Compact rule-path label (see below). |
| `eliza_keyword` | Matched keyword, if any. |
| `eliza_reassembly` | Reassembly rule id or redirect label. |

## Branch tracing

Each reply records how rdimaio resolved the turn:

| Label | Path |
| --- | --- |
| `keyword:…` | Keyword match (+ `+memory` when stack pushed). |
| `memory:pop` | Remembered reply from the memory stack. |
| `generic:$` | Stock rdimaio `$` text (passthrough or nudge). |
| `generic:topic_switch` | Live-feed topic switch on `$`. |

Topic and self-reference redirects replace `LAST_TOPIC_REDIRECT` or
first-person peer turns with `We are talking about {topic} - not me.`
and label the reassembly as `self_reference_redirect`.

## Interventions

An intervention is an explicit extension point for a failed keyword
decomposition with an empty memory stack:

```python
class GenericIntervention(Protocol):
    def on_generic(self, ctx: GenericContext) -> GenericResult: ...
```

`GenericContext` contains the incoming text, stripped rdimaio default,
message history, and turn index. A result with `partner_text=None`
retains the upstream default. It never creates a hidden system prompt:
its output is appended to the transcript as an ordinary partner turn.

| Mode | `$` behavior | Status |
| --- | --- | --- |
| `passthrough` | Keep stock rdimaio generic text. | Available |
| `llm_nudge` | Return a rotated steering phrase. | Available |
| `live_feed` | Maybe return a feed topic switch on `$`. | Available |
| `custom` | Delegate to a plugin. | Reserved |

Unknown configuration values should resolve to `passthrough` in the
target architecture. The current factory validates enum values, so
callers must handle unknown strings before calling it.

## Live feed

`LiveFeedProvider.pick_topic()` abstracts topic sources. Configure
sources on `RuleBasedAgentConfig.feed_sources` (default
`["topic_bank"]`).

| Source | Behavior |
| --- | --- |
| `topic_bank` | Random topic from bundled `topic_bank.json` (offline). |
| `hackernews` | Random title from HN top stories (network optional). |

`build_feed()` composes multiple sources via `CombinedFeed`: it shuffles
configured feeds and returns the first non-empty topic. Unknown source
names log a warning and are skipped.

With `generic_intervention: live_feed`, a random draw against
`topic_switch_probability` (default `0.5`) decides whether a generic
`$` turn becomes a topic switch:

`I came across {topic}. What comes to mind?`

When the draw fails or no topic is available, stock rdimaio `$` text is
kept. Branch metadata records switches as `generic:topic_switch`
instead of `generic:$`.

## Scaffolding ladder

| Level | Participant or intervention | Effect |
| --- | --- | --- |
| 1 | `MirrorConversationAgent` | Echo the latest peer message. |
| 2 | ELIZA + `passthrough` | Stock rdimaio with clean output. |
| 3 | ELIZA + `llm_nudge` | `$` emits a visible steering phrase. |
| 4 | ELIZA + `live_feed` | `$` introduces external topic input. |
| 4b | ELIZA + `custom` | Plugin-controlled starter or template. |

Levels 1 and 2 are distinct: mirror is a minimal conversation agent,
whereas ELIZA uses keyword ranking, decomposition, reassembly, and a
memory stack.
