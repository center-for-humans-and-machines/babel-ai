# ELIZA partner

`PartnerSession` is the stateful adapter around vendored
[rdimaio/eliza-py](https://github.com/rdimaio/eliza-py). It is intended
to provide a deterministic, non-LLM participant for managed
conversations.

## `PartnerSession`

`PartnerSession.respond(messages)` will:

1. Load `general.json` and a per-session deep copy of `doctor.json`.
2. Preserve the rdimaio memory stack across turns.
3. Select the latest non-partner turn as the ELIZA input.
4. Invoke the generic intervention only on the `$` fallback path.
5. Remove `Eliza: ` and interactive `You: ` framing.
6. Return `PartnerTurn(text, used_generic_fallback)`.

Copying the doctor script isolates rdimaio's cyclic reassembly counters
between sessions. Prefix removal occurs in the wrapper so vendor logic
can remain otherwise unchanged.

`PartnerSession.respond()` is currently a scaffold and raises
`NotImplementedError`; the behavior above is the integration contract.

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
| `llm_nudge` | Return a rotated steering phrase. | Scaffold |
| `live_feed` | Return a headline-based prompt. | Scaffold |
| `custom` | Delegate to a plugin. | Reserved |

Unknown configuration values should resolve to `passthrough` in the
target architecture. The current factory validates enum values, so
callers must handle unknown strings before calling it.

## Live feed vendor boundary

`LiveFeedProvider.fetch_headline()` abstracts a future RSS, JSON, or AT
Protocol source. The present `LiveFeedStub` raises
`NotImplementedError`; no network access occurs in phase 1. A future
live reply is formatted as: `I saw that {headline}. What comes to mind?`

## Scaffolding ladder

| Level | Participant or intervention | Effect |
| --- | --- | --- |
| 1 | `MirrorConversationAgent` | Echo the latest peer message. |
| 2 | ELIZA + `passthrough` | Stock rdimaio with clean output. |
| 3 | ELIZA + `llm_nudge` | `$` emits a visible steering phrase. |
| 4 | ELIZA + `live_feed` | `$` introduces current external input. |
| 4b | ELIZA + `custom` | Plugin-controlled starter or template. |

Levels 1 and 2 are distinct: mirror is a minimal conversation agent,
whereas ELIZA uses keyword ranking, decomposition, reassembly, and a
memory stack.
