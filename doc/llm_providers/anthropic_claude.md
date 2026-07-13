# Anthropic Claude — provider reference for babel-ai

Condensed setup and API notes for `src/api/anthropic.py`.
Official docs: [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)

---

## Authentication

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Header (handled by SDK):

```http
x-api-key: YOUR_KEY
anthropic-version: 2023-06-01
```

---

## Request shape

Anthropic uses the **Messages API**, not OpenAI chat completions:

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,          # required — no default in API
    messages=[
        {"role": "user", "content": "Hello"},
    ],
    temperature=0.7,
    top_p=1.0,
)

text = response.content[0].text
input_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens
```

---

## Differences from OpenAI provider

| Feature | OpenAI | Anthropic |
| --- | --- | --- |
| System prompt | `role: system` message | `system=` parameter (preferred) |
| `max_tokens` | optional | **required** |
| `frequency_penalty` | supported | **not supported** |
| `presence_penalty` | supported | **not supported** |
| Async | `AsyncOpenAI` | `AsyncAnthropic` (consider for refactor) |

babel-ai currently passes penalties anyway and logs warnings — keep
compatible signature on unified `LLMInterface` but document ignored
params.

---

## System messages

Recommended refactor: extract `system` messages from the list before
calling API:

```python
system_parts = [m["content"] for m in messages if m["role"] == "system"]
user_messages = [m for m in messages if m["role"] != "system"]

response = client.messages.create(
    model=model.value,
    system="\n".join(system_parts) if system_parts else None,
    messages=user_messages,
    max_tokens=max_tokens or 2048,
    ...
)
```

Current code passes full message list — works if no system role or
Anthropic accepts conversion; verify for multi-turn drift experiments.

---

## Current babel-ai enum (`AnthropicModels`)

| Enum member | Model id |
| --- | --- |
| `CLAUDE_OPUS_4_20250514` | `claude-opus-4-20250514` |
| `CLAUDE_SONNET_4_20250514` | `claude-sonnet-4-20250514` |
| `CLAUDE_3_5_HAIKU_20241022` | `claude-3-5-haiku-20241022` |

**Action:** refresh against [Anthropic model docs](https://docs.anthropic.com/en/docs/about-claude/models)
when implementing pillar A.

---

## Budget tracking bug

`src/api/budget.py` — Haiku input price may be wrong (`0.04` vs
comment `$0.0008`). Fix during pillar A audit.

---

## Token limits

- Always set `max_tokens` explicitly (babel-ai defaults to 2048 with
  warning if missing).
- Context window depends on model — check docs for Claude 4 vs 3.5.

---

## Error handling

- `anthropic.RateLimitError` — retry (already in `LLMInterface`)
- `anthropic.BadRequestError` — often max_tokens or message format

---

## Implementation checklist (pillar A)

- [ ] Refresh model enum ids
- [ ] Fix Haiku pricing in budget tracker
- [ ] Split system vs conversation messages
- [ ] Optional: async `anthropic_request` for experiment throughput
- [ ] Integration test behind `ANTHROPIC_API_KEY` env gate
