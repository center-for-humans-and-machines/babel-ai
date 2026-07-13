# Azure OpenAI — provider reference for babel-ai

Condensed setup and API notes for `src/api/azure_openai.py`.
Official docs: [Azure OpenAI REST API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference)

---

## Authentication

Two supported methods:

| Method | Header | Notes |
| --- | --- | --- |
| API key | `api-key: YOUR_KEY` | Simplest; matches current babel-ai |
| Microsoft Entra ID | `Authorization: Bearer TOKEN` | Enterprise / managed identity |

babel-ai today uses the **Azure OpenAI Python SDK** (`AzureOpenAI`) with
API key from environment.

---

## Environment variables

```bash
AZURE_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com
AZURE_KEY=your-api-key
```

Optional (recommended for refactor):

```bash
AZURE_API_VERSION=2024-12-01-preview   # pin explicitly
```

Current code hardcodes `api_version = "2024-12-01-preview"` in
`azure_openai.py`.

---

## Request shape

Chat completions use **deployment name** as `model`, not the public
OpenAI model id:

```http
POST https://{endpoint}/openai/deployments/{deployment-id}/chat/completions?api-version={version}
```

SDK equivalent (babel-ai pattern):

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ["AZURE_KEY"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
)

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",  # must match deployment name in Azure
    messages=[{"role": "user", "content": "..."}],
    temperature=0.7,
    max_tokens=150,
)
```

---

## Model families and parameters

### Legacy chat models (GPT-4o class)

- Parameters: `temperature`, `top_p`, `max_tokens`, `frequency_penalty`,
  `presence_penalty`.
- Response: `response.choices[0].message.content` (string).

### Reasoning models (o3, o4-mini)

Azure/o-series models use different request fields:

| Legacy | New |
| --- | --- |
| `max_tokens` | `max_completion_tokens` |
| `temperature` (variable) | often fixed to `1.0` only |
| `top_p` | not supported |

babel-ai handles this via `AzureModels.uses_new_parameters()` in
`src/api/enums.py` — **bug:** references undefined
`AzureModels.GPT5_MINI_2025_08_07`; remove or define.

Response parsing for some new models may return structured content
blocks — verify against deployed deployment before production use.

---

## Current babel-ai enum (`AzureModels`)

| Enum member | Deployment id string |
| --- | --- |
| `GPT4O_2024_08_06` | `gpt-4o-2024-08-06` |
| `O3_2025_04_16` | `o3` |
| `O4_MINI_2025_04_16` | `o4-mini` |

**Action:** audit deployed names in your Azure resource; enum values
must match deployment names exactly.

---

## Token usage

```python
input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens
```

Map into babel-ai `LLMResponse` for budget tracking.

---

## Error handling

- 401 — bad key or endpoint
- 404 — deployment name mismatch
- 429 — rate limit; babel-ai `LLMInterface` already retries with
  exponential backoff

---

## Implementation checklist (pillar A)

- [ ] Pin `AZURE_API_VERSION` via env
- [ ] Fix `uses_new_parameters()` enum bug
- [ ] Document deployment ↔ enum mapping in `doc/llm_providers/azure.md`
- [ ] Add integration test behind `AZURE_KEY` env gate
- [ ] Verify o3/o4-mini response content parsing
