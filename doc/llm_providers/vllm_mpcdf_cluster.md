# vLLM cluster access — provider reference for babel-ai

Pattern from [machine-cultural-evolution](https://github.com/)
(`feat/vllm_backend` branch) — adapt for babel-ai `Provider.VLLM`.

Repo path: `/Users/mienhardt/Programming/machine-cultural-evolution`

---

## Two deployment modes

| Mode | Where | Use case |
| --- | --- | --- |
| **Co-located SLURM** | vLLM on `localhost:8000` inside GPU job | Batch HPC (MCE worker) |
| **Remote OpenAI-compat** | Hosted or SSH-tunneled `/v1` URL | babel-ai laptop / login node |

babel-ai needs the **remote / OpenAI-compat client** pattern first;
SLURM orchestration stays in MCE unless we port job queue later.

---

## OpenAI-compatible API

vLLM serves standard chat completions:

```http
GET  http://host:8000/health
POST http://host:8000/v1/chat/completions
```

Auth: often none locally — use dummy key so OpenAI SDK initializes:

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["VLLM_BASE_URL"],  # e.g. http://127.0.0.1:8000/v1
    api_key=os.environ.get("VLLM_API_KEY", "local-no-auth"),
)

response = client.chat.completions.create(
    model=os.environ["VLLM_MODEL"],  # HF id or served name
    messages=[{"role": "user", "content": "..."}],
    max_tokens=512,
    temperature=0.7,
)
```

---

## MCE reference implementation

File: `src/llm_interface/apis/mpcdf_vllm.py` (branch `feat/vllm_backend`)

Key patterns to port:

```python
CLIENT = AsyncOpenAI(
    base_url=os.getenv("MPCDF_VLLM_ENDPOINT_URL"),
    api_key=os.getenv("MPCDF_VLLM_API_KEY") or "local-no-auth",
)
```

Model-specific `extra_body` (example — Llama 3 stop tokens):

```python
extra_body = {}
if model.value.startswith("meta-llama/Meta-Llama-3"):
    extra_body["stop_token_ids"] = [128001, 128009]

response = await CLIENT.chat.completions.create(
    **request_params,
    extra_body=extra_body or None,
)
```

Structured JSON output via `response_format` + Pydantic schema — only
if babel-ai experiments need it later.

---

## Environment variables

### babel-ai (proposed)

```bash
# Remote or tunneled vLLM
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=local-no-auth
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

### MCE cluster worker (co-located)

```bash
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
TENSOR_PARALLEL_SIZE=2
VLLM_PORT=8000
HF_CACHE=/ptmp/$USER/hf_cache
HF_TOKEN=hf_...
MONGO_URI=mongodb://...
```

Worker polls `http://localhost:8000/health` before claiming jobs.

---

## SLURM + Singularity (MCE overview)

On compute node, GPU container runs:

```bash
singularity exec "$VLLM_SIF" vllm serve "$VLLM_MODEL" \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
```

CPU sim container runs Python worker; calls `localhost:8000/v1`.

**For babel-ai:** submit experiments from login node with
`VLLM_BASE_URL` pointing at:

- SSH tunnel to worker node, or
- MPCDF-hosted inference URL if available

---

## babel-ai integration plan

1. Add `Provider.VLLM` to `src/api/enums.py`.
2. Add `VLLMModels` enum (HF model ids served by your cluster).
3. New module `src/api/vllm.py`:
   - sync `vllm_request()` matching existing provider signature
   - returns `LLMResponse` with token counts from `response.usage`
4. Wire into `Provider.get_request_function()`.
5. YAML config:

```yaml
provider: vllm
model: Qwen/Qwen3-30B-A3B-Instruct-2507
```

6. Document tunnel workflow in `doc/llm_providers/vllm.md`:

```bash
# Example: tunnel from laptop to compute node
ssh -L 8000:localhost:8000 user@raven
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

---

## Differences vs Ollama provider

| | Ollama | vLLM |
| --- | --- | --- |
| API | Ollama-native or compat | OpenAI `/v1/chat/completions` |
| Token counts | estimated in babel-ai | from `usage` object |
| Cluster scale | single host | multi-GPU tensor parallel |
| Model ids | tags (`mistral:7b`) | HF ids |

Keep both providers — Ollama for local dev, vLLM for cluster.

---

## Known pitfalls (from MCE)

- Qwen3 vs Llama 3 stop-token handling differs
- Guided JSON decoding may need model-specific fixes
- Viper compute nodes may lack outbound internet — prefetch weights
  on login node
- Compare vLLM vs Ollama verbosity parity (`feat/vllm-ollama-parity`)

---

## Implementation checklist (pillar A)

- [ ] Add `Provider.VLLM` + `vllm.py`
- [ ] Env vars documented in `.env.example`
- [ ] Port MCE dummy-api-key pattern
- [ ] Optional async for batch experiments
- [ ] Health-check helper before long experiment runs
- [ ] Integration test with mock HTTP or local vLLM
