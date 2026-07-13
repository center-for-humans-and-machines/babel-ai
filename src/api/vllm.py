"""vLLM OpenAI-compatible cluster endpoint (pillar A)."""

import os
from typing import Dict, List, Optional

from api.enums import APIModels


def vllm_request(
    messages: List[Dict[str, str]],
    model: APIModels,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Call a vLLM server via OpenAI-compatible client."""
    _ = os.environ.get("VLLM_BASE_URL")
    raise NotImplementedError("vLLM provider not implemented")
