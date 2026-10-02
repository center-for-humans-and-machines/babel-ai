"""vLLM OpenAI-compatible cluster endpoint."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from api.enums import VLLMModels
from models.api import LLMResponse

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
base_url = os.getenv("VLLM_BASE_URL")
api_key = os.getenv("VLLM_API_KEY", "local-no-auth")

# Create the OpenAI-compatible vLLM client only when an endpoint is
# configured. Without this guard ``OpenAI(base_url=None)`` would fall
# back to the real OpenAI API, silently sending cluster requests to the
# wrong place.
CLIENT = OpenAI(base_url=base_url, api_key=api_key) if base_url else None

logger.debug("Initializing vLLM API client.")

# Llama 3's chat template ends turns with ``<|eot_id|>`` (and can emit
# ``<|end_of_text|>``), which vLLM does not treat as stop tokens by
# default. Without them generation runs to the length limit and
# truncates the reply, so they are supplied explicitly for these
# checkpoints.
LLAMA3_PREFIX = "meta-llama/Meta-Llama-3"
LLAMA3_STOP_TOKEN_IDS = [128001, 128009]


def _resolve_model(model: VLLMModels) -> str:
    """Resolve the model name sent to the vLLM server.

    The ``VLLMModels.DEFAULT`` sentinel delegates to the ``VLLM_MODEL``
    environment variable, so the served Hugging Face id can change
    without editing the enum or the experiment config. Any other enum
    member is used verbatim.

    Args:
        model: Requested vLLM model.

    Returns:
        The model name to send in the request.

    Raises:
        ValueError: ``model`` is ``VLLMModels.DEFAULT`` but
            ``VLLM_MODEL`` is unset or blank.
    """
    if model is VLLMModels.DEFAULT:
        env_model = os.getenv("VLLM_MODEL")
        if not env_model:
            raise ValueError(
                "VLLM_MODEL must be set when using VLLMModels.DEFAULT"
            )
        return env_model
    return model.value


def vllm_request(
    messages: list,
    model: VLLMModels = VLLMModels.DEFAULT,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """Send a request to the OpenAI-compatible vLLM endpoint.

    Args:
        messages: List of message dicts (role/content etc.)
        model: vLLM model to use. ``VLLMModels.DEFAULT`` resolves the
            served id from the ``VLLM_MODEL`` environment variable.
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p parameter
        max_tokens: Max tokens in response

    Returns:
        LLMResponse with content and token counts

    Raises:
        ValueError: No ``VLLM_BASE_URL`` is configured, or the default
            model was requested without a ``VLLM_MODEL``.
    """
    if CLIENT is None:
        raise ValueError("VLLM_BASE_URL must be set to reach a vLLM server")

    resolved_model = _resolve_model(model)
    request_params = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    # Llama 3 checkpoints need their end-of-turn token ids as explicit
    # stops; without them the reply runs on past the requested content.
    if resolved_model.startswith(LLAMA3_PREFIX):
        logger.debug("Adding Llama 3 stop token ids to vLLM request.")
        request_params["extra_body"] = {
            "stop_token_ids": LLAMA3_STOP_TOKEN_IDS,
        }

    logger.debug(
        "Sending request to vLLM API "
        f"with model {request_params['model']}, "
        f"temperature {request_params['temperature']}"
    )
    if max_tokens is not None:
        logger.debug(f"max_tokens {max_tokens}")

    for msg in messages:
        logger.debug(f"Message: {msg['role']}: {msg['content'][:50]}")

    try:
        response = CLIENT.chat.completions.create(**request_params)
        content = response.choices[0].message.content
        logger.debug("Successfully received response from vLLM API")
        logger.debug(f"Response: {content[:50]}")

        return LLMResponse(
            content=content,
            input_token_count=response.usage.prompt_tokens,
            output_token_count=response.usage.completion_tokens,
        )
    except Exception as error:
        logger.error(f"Error in vLLM API request: {str(error)}")
        raise
