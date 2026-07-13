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

# Create OpenAI-compatible vLLM client
CLIENT = OpenAI(base_url=base_url, api_key=api_key)

logger.debug("Initializing vLLM API client.")


def vllm_request(
    messages: list,
    model: VLLMModels = VLLMModels.DEFAULT,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """Send a request to the OpenAI-compatible vLLM endpoint."""
    request_params = {
        "model": model.value,
        "messages": messages,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
        "max_tokens": max_tokens,
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
