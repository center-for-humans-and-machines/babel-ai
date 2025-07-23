"""Ollama API interface."""

import json
import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

from api.enums import OllamaModels
from models.api import LLMResponse

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
# Default behavior is running locally
API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

logger.info("Initializing Ollama API.")


def _estimate_token_count(text: str) -> int:
    """Estimate token count using word count approximation.

    This is a rough approximation since Ollama may
    not provide exact token counts.
    The ratio of ~0.75 words per token is commonly used as an estimate.
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words / 0.75)


def ollama_request(
    messages: list,
    model: OllamaModels = OllamaModels.LLAMA3_70B,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    api_base_url: Optional[str] = API_BASE,
    endpoint: str = "api/chat",
    stream: bool = False,
) -> LLMResponse:
    """
    Send a request to the Ollama API using the specified model.

    Args:
        messages: List of message dicts (role/content etc.)
        model: Ollama model to use
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p parameter
        max_tokens: Max tokens in response
        api_base_url:
            Base URL for Ollama API (defaults to env variable or localhost)
        endpoint: API endpoint to use (defaults to "api/chat")
        stream: Whether to stream the response

    Returns:
        LLMResponse with content and estimated token counts
    """

    def _handle_response(
        response: requests.Response, is_streaming: bool = False
    ) -> str:
        """Handle the response from Ollama API."""
        logger.debug("Handling response from Ollama API.")
        logger.debug(f"Response: {response}")

        response.raise_for_status()

        match is_streaming:
            case True:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = line.decode("utf-8")
                        if data.startswith("data: "):
                            json_data = json.loads(data[6:])
                            match json_data:
                                case {"message": {"content": content}}:
                                    full_response += content
                return full_response

            case False:
                data = response.json()
                match data:
                    case {"choices": [{"message": {"content": content}}]}:
                        # Raven format
                        return content
                    case {"message": {"content": content}}:
                        # Standard Ollama format
                        return content
                    case _:
                        logger.error(f"Unexpected response format: {data}")
                        return ""

    # Validate messages
    if (
        messages
        and isinstance(messages, list)
        and all(isinstance(m, dict) for m in messages)
    ):
        messages = messages
    else:
        raise ValueError("Messages must be a list of dictionaries")

    # Log request info
    logger.info(
        f"Sending {'streaming' if stream else 'standard'} request to Ollama API with "  # noqa: E501
        f"model {model.value}, temperature {temperature}, max_tokens {max_tokens}"  # noqa: E501
    )
    logger.debug(
        f"API url endpoint for Ollama request: {api_base_url}/{endpoint}"
    )
    for msg in messages:
        logger.debug(f"Message: {msg['role']}: {msg['content'][:50]}")
    logger.debug(
        f"Generation parameters: "
        f"temperature {temperature}, "
        f"frequency_penalty {frequency_penalty}, "
        f"presence_penalty {presence_penalty}, "
        f"top_p {top_p}, "
        f"max_tokens {max_tokens}"
    )

    # Build payload
    payload = {
        "model": model.value,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
        "stream": stream,
    }

    try:
        response = requests.post(
            url=f"{api_base_url}/{endpoint}",
            json=payload,
            stream=stream,
        )
        response_content = _handle_response(response, is_streaming=stream)

        logger.info("Successfully received response from Ollama API")
        logger.debug(f"Response: {response_content[:50]}")

        # Estimate token counts
        input_text = " ".join(msg.get("content", "") for msg in messages)
        input_tokens = _estimate_token_count(input_text)
        output_tokens = _estimate_token_count(response_content)

        # Return LLMResponse object
        return LLMResponse(
            content=response_content,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Ollama API request: {str(e)}")
        raise


def ollama_request_stream(*args, **kwargs) -> LLMResponse:
    """
    Send a streaming request to the Ollama API and return the full response.

    This function sends a streaming request but collects the entire response.
    For actual streaming, you would need to yield each chunk as it arrives.

    Args:
        Same as ollama_request

    Returns:
        LLMResponse with content and estimated token counts
    """
    return ollama_request(*args, stream=True, **kwargs)


def raven_ollama_request(*args, **kwargs) -> LLMResponse:
    """
    Send a request to the Ollama API using the specified model.

    Args:
        Same as ollama_request, but with different defaults:
        - model: OllamaModel.LLAMA33_70B
        - api_base_url:
            "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1"
        - endpoint: "chat/completions"

    Returns:
        LLMResponse with content and estimated token counts
    """
    defaults = {
        "model": OllamaModels.LLAMA3_70B,
        "api_base_url": "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1",  # noqa: E501
        "endpoint": "chat/completions",
    }
    logger.debug(f"Raven defaults: {defaults}")
    # Update kwargs with raven defaults
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return ollama_request(*args, **kwargs)
