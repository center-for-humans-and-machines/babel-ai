"""Ollama API interface."""

import json
import logging
import os
from enum import Enum
from typing import Optional

import requests
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
# Default behavior is running locally
API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")


class OllamaModel(Enum):
    """Enum for available Ollama models."""

    LLAMA3 = "llama3"
    LLAMA3_8B = "llama3:8b"
    MISTRAL = "mistral"
    MISTRAL_7B = "mistral:7b-instruct"
    MISTRAL_7B_TEXT = "mistral:7b-text"
    MIXTRAL = "mixtral"
    LLAMA3_70B = "llama3:70b"
    LLAMA3_70B_TEXT = "llama3:70b-text"
    LLAMA33_70B = "llama3.3:70b"
    DEEPSEEK_R1 = "deepseek-r1:70b"
    GPT_2_1_5B = "gpt2:1.5b"


def ollama_request(
    messages: list,
    model: OllamaModel = OllamaModel.LLAMA3,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    api_base_url: Optional[str] = API_BASE,
    endpoint: str = "api/chat",
    stream: bool = False,
) -> str:
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
        The generated text response.
    """

    def _handle_response(
        response: requests.Response, is_streaming: bool = False
    ) -> str:
        """Handle the response from Ollama API."""
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
    logger.debug(f"Messages: {messages}")

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
        logger.debug(f"Response: {response_content}")

        return response_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Ollama API request: {str(e)}")
        raise


def ollama_request_stream(*args, **kwargs) -> str:
    """
    Send a streaming request to the Ollama API and return the full response.

    This function sends a streaming request but collects the entire response.
    For actual streaming, you would need to yield each chunk as it arrives.

    Args:
        Same as ollama_request

    Returns:
        The complete generated text response.
    """
    return ollama_request(*args, stream=True, **kwargs)


def raven_ollama_request(*args, **kwargs) -> str:
    """
    Send a request to the Ollama API using the specified model.

    Args:
        Same as ollama_request, but with different defaults:
        - model: OllamaModel.LLAMA33_70B
        - api_base_url:
            "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1"
        - endpoint: "chat/completions"

    Returns:
        The generated text response.
    """
    defaults = {
        "model": OllamaModel.LLAMA33_70B,
        "api_base_url": "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1",  # noqa: E501
        "endpoint": "chat/completions",
    }
    # Update kwargs with raven defaults
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return ollama_request(*args, **kwargs)
