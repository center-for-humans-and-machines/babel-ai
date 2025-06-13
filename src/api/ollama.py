"""Ollama API interface."""

import json
import logging
import os
from typing import Literal, Optional

import requests
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
# Default behavior is running locally
api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")


def ollama_request(
    messages: list,
    model: Literal[
        "llama3",
        "llama3:8b",
        "llama3:70b",
        "mistral",
        "mixtral",
        # add other models as desired
    ] = "llama3",
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    api_base_url: Optional[str] = None,
) -> str:
    """
    Send a request to the Ollama API using the specified model.

    Args:
        messages: List of message dicts (role/content etc.)
        model: Ollama model name (e.g. "llama3")
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p parameter
        max_tokens: Max tokens in response
        api_base_url:
            Base URL for Ollama API (defaults to env variable or localhost)

    Returns:
        The generated text response.
    """
    if api_base_url is None:
        api_base_url = api_base

    endpoint = f"{api_base_url}/api/chat"

    logger.info(
        f"Sending request to Ollama API with model {model}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
    )
    logger.debug(f"Messages: {messages}")

    # Convert OpenAI-style messages to Ollama format if needed
    if (
        messages
        and isinstance(messages, list)
        and all(isinstance(m, dict) for m in messages)
    ):
        # Ollama format is the same as OpenAI's for chat messages
        ollama_messages = messages

    payload = {
        "model": model,
        "messages": ollama_messages,
        "options": {
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()

        print(response)

        data = response.json()
        content = data.get("message", {}).get("content", "")

        logger.info("Successfully received response from Ollama API")
        logger.debug(f"Response: {content}")
        return content

    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Ollama API request: {str(e)}")
        raise


def ollama_request_stream(
    messages: list,
    model: str = "llama3",
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    api_base_url: Optional[str] = None,
) -> str:
    """
    Send a streaming request to the Ollama API and return the full response.

    This function sends a streaming request but collects the entire response.
    For actual streaming, you would need to yield each chunk as it arrives.

    Args:
        Same as ollama_request

    Returns:
        The complete generated text response.
    """
    if api_base_url is None:
        api_base_url = api_base

    endpoint = f"{api_base_url}/api/chat"

    logger.info(
        f"Sending streaming request to Ollama API with model {model}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
    )

    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    try:
        response = requests.post(endpoint, json=payload, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if data.startswith("data: "):
                    json_data = json.loads(data[6:])
                    if (
                        "message" in json_data
                        and "content" in json_data["message"]
                    ):
                        chunk = json_data["message"]["content"]
                        full_response += chunk

        logger.info("Successfully received streamed response from Ollama API")
        logger.debug(f"Complete response: {full_response}")
        return full_response

    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Ollama API streaming request: {str(e)}")
        raise


def raven_ollama_request(
    messages: list,
    model: Literal[
        "llama3.3:70b",
    ] = "llama3.3:70b",
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    api_base_url: Optional[
        str
    ] = "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1",
    endpoint: Optional[str] = "chat/completions",
) -> str:
    """
    Send a request to the Ollama API using the specified model.

    Args:
        messages: List of message dicts (role/content etc.)
        model: Ollama model name (e.g. "llama3")
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p parameter
        max_tokens: Max tokens in response
        api_base_url:
            Base URL for Ollama API (defaults to env variable or localhost)

    Returns:
        The generated text response.
    """
    if api_base_url is None:
        api_base_url = api_base

    endpoint = f"{api_base_url}/{endpoint}"

    logger.info(
        f"Sending request to Ollama API with model {model}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
    )
    logger.debug(f"Messages: {messages}")

    # Convert OpenAI-style messages to Ollama format if needed
    if (
        messages
        and isinstance(messages, list)
        and all(isinstance(m, dict) for m in messages)
    ):
        # Ollama format is the same as OpenAI's for chat messages
        ollama_messages = messages

    payload = {
        "model": model,
        "messages": ollama_messages,
        "options": {
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()

        print(response)

        data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        logger.info("Successfully received response from Ollama API")
        logger.debug(f"Response: {content}")
        return content

    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Ollama API request: {str(e)}")
        raise
