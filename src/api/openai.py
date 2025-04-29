"""OpenAI API interface."""

import logging
import os
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
CLIENT = OpenAI(api_key=api_key)


def openai_request(
    messages: list,
    model: Literal[
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        # add other gpt-4.x models as desired
    ] = "gpt-4-1106-preview",
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1000,
) -> str:
    """
    Send a request to the OpenAI API using the specified GPT-4 model.

    Args:
        messages: List of message dicts (role/content etc.)
        model: OpenAI model name (e.g. "gpt-4-1106-preview")
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p parameter
        max_tokens: Max tokens in response

    Returns:
        The generated text response.
    """
    logger.info(
        f"Sending request to OpenAI API with model {model}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
    )
    logger.debug(f"Messages: {messages}")

    try:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        logger.info("Successfully received response from OpenAI API")
        logger.debug(f"Response: {content}")
        return content

    except Exception as e:
        logger.error(f"Error in OpenAI API request: {str(e)}")
        raise
