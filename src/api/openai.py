"""OpenAI API interface."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from api.enums import OpenAIModels

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
CLIENT = OpenAI(api_key=api_key)

logger.info("Initializing OpenAI API client.")


def openai_request(
    messages: list,
    model: OpenAIModels = OpenAIModels.GPT4_1106_PREVIEW,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Send a request to the OpenAI API using the specified GPT-4 model.

    Args:
        messages: List of message dicts (role/content etc.)
        model: OpenAI model to use
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p parameter
        max_tokens: Max tokens in response

    Returns:
        The generated text response.
    """
    logger.info(
        f"Sending request to OpenAI API with model {model.value}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
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

    try:
        response = CLIENT.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        logger.info("Successfully received response from OpenAI API")
        logger.debug(f"Response: {content[:50]}")
        return content

    except Exception as e:
        logger.error(f"Error in OpenAI API request: {str(e)}")
        raise
