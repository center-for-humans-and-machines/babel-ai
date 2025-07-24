"""Anthropic API interface."""

import logging
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

from api.enums import AnthropicModels
from models.api import LLMResponse

# Configure logging
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Create Anthropic client
CLIENT = anthropic.Anthropic(api_key=api_key)

logger.info("Initializing Anthropic API client.")


def anthropic_request(
    messages: list,
    model: AnthropicModels = AnthropicModels.CLAUDE_3_5_SONNET_20241022,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """
    Send a request to the Anthropic API using the specified Claude model.

    Args:
        messages: List of message dicts (role/content etc.)
        model: Anthropic model to use
        temperature: Sampling temperature (0.0 to 1.0)
        frequency_penalty: Not used by Anthropic API, kept for compatibility
        presence_penalty: Not used by Anthropic API, kept for compatibility
        top_p: Top-p parameter (0.0 to 1.0)
        max_tokens: Max tokens in response (passed through to API)

    Returns:
        LLMResponse with content and token counts
    """
    logger.info(
        f"Sending request to Anthropic API with model {model.value}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
    )
    for msg in messages:
        logger.debug(f"Message: {msg['role']}: {msg['content'][:50]}")
    logger.debug(
        f"Generation parameters: "
        f"temperature {temperature}, "
        f"top_p {top_p}, "
        f"max_tokens {max_tokens}"
    )

    # Note: Anthropic API doesn't support frequency_penalty or presence_penalty
    if frequency_penalty != 0.0 or presence_penalty != 0.0:
        logger.warning(
            "Anthropic API does not support frequency_penalty or "
            "presence_penalty. These parameters will be ignored."
        )

    if max_tokens is None:
        logger.warning(
            "Max tokens not provided. Using default of 1024."
            "As Anthropic API requires max_tokens to be set."
        )
        max_tokens = 2048

    try:
        response = CLIENT.messages.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        content = response.content[0].text
        logger.info("Successfully received response from Anthropic API")
        logger.debug(f"Response: {content[:50]}")

        # Extract content and token counts
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Return LLMResponse object
        return LLMResponse(
            content=content,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
        )

    except Exception as e:
        logger.error(f"Error in Anthropic API request: {str(e)}")
        raise
