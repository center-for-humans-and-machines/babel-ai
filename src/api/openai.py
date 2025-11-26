"""OpenAI API interface."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from api.enums import OpenAIModels
from models.api import LLMResponse

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
) -> LLMResponse:
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
        LLMResponse with content and token counts
    """
    request_params = {
        "model": model.value,
        "messages": messages,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if model.uses_new_parameters():
        # O4-mini model does not support the same parameters
        # as older models. Make adjustments
        request_params["temperature"] = 1.0
        request_params["max_completion_tokens"] = request_params.pop(
            "max_tokens"
        )
        request_params.pop("top_p")
        logger.warning(
            f"{model.value} model does not support "
            "temperature, top_p, or max_tokens. "
            "The only allowed value for temperature is 1.0. "
            "Removing top_p and max_tokens. "
            "Setting temperature to 1.0."
        )

    logger.info(
        "Sending request to OpenAI API "
        f"with model {request_params['model']}, "
        f"temperature {request_params['temperature']}, "
    )
    if "max_completion_tokens" in request_params:
        logger.info(f"max_tokens {request_params['max_completion_tokens']}")
    elif "max_tokens" in request_params:
        logger.info(f"max_tokens {request_params['max_tokens']}")

    for msg in messages:
        logger.debug(f"Message: {msg['role']}: {msg['content'][:50]}")

    try:
        response = CLIENT.chat.completions.create(**request_params)

        if model.uses_new_parameters():
            content = response.choices[0].message.content.content[0].text
        else:
            content = response.choices[0].message.content
        logger.info("Successfully received response from OpenAI API")
        logger.debug(f"Response: {content[:50]}")

        # Extract content and token counts
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Return LLMResponse object
        return LLMResponse(
            content=content,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
        )

    except Exception as e:
        logger.error(f"Error in OpenAI API request: {str(e)}")
        raise
