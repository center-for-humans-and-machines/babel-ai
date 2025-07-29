"""Azure OpenAI API interface."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from api.enums import AzureModels
from models.api import LLMResponse

# Configure logging
logger = logging.getLogger(__name__)

# Define the Azure OpenAI endpoint and key
load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("AZURE_KEY")
api_version = "2024-12-01-preview"

# Create an instance of the AzureOpenAI client
CLIENT = AzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)

logger.info("Initialized Azure OpenAI client.")


def azure_openai_request(
    messages: list,
    model: AzureModels = AzureModels.GPT4O_2024_08_06,
    temperature: float = 1.0,  # 0.0 to 1.0 higher = more creative
    frequency_penalty: float = 0.0,  # -2.0 to 2.0 higher = less repetition
    presence_penalty: float = 0.0,  # -2.0 to 2.0 higher = more diverse
    top_p: float = 1.0,  # 0.0 to 1.0 higher = more creative
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """Send a request to Azure OpenAI API.

    Args:
        messages: List of message dictionaries
        model: Model to use
        temperature: Sampling temperature
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate

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

    if model == AzureModels.O4_MINI_2025_04_16:
        # O4-mini model does not support the same parameters
        # as older models. Make adjustments
        request_params["temperature"] = 1.0
        request_params["max_completion_tokens"] = request_params.pop(
            "max_tokens"
        )
        request_params.pop("top_p")
        logger.warning(
            "O4-mini model does not support "
            "temperature, top_p, or max_tokens. "
            "The only allowed value for temperature is 1.0. "
            "Removing top_p and max_tokens. "
            "Setting temperature to 1.0."
        )

    logger.info(
        "Sending request to Azure OpenAI API "
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
        # Send the prompt to AzureOpenAI with unpacked request params
        response = CLIENT.chat.completions.create(**request_params)

        # Log response
        logger.info("Successfully received response from Azure OpenAI API")
        logger.debug(f"Response: {response.choices[0].message.content[:50]}")

        # Extract content and token counts
        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Return LLMResponse object
        return LLMResponse(
            content=content,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
        )

    except Exception as e:
        logger.error(f"Error in Azure OpenAI API request: {str(e)}")
        raise
