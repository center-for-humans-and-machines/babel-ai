"""Azure OpenAI API interface."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from api.enums import AzureModels

# Configure logging
logger = logging.getLogger(__name__)

# Define the Azure OpenAI endpoint and key
load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("AZURE_KEY")
api_version = "2024-10-01-preview"

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
) -> str:
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
        Generated text response
    """
    logger.info(
        f"Sending request to Azure OpenAI API with model {model.value}, "
        f"temperature {temperature}, max_tokens {max_tokens}"
    )
    for msg in messages:
        logger.debug(f"Message: {msg['role']}: {msg['content'][:50]}")

    try:
        # Send the prompt to AzureOpenAI
        response = CLIENT.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Log response
        logger.info("Successfully received response from Azure OpenAI API")
        logger.debug(f"Response: {response.choices[0].message.content[:50]}")

        # Return the response
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in Azure OpenAI API request: {str(e)}")
        raise
