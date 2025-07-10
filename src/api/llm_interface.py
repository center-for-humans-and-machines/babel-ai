"""LLM interface for drift experiments."""

import logging
import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Type, Union

from api.azure_openai import AzureModel, azure_openai_request
from api.ollama import OllamaModel, ollama_request, raven_ollama_request
from api.openai import OpenAIModel, openai_request

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Enum for available LLM providers.

    This enum defines the different providers that can be used to access
    language models for generating responses in drift experiments.

    Available providers:
        OPENAI: OpenAI API provider (GPT models)
        OLLAMA: Local Ollama provider for open source models
        RAVEN: Raven provider using Ollama models
        AZURE: Azure OpenAI API provider

    Example:
        >>> provider = Provider.OPENAI
        >>> model_enum = provider.get_model_enum()
        >>> request_fn = provider.get_request_function()
        >>> response = request_fn(messages, model_enum.GPT4)
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    RAVEN = "raven"
    AZURE = "azure"

    def get_model_enum(self) -> Type[Enum]:
        """Get the corresponding model enum for this provider."""
        match self:
            case Provider.OPENAI:
                return OpenAIModel
            case Provider.OLLAMA:
                return OllamaModel
            case Provider.RAVEN:
                return OllamaModel  # Raven uses Ollama models
            case Provider.AZURE:
                return AzureModel
            case _:
                logger.error(
                    f"Invalid provider: {self}, available providers: {Provider}"  # noqa: E501
                )
                raise ValueError(f"Invalid provider: {self}")

    def get_request_function(self) -> Callable:
        """Get the corresponding request function for this provider."""
        match self:
            case Provider.OPENAI:
                return openai_request
            case Provider.OLLAMA:
                return ollama_request
            case Provider.RAVEN:
                return raven_ollama_request
            case Provider.AZURE:
                return azure_openai_request
            case _:
                logger.error(
                    f"Invalid provider: {self}, available providers: {Provider}"  # noqa: E501
                )
                raise ValueError(f"Invalid provider: {self}")


# Union type for all available models
ModelType = Union[OpenAIModel, OllamaModel, AzureModel]


def generate_response(
    messages: List[Dict[str, str]],
    provider: Provider,
    model: ModelType,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    # TODO: Configurability is missing here.
    max_retries: int = 3,
    initial_delay: float = 3.0,
) -> str:
    """Generate text response using the specified provider and model.

    Args:
        messages: List of messages in the conversation
        provider: The LLM provider to use
        model: The model to use (must be a valid model for the provider)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        frequency_penalty: Penalty for frequency
        presence_penalty: Penalty for presence
        top_p: Top-p sampling parameter
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retrying

    Returns:
        Generated text response

    Raises:
        Exception: If all retry attempts fail
    """
    request_function = provider.get_request_function()

    # Collect errors for final exception message
    errors = []

    # Try to process request for max_retries
    for attempt in range(max_retries):
        try:
            # Log attempt number
            logger.info(
                f"Attempt {attempt + 1} of {max_retries} for "
                f"provider: {provider}, model: {model}"
            )

            # Make the request
            response = request_function(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
            )

            # If successful, return response
            logger.info(
                f"Successfully generated response on attempt " f"{attempt + 1}"
            )
            return response

        except Exception as e:
            # Log error and prepare for next attempt
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            # If this was the last attempt, don't wait
            if attempt == max_retries - 1:
                break

            # Wait for next attempt with exponential backoff
            delay = initial_delay**attempt
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    # If max retries reached, raise exception with all errors
    error_summary = (
        f"Max retries ({max_retries}) reached for provider: "
        f"{provider}, model: {model}. Errors: " + " | ".join(errors)
    )
    logger.error(error_summary)
    raise Exception(error_summary)
