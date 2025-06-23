"""LLM interface for drift experiments."""

import logging
from enum import Enum
from typing import Callable, Dict, List, Optional, Type, Union

from api.azure_openai import AzureModel, azure_openai_request
from api.ollama import OllamaModel, ollama_request, raven_ollama_request
from api.openai import OpenAIModel, openai_request

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Enum for available LLM providers."""

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

    Returns:
        Generated text response
    """
    request_function = provider.get_request_function()
    return request_function(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
    )
