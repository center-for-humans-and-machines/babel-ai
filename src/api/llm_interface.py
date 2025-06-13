"""LLM interface for drift experiments."""

import logging
from enum import Enum
from typing import Callable, Dict, List, Type, Union

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


class LLMInterface:
    """Protocol defining the interface for LLM providers."""

    def __init__(self):
        """Initialize the LLM interface."""
        self.messages: List[Dict[str, str]] = []

    def generate(
        self,
        prompt: str,
        provider: Provider = Provider.OPENAI,
        model: ModelType = OpenAIModel.GPT4_1106_PREVIEW,
        temperature: float = 0.7,
        max_tokens: int = 100,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float = 1.0,
        swap_roles: bool = True,
    ) -> str:
        """Generate text from the given prompt.

        Args:
            prompt: The input prompt
            provider: The LLM provider to use
            model: The model to use (must be a valid model for the provider)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            frequency_penalty: Penalty for frequency
            presence_penalty: Penalty for presence
            top_p: Top-p sampling parameter
            swap_roles: Whether to swap roles of previous messages

        Returns:
            Generated text response
        """
        if swap_roles and self.messages:
            # Swap roles of previous messages
            logger.debug(
                "Swapping roles of previous messages."
                f" Role of last message: {self.messages[-1]['role']}"
            )
            for message in self.messages:
                message["role"] = (
                    "assistant" if message["role"] == "user" else "user"
                )

        # Add new prompt
        logger.debug(f"Adding new prompt: {prompt}")
        self.messages.append({"role": "user", "content": prompt})

        request_function = provider.get_request_function()
        return request_function(
            messages=self.messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
        )
