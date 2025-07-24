"""Enums for API module."""

import logging
from enum import Enum
from typing import Callable, Type, Union

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
    ANTHROPIC = "anthropic"

    def get_model_enum(self) -> Type[Enum]:
        """Get the corresponding model enum for this provider."""
        logger.info(f"Getting model enum for provider: {self.value}")
        match self:
            case Provider.OPENAI:
                return OpenAIModels
            case Provider.OLLAMA:
                return OllamaModels
            case Provider.RAVEN:
                return OllamaModels  # Raven uses Ollama models
            case Provider.AZURE:
                return AzureModels
            case Provider.ANTHROPIC:
                return AnthropicModels
            case _:
                logger.error(
                    f"Invalid provider: {self}, available providers: "
                    f"{Provider}"
                )
                raise ValueError(f"Invalid provider: {self}")

    def get_request_function(self) -> Callable:
        """Get the corresponding request function for this provider."""
        logger.info(f"Getting request function for provider: {self.value}")
        # Import here to avoid circular imports
        from api.anthropic import anthropic_request
        from api.azure_openai import azure_openai_request
        from api.ollama import ollama_request, raven_ollama_request
        from api.openai import openai_request

        match self:
            case Provider.OPENAI:
                return openai_request
            case Provider.OLLAMA:
                return ollama_request
            case Provider.RAVEN:
                return raven_ollama_request
            case Provider.AZURE:
                return azure_openai_request
            case Provider.ANTHROPIC:
                return anthropic_request
            case _:
                logger.error(
                    f"Invalid provider: {self}, available providers: "
                    f"{Provider}"
                )
                raise ValueError(f"Invalid provider: {self}")


class OpenAIModels(Enum):
    """Enum for available OpenAI models.

    This enum defines the different OpenAI
    models that can be used for generating
    responses in drift experiments.

    Available models:
        GPT4_1106_PREVIEW: GPT-4 Turbo preview model (November 2023)
        GPT4_0125_PREVIEW: GPT-4 Turbo preview model (January 2024)

    Example:
        >>> model = OpenAIModel.GPT4_1106_PREVIEW
        >>> response = openai_request(messages, model=model)
    """

    GPT4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT4_0125_PREVIEW = "gpt-4-0125-preview"
    O3_2025_04_16 = "o3-2025-04-16"
    O4_MINI_2025_04_16 = "o4-mini-2025-04-16"


class OllamaModels(Enum):
    """Enum for available Ollama models.

    This enum defines the different Ollama
    models that can be used for generating
    responses in drift experiments.

    Available models:
        MISTRAL_7B: Mistral 7B instruct model
        MISTRAL_7B_TEXT: Mistral 7B text model
        LLAMA3_70B: Llama 3 70B instruct model
        LLAMA3_70B_TEXT: Llama 3 70B text model
        DEEPSEEK_R1: DeepSeek 70B model
        GPT_2_1_5B: GPT-2 1.5B model

    Example:
        >>> model = OllamaModel.MISTRAL_7B
        >>> response = ollama_request(messages, model=model)
    """

    MISTRAL_7B = "mistral:7b-instruct"
    MISTRAL_7B_TEXT = "mistral:7b-text"
    LLAMA3_70B = "llama3:70b"
    LLAMA3_70B_TEXT = "llama3:70b-text"
    DEEPSEEK_R1 = "deepseek-r1:70b"
    GPT_2_1_5B = "gpt2:1.5b"


class AzureModels(Enum):
    """Enum for available Azure OpenAI models.

    This enum defines the different Azure OpenAI models that can be used for
    generating responses in drift experiments.

    Available models:
        GPT4O_2024_08_06: GPT-4 Optimized model (August 2024)

    Example:
        >>> model = AzureModel.GPT4O_2024_08_06
        >>> response = azure_openai_request(messages, model=model)
    """

    GPT4O_2024_08_06 = "gpt-4o-2024-08-06"
    O3_2025_04_16 = "o3-2025-04-16"
    O4_MINI_2025_04_16 = "o4-mini-2025-04-16"


class AnthropicModels(Enum):
    """Enum for available Anthropic models.

    This enum defines the different Anthropic Claude models that can be used
    for generating responses in drift experiments.

    Available models:
        CLAUDE_OPUS_4_20250514: Claude Opus 4 model (May 2025)
        CLAUDE_3_5_SONNET_20241022: Claude 3.5 Sonnet model (October 2024)
        CLAUDE_3_5_HAIKU_20241022: Claude 3.5 Haiku model (October 2024)

    Example:
        >>> model = AnthropicModels.CLAUDE_OPUS_4_20250514
        >>> response = anthropic_request(messages, model=model)
    """

    CLAUDE_OPUS_4_20250514 = "claude-opus-4-20250514"
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"


# Union type for all available models
APIModels = Union[OpenAIModels, OllamaModels, AzureModels, AnthropicModels]
