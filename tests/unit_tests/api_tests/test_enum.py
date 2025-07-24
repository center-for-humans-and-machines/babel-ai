"""Tests for Provider enum, model enums, and ModelType union."""

from typing import get_args

import pytest

from api.enums import (
    AnthropicModels,
    APIModels,
    AzureModels,
    OllamaModels,
    OpenAIModels,
    Provider,
)


def test_provider_enum_values():
    """Test Provider enum has expected values via dot notation."""
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
    assert Provider.RAVEN.value == "raven"
    assert Provider.AZURE.value == "azure"
    assert Provider.ANTHROPIC.value == "anthropic"


def test_provider_by_string():
    """Test accessing Provider enum by string value."""
    assert Provider("openai") == Provider.OPENAI
    assert Provider("ollama") == Provider.OLLAMA
    assert Provider("raven") == Provider.RAVEN
    assert Provider("azure") == Provider.AZURE
    assert Provider("anthropic") == Provider.ANTHROPIC


def test_get_model_enum_method():
    """Test get_model_enum returns correct enum types."""
    assert Provider.OPENAI.get_model_enum() == OpenAIModels
    assert Provider.OLLAMA.get_model_enum() == OllamaModels
    assert Provider.RAVEN.get_model_enum() == OllamaModels
    assert Provider.AZURE.get_model_enum() == AzureModels
    assert Provider.ANTHROPIC.get_model_enum() == AnthropicModels


def test_get_request_function_method():
    """Test get_request_function returns callable functions."""
    from api.anthropic import anthropic_request
    from api.azure_openai import azure_openai_request
    from api.ollama import ollama_request, raven_ollama_request
    from api.openai import openai_request

    assert Provider.OPENAI.get_request_function() == openai_request
    assert Provider.OLLAMA.get_request_function() == ollama_request
    assert Provider.RAVEN.get_request_function() == raven_ollama_request
    assert Provider.AZURE.get_request_function() == azure_openai_request
    assert Provider.ANTHROPIC.get_request_function() == anthropic_request


def test_invalid_provider_raises_error():
    """Test invalid provider value raises ValueError."""
    with pytest.raises(ValueError):
        Provider("invalid")


def test_openai_model_enum():
    """Test OpenAIModel enum values."""
    assert OpenAIModels.GPT4_1106_PREVIEW.value == "gpt-4-1106-preview"
    assert OpenAIModels.GPT4_0125_PREVIEW.value == "gpt-4-0125-preview"


def test_ollama_model_enum():
    """Test OllamaModel enum values."""
    assert OllamaModels.MISTRAL_7B.value == "mistral:7b-instruct"
    assert OllamaModels.LLAMA3_70B.value == "llama3:70b"
    assert OllamaModels.DEEPSEEK_R1.value == "deepseek-r1:70b"


def test_azure_model_enum():
    """Test AzureModel enum values."""
    assert AzureModels.GPT4O_2024_08_06.value == "azure-gpt-4o-2024-08-06"


def test_anthropic_model_enum():
    """Test AnthropicModel enum values."""
    assert (
        AnthropicModels.CLAUDE_OPUS_4_20250514.value
        == "claude-opus-4-20250514"
    )
    assert (
        AnthropicModels.CLAUDE_SONNET_4_20250514.value
        == "claude-sonnet-4-20250514"
    )
    assert (
        AnthropicModels.CLAUDE_3_5_HAIKU_20241022.value
        == "claude-3-5-haiku-20241022"
    )


def test_model_type_contains_all_model_enums():
    """Test ModelType union contains all expected model enum classes."""
    model_types = get_args(APIModels)
    expected_types = {OpenAIModels, OllamaModels, AzureModels, AnthropicModels}
    assert set(model_types) == expected_types


def test_model_instances_are_valid_model_types():
    """Test individual model instances match ModelType union."""
    # Test OpenAI model instance
    openai_model = OpenAIModels.GPT4_1106_PREVIEW
    assert isinstance(openai_model, OpenAIModels)

    # Test Ollama model instance
    ollama_model = OllamaModels.LLAMA3_70B
    assert isinstance(ollama_model, OllamaModels)

    # Test Azure model instance
    azure_model = AzureModels.GPT4O_2024_08_06
    assert isinstance(azure_model, AzureModels)

    # Test Anthropic model instance
    anthropic_model = AnthropicModels.CLAUDE_SONNET_4_20250514
    assert isinstance(anthropic_model, AnthropicModels)
