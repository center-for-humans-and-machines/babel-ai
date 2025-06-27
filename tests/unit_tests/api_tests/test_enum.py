"""Tests for Provider enum, model enums, and ModelType union."""

from typing import get_args

import pytest

from api.azure_openai import AzureModel
from api.llm_interface import ModelType, Provider
from api.ollama import OllamaModel
from api.openai import OpenAIModel


def test_provider_enum_values():
    """Test Provider enum has expected values via dot notation."""
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
    assert Provider.RAVEN.value == "raven"
    assert Provider.AZURE.value == "azure"


def test_provider_by_string():
    """Test accessing Provider enum by string value."""
    assert Provider("openai") == Provider.OPENAI
    assert Provider("ollama") == Provider.OLLAMA
    assert Provider("raven") == Provider.RAVEN
    assert Provider("azure") == Provider.AZURE


def test_get_model_enum_method():
    """Test get_model_enum returns correct enum types."""
    assert Provider.OPENAI.get_model_enum() == OpenAIModel
    assert Provider.OLLAMA.get_model_enum() == OllamaModel
    assert Provider.RAVEN.get_model_enum() == OllamaModel
    assert Provider.AZURE.get_model_enum() == AzureModel


def test_get_request_function_method():
    """Test get_request_function returns callable functions."""
    from api.azure_openai import azure_openai_request
    from api.ollama import ollama_request, raven_ollama_request
    from api.openai import openai_request

    assert Provider.OPENAI.get_request_function() == openai_request
    assert Provider.OLLAMA.get_request_function() == ollama_request
    assert Provider.RAVEN.get_request_function() == raven_ollama_request
    assert Provider.AZURE.get_request_function() == azure_openai_request


def test_invalid_provider_raises_error():
    """Test invalid provider value raises ValueError."""
    with pytest.raises(ValueError):
        Provider("invalid")


def test_openai_model_enum():
    """Test OpenAIModel enum values."""
    assert OpenAIModel.GPT4_1106_PREVIEW.value == "gpt-4-1106-preview"
    assert OpenAIModel.GPT4_0125_PREVIEW.value == "gpt-4-0125-preview"


def test_ollama_model_enum():
    """Test OllamaModel enum values."""
    assert OllamaModel.MISTRAL_7B.value == "mistral:7b-instruct"
    assert OllamaModel.LLAMA3_70B.value == "llama3:70b"
    assert OllamaModel.DEEPSEEK_R1.value == "deepseek-r1:70b"


def test_azure_model_enum():
    """Test AzureModel enum values."""
    assert AzureModel.GPT4O_2024_08_06.value == "gpt-4o-2024-08-06"


def test_model_type_contains_all_model_enums():
    """Test ModelType union contains all expected model enum classes."""
    model_types = get_args(ModelType)
    expected_types = {OpenAIModel, OllamaModel, AzureModel}
    assert set(model_types) == expected_types


def test_model_instances_are_valid_model_types():
    """Test individual model instances match ModelType union."""
    # Test OpenAI model instance
    openai_model = OpenAIModel.GPT4_1106_PREVIEW
    assert isinstance(openai_model, OpenAIModel)

    # Test Ollama model instance
    ollama_model = OllamaModel.LLAMA3_70B
    assert isinstance(ollama_model, OllamaModel)

    # Test Azure model instance
    azure_model = AzureModel.GPT4O_2024_08_06
    assert isinstance(azure_model, AzureModel)
