"""Tests for the generate_response function."""

from unittest.mock import patch

import pytest

from api.azure_openai import AzureModel
from api.llm_interface import Provider, generate_response
from api.ollama import OllamaModel
from api.openai import OpenAIModel


@pytest.fixture
def mock_request_functions():
    """Mock all request functions for testing."""
    with patch("api.llm_interface.openai_request") as mock_openai:
        with patch("api.llm_interface.ollama_request") as mock_ollama:
            with patch("api.llm_interface.raven_ollama_request") as mock_raven:
                with patch(
                    "api.llm_interface.azure_openai_request"
                ) as mock_azure:
                    mock_openai.return_value = "Test response"
                    mock_ollama.return_value = "Test response"
                    mock_raven.return_value = "Test response"
                    mock_azure.return_value = "Test response"
                    yield {
                        Provider.OPENAI: mock_openai,
                        Provider.OLLAMA: mock_ollama,
                        Provider.RAVEN: mock_raven,
                        Provider.AZURE: mock_azure,
                    }


def test_generate_response_with_default_params(mock_request_functions):
    """Test generate_response function with default parameters."""
    messages = [{"role": "user", "content": "Test prompt"}]

    response = generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT4_1106_PREVIEW,
    )

    # Verify response
    assert response == "Test response"

    # Verify openai_request was called with correct parameters
    mock_request = mock_request_functions[Provider.OPENAI]
    mock_request.assert_called_once()
    call_args = mock_request.call_args[1]
    assert call_args["messages"] == messages
    assert call_args["model"] == OpenAIModel.GPT4_1106_PREVIEW
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 100
    assert call_args["frequency_penalty"] == 0.0
    assert call_args["presence_penalty"] == 0.0
    assert call_args["top_p"] == 1.0


def test_generate_response_ollama_backend(mock_request_functions):
    """Test that generate_response works with Ollama backend."""
    messages = [{"role": "user", "content": "test"}]

    response = generate_response(
        messages=messages,
        provider=Provider.OLLAMA,
        model=OllamaModel.LLAMA3_70B,
    )

    mock_ollama_request = mock_request_functions[Provider.OLLAMA]
    mock_ollama_request.assert_called_once()
    assert mock_ollama_request.call_args[1]["model"] == OllamaModel.LLAMA3_70B

    # Verify response
    assert response == "Test response"


def test_generate_response_raven_backend(mock_request_functions):
    """Test that generate_response works with Raven backend."""
    messages = [{"role": "user", "content": "test"}]

    response = generate_response(
        messages=messages,
        provider=Provider.RAVEN,
        model=OllamaModel.LLAMA3_70B,
    )

    mock_raven_request = mock_request_functions[Provider.RAVEN]
    mock_raven_request.assert_called_once()
    assert mock_raven_request.call_args[1]["model"] == OllamaModel.LLAMA3_70B

    # Verify response
    assert response == "Test response"


def test_generate_response_azure_backend(mock_request_functions):
    """Test that generate_response works with Azure backend."""
    messages = [{"role": "user", "content": "test"}]

    response = generate_response(
        messages=messages,
        provider=Provider.AZURE,
        model=AzureModel.GPT4O_2024_08_06,
    )

    mock_azure_request = mock_request_functions[Provider.AZURE]
    mock_azure_request.assert_called_once()
    assert (
        mock_azure_request.call_args[1]["model"] == AzureModel.GPT4O_2024_08_06
    )

    # Verify response
    assert response == "Test response"


def test_generate_response_with_custom_params(mock_request_functions):
    """Test generate_response function with custom parameters."""
    messages = [{"role": "user", "content": "Custom prompt"}]

    response = generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT4_0125_PREVIEW,
        temperature=0.5,
        max_tokens=200,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        top_p=0.9,
    )

    # Verify response
    assert response == "Test response"

    # Verify openai_request was called with correct parameters
    mock_request = mock_request_functions[Provider.OPENAI]
    mock_request.assert_called_once()
    call_args = mock_request.call_args[1]
    assert call_args["messages"] == messages
    assert call_args["model"] == OpenAIModel.GPT4_0125_PREVIEW
    assert call_args["temperature"] == 0.5
    assert call_args["max_tokens"] == 200
    assert call_args["frequency_penalty"] == 0.1
    assert call_args["presence_penalty"] == 0.1
    assert call_args["top_p"] == 0.9


def test_generate_response_multiple_messages(mock_request_functions):
    """Test generate_response with multiple messages in conversation."""
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
    ]

    response = generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT4_1106_PREVIEW,
    )

    # Verify response
    assert response == "Test response"

    # Verify all messages were passed correctly
    mock_request = mock_request_functions[Provider.OPENAI]
    mock_request.assert_called_once()
    call_args = mock_request.call_args[1]
    assert call_args["messages"] == messages
