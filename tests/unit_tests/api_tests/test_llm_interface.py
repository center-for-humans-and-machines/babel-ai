"""Tests for the generate_response function."""

from unittest.mock import patch

import pytest

from api.enums import AzureModels, OllamaModels, OpenAIModels, Provider
from api.llm_interface import generate_response


@pytest.fixture
def mock_request_functions():
    """Mock all request functions for testing."""
    from unittest.mock import Mock

    # Create mock functions that return test responses
    mock_openai_func = Mock(return_value="Test response")
    mock_ollama_func = Mock(return_value="Test response")
    mock_raven_func = Mock(return_value="Test response")
    mock_azure_func = Mock(return_value="Test response")

    with patch.object(
        Provider.OPENAI, "get_request_function", return_value=mock_openai_func
    ) as openai_mock:
        with patch.object(
            Provider.OLLAMA,
            "get_request_function",
            return_value=mock_ollama_func,
        ) as ollama_mock:
            with patch.object(
                Provider.RAVEN,
                "get_request_function",
                return_value=mock_raven_func,
            ) as raven_mock:
                with patch.object(
                    Provider.AZURE,
                    "get_request_function",
                    return_value=mock_azure_func,
                ) as azure_mock:
                    yield {
                        Provider.OPENAI: openai_mock,
                        Provider.OLLAMA: ollama_mock,
                        Provider.RAVEN: raven_mock,
                        Provider.AZURE: azure_mock,
                    }


def test_generate_response_with_default_params(mock_request_functions):
    """Test generate_response function with default parameters."""
    messages = [{"role": "user", "content": "Test prompt"}]

    response = generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
    )

    # Verify response
    assert response == "Test response"

    # Verify get_request_function was called and the returned function was used
    mock_get_request = mock_request_functions[Provider.OPENAI]
    mock_get_request.assert_called_once()

    # The returned function should have been called with the correct parameters
    mock_function = mock_get_request.return_value
    mock_function.assert_called_once()
    call_args = mock_function.call_args[1]
    assert call_args["messages"] == messages
    assert call_args["model"] == OpenAIModels.GPT4_1106_PREVIEW
    assert call_args["temperature"] == 1.0
    assert call_args["max_tokens"] is None
    assert call_args["frequency_penalty"] == 0.0
    assert call_args["presence_penalty"] == 0.0
    assert call_args["top_p"] == 1.0


def test_generate_response_ollama_backend(mock_request_functions):
    """Test that generate_response works with Ollama backend."""
    messages = [{"role": "user", "content": "test"}]

    response = generate_response(
        messages=messages,
        provider=Provider.OLLAMA,
        model=OllamaModels.LLAMA3_70B,
    )

    mock_get_request = mock_request_functions[Provider.OLLAMA]
    mock_get_request.assert_called_once()

    mock_function = mock_get_request.return_value
    mock_function.assert_called_once()
    assert mock_function.call_args[1]["model"] == OllamaModels.LLAMA3_70B

    # Verify response
    assert response == "Test response"


def test_generate_response_raven_backend(mock_request_functions):
    """Test that generate_response works with Raven backend."""
    messages = [{"role": "user", "content": "test"}]

    response = generate_response(
        messages=messages,
        provider=Provider.RAVEN,
        model=OllamaModels.LLAMA3_70B,
    )

    mock_get_request = mock_request_functions[Provider.RAVEN]
    mock_get_request.assert_called_once()

    mock_function = mock_get_request.return_value
    mock_function.assert_called_once()
    assert mock_function.call_args[1]["model"] == OllamaModels.LLAMA3_70B

    # Verify response
    assert response == "Test response"


def test_generate_response_azure_backend(mock_request_functions):
    """Test that generate_response works with Azure backend."""
    messages = [{"role": "user", "content": "test"}]

    response = generate_response(
        messages=messages,
        provider=Provider.AZURE,
        model=AzureModels.GPT4O_2024_08_06,
    )

    mock_get_request = mock_request_functions[Provider.AZURE]
    mock_get_request.assert_called_once()

    mock_function = mock_get_request.return_value
    mock_function.assert_called_once()
    assert mock_function.call_args[1]["model"] == AzureModels.GPT4O_2024_08_06

    # Verify response
    assert response == "Test response"


def test_generate_response_with_custom_params(mock_request_functions):
    """Test generate_response function with custom parameters."""
    messages = [{"role": "user", "content": "Custom prompt"}]

    response = generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_0125_PREVIEW,
        temperature=0.5,
        max_tokens=200,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        top_p=0.9,
    )

    # Verify response
    assert response == "Test response"

    # Verify get_request_function was called and returned function was used
    mock_get_request = mock_request_functions[Provider.OPENAI]
    mock_get_request.assert_called_once()

    mock_function = mock_get_request.return_value
    mock_function.assert_called_once()
    call_args = mock_function.call_args[1]
    assert call_args["messages"] == messages
    assert call_args["model"] == OpenAIModels.GPT4_0125_PREVIEW
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
        model=OpenAIModels.GPT4_1106_PREVIEW,
    )

    # Verify response
    assert response == "Test response"

    # Verify all messages were passed correctly
    mock_get_request = mock_request_functions[Provider.OPENAI]
    mock_get_request.assert_called_once()

    mock_function = mock_get_request.return_value
    mock_function.assert_called_once()
    call_args = mock_function.call_args[1]
    assert call_args["messages"] == messages
