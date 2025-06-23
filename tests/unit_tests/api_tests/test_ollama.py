"""Unit tests for Ollama API with mocked responses."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from api.ollama import (
    OllamaModel,
    ollama_request,
    ollama_request_stream,
    raven_ollama_request,
)


@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "This is a test response from the model."}
    }
    return mock_response


@pytest.fixture
def mock_ollama_stream_response():
    """Create a mock Ollama streaming response."""
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b'data: {"message": {"content": "This is a test"}}',
        b'data: {"message": {"content": " response from"}}',
        b'data: {"message": {"content": " the model."}}',
    ]
    return mock_response


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


def test_successful_api_call(mock_ollama_response, sample_messages):
    """Test successful API call to Ollama."""
    with patch(
        "requests.post", return_value=mock_ollama_response
    ) as mock_post:
        response = ollama_request(
            messages=sample_messages,
            model=OllamaModel.LLAMA3_70B,
            temperature=0.7,
            max_tokens=100,
        )

        # Verify the response
        assert response == "This is a test response from the model."

        # Verify the API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == OllamaModel.LLAMA3_70B.value
        assert call_args["json"]["messages"] == sample_messages
        assert call_args["json"]["options"]["temperature"] == 0.7
        assert call_args["json"]["options"]["num_predict"] == 100


def test_successful_streaming_api_call(
    mock_ollama_stream_response, sample_messages
):
    """Test successful streaming API call to Ollama."""
    with patch(
        "requests.post", return_value=mock_ollama_stream_response
    ) as mock_post:
        response = ollama_request(
            messages=sample_messages,
            model=OllamaModel.LLAMA3_70B,
            stream=True,
        )

        # Verify the response
        assert response == "This is a test response from the model."

        # Verify the API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["stream"] is True


def test_api_error_handling(sample_messages):
    """Test error handling when API call fails."""
    with patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("API Error"),
    ):
        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            ollama_request(messages=sample_messages)
        assert str(exc_info.value) == "API Error"


def test_default_parameters(sample_messages):
    """Test that default parameters are used correctly."""
    with patch(
        "requests.post",
        return_value=MagicMock(
            json=lambda: {"message": {"content": "Test response"}}
        ),
    ) as mock_post:
        ollama_request(messages=sample_messages)

        # Verify default parameters were used
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == OllamaModel.LLAMA3_70B.value
        assert call_args["json"]["options"]["temperature"] == 1.0
        assert call_args["json"]["options"]["num_predict"] is None
        assert call_args["json"]["options"]["frequency_penalty"] == 0.0
        assert call_args["json"]["options"]["presence_penalty"] == 0.0
        assert call_args["json"]["options"]["top_p"] == 1.0


def test_invalid_messages():
    """Test that invalid messages raise ValueError."""
    with pytest.raises(
        ValueError, match="Messages must be a list of dictionaries"
    ):
        ollama_request(messages="invalid")


def test_ollama_request_stream(mock_ollama_stream_response, sample_messages):
    """Test that ollama_request_stream correctly sets stream=True."""
    with patch(
        "requests.post", return_value=mock_ollama_stream_response
    ) as mock_post:
        response = ollama_request_stream(
            messages=sample_messages,
            model=OllamaModel.LLAMA3_70B,
        )

        # Verify the response
        assert response == "This is a test response from the model."

        # Verify the API was called with stream=True
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["stream"] is True
        assert call_args["stream"] is True


def test_raven_ollama_request(mock_ollama_response, sample_messages):
    """Test that raven_ollama_request uses correct defaults."""
    with patch(
        "requests.post", return_value=mock_ollama_response
    ) as mock_post:
        response = raven_ollama_request(messages=sample_messages)

        # Verify the response
        assert response == "This is a test response from the model."

        # Verify the API was called with Raven defaults
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == OllamaModel.LLAMA3_70B.value
        assert (
            call_args["url"]
            == "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1/chat/completions"  # noqa: E501
        )


def test_raven_ollama_request_custom_params(
    mock_ollama_response, sample_messages
):
    """Test that raven_ollama_request allows overriding defaults."""
    with patch(
        "requests.post", return_value=mock_ollama_response
    ) as mock_post:
        response = raven_ollama_request(
            messages=sample_messages,
            model=OllamaModel.LLAMA3_70B,  # Override default model
            temperature=0.5,  # Add custom parameter
        )

        # Verify the response
        assert response == "This is a test response from the model."

        # Verify custom parameters were used
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == OllamaModel.LLAMA3_70B.value
        assert call_args["json"]["options"]["temperature"] == 0.5
        # Verify other Raven defaults are still present
        assert (
            call_args["url"]
            == "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1/chat/completions"  # noqa: E501
        )
