"""Unit tests for Ollama API with mocked responses."""

from unittest.mock import MagicMock, patch

import pytest

from api.enums import OllamaModels
from api.ollama import (
    _estimate_token_count,
    ollama_request,
    ollama_request_stream,
    raven_ollama_request,
)
from models.api import LLMResponse


@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "This is a test response from the model."}
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_ollama_stream_response():
    """Create a mock Ollama streaming response."""
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b'data: {"message": {"content": "This "}}',
        b'data: {"message": {"content": "is "}}',
        b'data: {"message": {"content": "a test."}}',
    ]
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_ollama_raven_response():
    """Create a mock Raven (OpenAI-compatible) response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "This is a test response from the model."}}
        ]
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


def test_estimate_token_count():
    """Test the token count estimation function."""
    # Test empty string
    assert _estimate_token_count("") == 0

    # Test simple text (3 words / 0.75 = 4 tokens)
    assert _estimate_token_count("Here a test") == 4

    # Test longer text
    text = "This is a longer piece of text with many words"
    expected = int(len(text.split()) / 0.75)  # 11 words / 0.75 = 14 tokens
    assert _estimate_token_count(text) == expected


def test_successful_api_call(mock_ollama_response, sample_messages):
    """Test successful API call to Ollama."""
    with patch(
        "requests.post", return_value=mock_ollama_response
    ) as mock_post:
        response = ollama_request(
            messages=sample_messages,
            model=OllamaModels.LLAMA3_70B,
            temperature=0.7,
            max_tokens=100,
        )

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from the model."
        assert (
            response.input_token_count > 0
        )  # Should have estimated input tokens
        assert (
            response.output_token_count > 0
        )  # Should have estimated output tokens

        # Verify the API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == OllamaModels.LLAMA3_70B.value
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
            model=OllamaModels.LLAMA3_70B,
            temperature=0.7,
            stream=True,
        )

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert (
            response.content == "This is a test."
        )  # Concatenated streaming content
        assert response.input_token_count > 0
        assert response.output_token_count > 0

        # Verify streaming was enabled
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["stream"] is True


def test_api_error_handling(sample_messages):
    """Test error handling when API call fails."""
    with patch("requests.post", side_effect=Exception("API Error")):
        with pytest.raises(Exception) as exc_info:
            ollama_request(messages=sample_messages)
        assert str(exc_info.value) == "API Error"


def test_default_parameters(mock_ollama_response, sample_messages):
    """Test that default parameters are used correctly."""
    with patch(
        "requests.post", return_value=mock_ollama_response
    ) as mock_post:
        response = ollama_request(messages=sample_messages)

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from the model."

        # Verify default parameters were used
        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == OllamaModels.LLAMA3_70B.value
        assert call_args["json"]["options"]["temperature"] == 1.0
        assert call_args["json"]["options"]["num_predict"] is None
        assert call_args["json"]["stream"] is False


def test_invalid_messages():
    """Test error handling for invalid messages format."""
    with pytest.raises(ValueError) as exc_info:
        ollama_request(messages="invalid")
    assert "Messages must be a list of dictionaries" in str(exc_info.value)


def test_ollama_request_stream(mock_ollama_stream_response, sample_messages):
    """Test that ollama_request_stream correctly sets stream=True."""
    with patch(
        "requests.post", return_value=mock_ollama_stream_response
    ) as mock_post:
        response = ollama_request_stream(
            messages=sample_messages,
            model=OllamaModels.LLAMA3_70B,
        )

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test."

        # Verify streaming was enabled
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["json"]["stream"] is True


def test_raven_ollama_request(mock_ollama_raven_response, sample_messages):
    """Test that raven_ollama_request uses correct defaults."""
    with patch(
        "requests.post", return_value=mock_ollama_raven_response
    ) as mock_post:
        response = raven_ollama_request(messages=sample_messages)

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from the model."

        # Verify Raven-specific defaults were used
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert (
            "https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1/chat/completions"  # noqa: E501
            in call_args[1]["url"]
        )


def test_raven_ollama_request_custom_params(
    mock_ollama_raven_response, sample_messages
):
    """Test that raven_ollama_request allows overriding defaults."""
    with patch(
        "requests.post", return_value=mock_ollama_raven_response
    ) as mock_post:
        response = raven_ollama_request(
            messages=sample_messages,
            model=OllamaModels.MISTRAL_7B,
            api_base_url="http://custom-url.com",
        )

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from the model."

        # Verify custom parameters were used
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "http://custom-url.com/chat/completions" in call_args[1]["url"]
        assert call_args[1]["json"]["model"] == OllamaModels.MISTRAL_7B.value
