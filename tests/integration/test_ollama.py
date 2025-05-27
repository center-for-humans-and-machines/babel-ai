"""Integration tests for Ollama API.

These tests make real API calls to Ollama to verify basic functionality.
They will fail if the Ollama server is not running or if there are
API access issues.
"""

from api.ollama import (
    ollama_request,
    ollama_request_stream,
    raven_ollama_request,
)


def test_basic_api_functionality():
    """Test basic API functionality with default parameters."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, this is a test!'"},
    ]

    response = ollama_request(messages=messages)

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "test" in response.lower()


def test_basic_streaming_functionality():
    """Test basic streaming API functionality."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, this is a test!'"},
    ]

    response = ollama_request_stream(messages=messages)

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "test" in response.lower()


def test_raven_server_api():
    """Test API functionality with Raven server endpoint."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, this is a test!'"},
    ]

    response = raven_ollama_request(
        messages=messages,
        api_base_url="https://hpc-llm-inference-fastapi.chm.mpib-berlin.mpg.de/v1",  # noqa: E501
        model="llama3.3:70b",
        endpoint="chat/completions",
    )

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "test" in response.lower()
