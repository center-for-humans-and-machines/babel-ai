"""Tests for the LLMInterface class."""

from unittest.mock import patch

import pytest

from api.llm_interface import LLMInterface


@pytest.fixture
def llm_interface():
    """Create an LLMInterface instance for testing."""
    with patch("api.llm_interface.openai_request") as mock_request:
        with patch("api.llm_interface.ollama_request") as mock_ollama_request:
            with patch(
                "api.llm_interface.raven_ollama_request"
            ) as mock_raven_request:
                with patch(
                    "api.llm_interface.azure_openai_request"
                ) as mock_azure_request:
                    mock_request.return_value = "Test response"
                    mock_ollama_request.return_value = "Test response"
                    mock_raven_request.return_value = "Test response"
                    mock_azure_request.return_value = "Test response"
                    interface = LLMInterface()
                    # Store mocks in a dict for easy access
                    interface.mocks = {
                        "openai": mock_request,
                        "ollama": mock_ollama_request,
                        "raven": mock_raven_request,
                        "azure": mock_azure_request,
                    }
                    yield interface


def test_generate_with_default_params(llm_interface):
    """Test generate method with default parameters."""
    # Call generate
    prompt = "Test prompt"
    response = llm_interface.generate(prompt)

    # Verify response
    assert response == "Test response"

    # Verify azure_openai_request was called with correct parameters
    mock_request = llm_interface.mocks["openai"]
    mock_request.assert_called_once()
    call_args = mock_request.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": prompt}]
    assert call_args["model"] == "gpt-4-1106-preview"
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 100
    assert call_args["frequency_penalty"] == 0.0
    assert call_args["presence_penalty"] == 0.0
    assert call_args["top_p"] == 1.0


def test_ollama_backend(llm_interface):
    """Test that LLMInterface works with Ollama backend."""
    response = llm_interface.generate("test", provider="ollama")
    mock_ollama_request = llm_interface.mocks["ollama"]
    mock_ollama_request.assert_called_once()

    # Verify response
    assert response == "Test response"


def test_raven_backend(llm_interface):
    """Test that LLMInterface works with Raven backend."""
    response = llm_interface.generate("test", provider="raven")
    mock_raven_request = llm_interface.mocks["raven"]
    mock_raven_request.assert_called_once()

    # Verify response
    assert response == "Test response"


def test_azure_backend(llm_interface):
    """Test that LLMInterface works with Azure backend."""
    response = llm_interface.generate("test", provider="azure")
    mock_azure_request = llm_interface.mocks["azure"]
    mock_azure_request.assert_called_once()

    # Verify response
    assert response == "Test response"


def test_generate_with_custom_params(llm_interface):
    """Test generate method with custom parameters."""
    # Setup mock response
    llm_interface.mocks["openai"].return_value = "Custom response"

    # Call generate with custom parameters
    prompt = "Custom prompt"
    response = llm_interface.generate(
        prompt=prompt,
        model="custom-model",
        temperature=0.5,
        max_tokens=200,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        top_p=0.9,
    )

    # Verify response
    assert response == "Custom response"

    # Verify azure_openai_request was called with correct parameters
    mock_request = llm_interface.mocks["openai"]
    mock_request.assert_called_once()
    call_args = mock_request.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": prompt}]
    assert call_args["model"] == "custom-model"
    assert call_args["temperature"] == 0.5
    assert call_args["max_tokens"] == 200
    assert call_args["frequency_penalty"] == 0.1
    assert call_args["presence_penalty"] == 0.1
    assert call_args["top_p"] == 0.9


def test_message_history_management(llm_interface):
    """Test that message history is properly managed."""
    # Make multiple calls
    llm_interface.generate("First prompt")
    llm_interface.generate("Second prompt")

    # Verify message history
    assert len(llm_interface.messages) == 2
    assert llm_interface.messages[0] == {
        "role": "assistant",
        "content": "First prompt",
    }
    assert llm_interface.messages[1] == {
        "role": "user",
        "content": "Second prompt",
    }


def test_message_history_without_role_swap(llm_interface):
    """Test message history when swap_roles is False."""
    # Make multiple calls with swap_roles=False
    llm_interface.generate("First prompt", swap_roles=False)
    llm_interface.generate("Second prompt", swap_roles=False)

    # Verify message history - roles should remain as user
    assert len(llm_interface.messages) == 2
    assert llm_interface.messages[0] == {
        "role": "user",
        "content": "First prompt",
    }
    assert llm_interface.messages[1] == {
        "role": "user",
        "content": "Second prompt",
    }
