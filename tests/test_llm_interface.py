"""Tests for the LLMInterface class."""

from unittest.mock import patch

import pytest

from src.babel_ai.llm_interface import LLMInterface


@pytest.fixture
def llm_interface():
    """Create an LLMInterface instance for testing."""
    return LLMInterface()


@patch("src.babel_ai.llm_interface.azure_openai_request")
def test_generate_with_default_params(mock_azure_request, llm_interface):
    """Test generate method with default parameters."""
    # Setup mock response
    mock_azure_request.return_value = "Test response"

    # Call generate
    prompt = "Test prompt"
    response = llm_interface.generate(prompt)

    # Verify response
    assert response == "Test response"

    # Verify azure_openai_request was called with correct parameters
    mock_azure_request.assert_called_once()
    call_args = mock_azure_request.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": prompt}]
    assert call_args["model"] == "gpt-4o-2024-08-06"
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 100
    assert call_args["frequency_penalty"] == 0.0
    assert call_args["presence_penalty"] == 0.0
    assert call_args["top_p"] == 1.0


@patch("src.babel_ai.llm_interface.azure_openai_request")
def test_generate_with_custom_params(mock_azure_request, llm_interface):
    """Test generate method with custom parameters."""
    # Setup mock response
    mock_azure_request.return_value = "Custom response"

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
    mock_azure_request.assert_called_once()
    call_args = mock_azure_request.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": prompt}]
    assert call_args["model"] == "custom-model"
    assert call_args["temperature"] == 0.5
    assert call_args["max_tokens"] == 200
    assert call_args["frequency_penalty"] == 0.1
    assert call_args["presence_penalty"] == 0.1
    assert call_args["top_p"] == 0.9


@patch("src.babel_ai.llm_interface.azure_openai_request")
def test_message_history_management(mock_azure_request, llm_interface):
    """Test that message history is properly managed."""
    # Setup mock response
    mock_azure_request.return_value = "Response"

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


@patch("src.babel_ai.llm_interface.azure_openai_request")
def test_message_history_without_role_swap(mock_azure_request, llm_interface):
    """Test message history when swap_roles is False."""
    # Setup mock response
    mock_azure_request.return_value = "Response"

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
