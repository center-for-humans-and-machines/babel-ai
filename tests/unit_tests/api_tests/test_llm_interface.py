"""Tests for the generate_response function."""

import threading
from unittest.mock import patch

import pytest

from api.enums import AzureModels, OllamaModels, OpenAIModels, Provider
from api.llm_interface import LLMInterface
from models.api import LLMResponse


def test_llm_interface_singleton():
    """Test that LLMInterface implements singleton pattern correctly."""
    # Create instances in parallel threads
    results = []

    def create_instance():
        results.append(LLMInterface())

    threads = [threading.Thread(target=create_instance) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All instances should be identical
    first = results[0]
    for instance in results[1:]:
        assert first is instance
        assert id(first) == id(instance)
    instance1 = LLMInterface()
    instance2 = LLMInterface()

    # Both should be the same object
    assert instance1 is instance2
    assert id(instance1) == id(instance2)


@pytest.fixture
def mock_request_functions():
    """Mock all request functions for testing."""
    from unittest.mock import Mock

    # Create mock functions that return LLMResponse objects
    mock_response = LLMResponse(
        content="Test response", input_token_count=10, output_token_count=5
    )

    mock_openai_func = Mock(return_value=mock_response)
    mock_ollama_func = Mock(return_value=mock_response)
    mock_raven_func = Mock(return_value=mock_response)
    mock_azure_func = Mock(return_value=mock_response)

    with patch.object(
        Provider.OPENAI, "get_request_function", return_value=mock_openai_func
    ), patch.object(
        Provider.OLLAMA, "get_request_function", return_value=mock_ollama_func
    ), patch.object(
        Provider.RAVEN, "get_request_function", return_value=mock_raven_func
    ), patch.object(
        Provider.AZURE, "get_request_function", return_value=mock_azure_func
    ):
        yield {
            "openai": mock_openai_func,
            "ollama": mock_ollama_func,
            "raven": mock_raven_func,
            "azure": mock_azure_func,
        }


def test_generate_response_with_default_params(mock_request_functions):
    """Test generate_response function with default parameters."""
    messages = [{"role": "user", "content": "Test prompt"}]

    response = LLMInterface.generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
    )

    # Verify response is a string (extracted from LLMResponse)
    assert isinstance(response, str)
    assert response == "Test response"

    # Verify the request function was called with correct parameters
    mock_func = mock_request_functions["openai"]
    mock_func.assert_called_once()
    call_args = mock_func.call_args[1]
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

    response = LLMInterface.generate_response(
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

    response = LLMInterface.generate_response(
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

    response = LLMInterface.generate_response(
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
    messages = [{"role": "user", "content": "Test prompt"}]

    response = LLMInterface.generate_response(
        messages=messages,
        provider=Provider.OLLAMA,
        model=OllamaModels.LLAMA3_70B,
        temperature=0.7,
        max_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        top_p=0.9,
    )

    # Verify response is a string (extracted from LLMResponse)
    assert isinstance(response, str)
    assert response == "Test response"

    # Verify the request function was called with custom parameters
    mock_func = mock_request_functions["ollama"]
    mock_func.assert_called_once()
    call_args = mock_func.call_args[1]
    assert call_args["messages"] == messages
    assert call_args["model"] == OllamaModels.LLAMA3_70B
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 100
    assert call_args["frequency_penalty"] == 0.5
    assert call_args["presence_penalty"] == 0.3
    assert call_args["top_p"] == 0.9


def test_generate_response_multiple_messages(mock_request_functions):
    """Test generate_response with multiple messages in conversation."""
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
    ]

    response = LLMInterface.generate_response(
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


def test_generate_response_different_providers(mock_request_functions):
    """Test generate_response works with different providers."""
    messages = [{"role": "user", "content": "Test prompt"}]

    # Test OpenAI
    response_openai = LLMInterface.generate_response(
        messages=messages,
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
    )
    assert isinstance(response_openai, str)
    assert response_openai == "Test response"

    # Test Ollama
    response_ollama = LLMInterface.generate_response(
        messages=messages,
        provider=Provider.OLLAMA,
        model=OllamaModels.LLAMA3_70B,
    )
    assert isinstance(response_ollama, str)
    assert response_ollama == "Test response"

    # Test Azure
    response_azure = LLMInterface.generate_response(
        messages=messages,
        provider=Provider.AZURE,
        model=AzureModels.GPT4O_2024_08_06,
    )
    assert isinstance(response_azure, str)
    assert response_azure == "Test response"

    # Verify each provider was called
    mock_request_functions["openai"].assert_called_once()
    mock_request_functions["ollama"].assert_called_once()
    mock_request_functions["azure"].assert_called_once()


def test_generate_response_retry_mechanism():
    """Test retry mechanism on failures."""
    messages = [{"role": "user", "content": "Test prompt"}]

    # Mock a function that fails twice then succeeds
    mock_response = LLMResponse(
        content="Success response", input_token_count=10, output_token_count=5
    )

    with patch.object(
        Provider.OPENAI, "get_request_function"
    ) as mock_get_func:
        mock_func = mock_get_func.return_value
        mock_func.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            mock_response,  # Third attempt succeeds
        ]

        with patch("time.sleep"):  # Mock sleep to speed up test
            response = LLMInterface.generate_response(
                messages=messages,
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_1106_PREVIEW,
                max_retries=3,
            )

        # Should eventually succeed and return string content
        assert isinstance(response, str)
        assert response == "Success response"
        assert mock_func.call_count == 3


def test_generate_response_max_retries_exceeded():
    """Test that exception is raised when max retries are exceeded."""
    messages = [{"role": "user", "content": "Test prompt"}]

    with patch.object(
        Provider.OPENAI, "get_request_function"
    ) as mock_get_func:
        mock_func = mock_get_func.return_value
        mock_func.side_effect = Exception("API Error")

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(Exception) as exc_info:
                LLMInterface.generate_response(
                    messages=messages,
                    provider=Provider.OPENAI,
                    model=OpenAIModels.GPT4_1106_PREVIEW,
                    max_retries=2,
                )

        # Should contain information about retries
        assert "Max retries (2) reached" in str(exc_info.value)
        assert mock_func.call_count == 2


def test_generate_response_budget_logging(mock_request_functions):
    """Test that budget logging is called correctly."""
    from unittest.mock import Mock, patch

    messages = [{"role": "user", "content": "Test prompt"}]

    # Create a mock budget tracker with a mock add_usage method
    mock_budget_tracker = Mock()
    mock_usage_summary = {
        "total_cost": 0.000015,
        "cumulative_total_cost": 0.000015,
    }
    mock_budget_tracker.add_usage.return_value = mock_usage_summary

    with patch(
        "api.llm_interface.BudgetTracker", return_value=mock_budget_tracker
    ):
        response = LLMInterface.generate_response(
            messages=messages,
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
        )

    # Verify response
    assert response == "Test response"

    # Verify budget tracker was called with correct parameters
    mock_budget_tracker.add_usage.assert_called_once_with(
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
        input_tokens=10,
        output_tokens=5,
    )
