"""Unit tests for Azure OpenAI API with mocked responses."""

from unittest.mock import MagicMock, patch

import pytest

from api.azure_openai import azure_openai_request
from api.enums import AzureModels


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="This is a test response from the model."
            )
        )
    ]
    return mock_response


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


def test_successful_api_call(mock_openai_response, sample_messages):
    """Test successful API call to Azure OpenAI."""
    with patch(
        "api.azure_openai.CLIENT.chat.completions.create",
        return_value=mock_openai_response,
    ) as mock_create:
        response = azure_openai_request(
            messages=sample_messages,
            model=AzureModels.GPT4O_2024_08_06,
            temperature=0.7,
            max_tokens=100,
        )

        # Verify the response
        assert response == "This is a test response from the model."

        # Verify the API was called with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        assert call_args["model"] == AzureModels.GPT4O_2024_08_06.value
        assert call_args["messages"] == sample_messages
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100


def test_api_error_handling(sample_messages):
    """Test error handling when API call fails."""
    with patch(
        "api.azure_openai.CLIENT.chat.completions.create",
        side_effect=Exception("API Error"),
    ):
        with pytest.raises(Exception) as exc_info:
            azure_openai_request(messages=sample_messages)
        assert str(exc_info.value) == "API Error"


def test_default_parameters(sample_messages):
    """Test that default parameters are used correctly."""
    with patch(
        "api.azure_openai.CLIENT.chat.completions.create",
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        ),
    ) as mock_create:
        azure_openai_request(messages=sample_messages)

        # Verify default parameters were used
        call_args = mock_create.call_args[1]
        assert call_args["model"] == AzureModels.GPT4O_2024_08_06.value
        assert call_args["temperature"] == 1.0
        assert call_args["max_tokens"] is None
        assert call_args["frequency_penalty"] == 0.0
        assert call_args["presence_penalty"] == 0.0
        assert call_args["top_p"] == 1.0
