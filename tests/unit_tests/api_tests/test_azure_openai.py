"""Unit tests for Azure OpenAI API with mocked responses."""

from unittest.mock import MagicMock, patch

import pytest

from api.azure_openai import azure_openai_request
from api.enums import AzureModels
from models.api import LLMResponse


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
    # Add usage information for token counts
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 25
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 35
    return mock_response


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


def test_azure_openai_request_success(mock_openai_response, sample_messages):
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

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response from the model."
        assert response.input_token_count == 25
        assert response.output_token_count == 10

        # Verify the API was called with correct parameters
        mock_create.assert_called_once_with(
            model="gpt-4o-2024-08-06",
            messages=sample_messages,
            temperature=0.7,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            top_p=1.0,
            max_tokens=100,
        )


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
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 25

    with patch(
        "api.azure_openai.CLIENT.chat.completions.create",
        return_value=mock_response,
    ) as mock_create:
        response = azure_openai_request(messages=sample_messages)

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.input_token_count == 20
        assert response.output_token_count == 5

        # Verify default parameters are used
        mock_create.assert_called_once_with(
            model=AzureModels.GPT4O_2024_08_06.value,
            messages=sample_messages,
            temperature=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            top_p=1.0,
            max_tokens=None,
        )
