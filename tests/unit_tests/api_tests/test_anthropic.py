"""Unit tests for Anthropic API with mocked responses."""

from unittest.mock import MagicMock, patch

import pytest

from api.anthropic import anthropic_request
from api.enums import AnthropicModels
from models.api import LLMResponse


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response.")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 25
    mock_response.usage.output_tokens = 10
    return mock_response


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [{"role": "user", "content": "Hello, how are you?"}]


def test_anthropic_request_success(mock_anthropic_response, sample_messages):
    """Test successful API call to Anthropic."""
    with patch(
        "api.anthropic.CLIENT.messages.create",
        return_value=mock_anthropic_response,
    ) as mock_create:
        response = anthropic_request(
            messages=sample_messages,
            model=AnthropicModels.CLAUDE_3_5_SONNET_20241022,
            temperature=0.7,
            max_tokens=100,
        )

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response."
        assert response.input_token_count == 25
        assert response.output_token_count == 10

        # Verify the API was called with correct parameters
        mock_create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            messages=sample_messages,
            temperature=0.7,
            top_p=1.0,
            max_tokens=100,
        )


def test_api_error_handling(sample_messages):
    """Test error handling when API call fails."""
    with patch(
        "api.anthropic.CLIENT.messages.create",
        side_effect=Exception("API Error"),
    ):
        with pytest.raises(Exception) as exc_info:
            anthropic_request(messages=sample_messages)
        assert str(exc_info.value) == "API Error"


def test_default_parameters(sample_messages):
    """Test that default parameters are used correctly."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 5

    with patch(
        "api.anthropic.CLIENT.messages.create",
        return_value=mock_response,
    ) as mock_create:
        response = anthropic_request(messages=sample_messages)

        # Verify the response is an LLMResponse object
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.input_token_count == 20
        assert response.output_token_count == 5

        # Verify default parameters are used
        mock_create.assert_called_once_with(
            model=AnthropicModels.CLAUDE_3_5_SONNET_20241022.value,
            messages=sample_messages,
            temperature=1.0,
            top_p=1.0,
            max_tokens=2048,  # Default applied when None provided
        )


def test_frequency_presence_penalty_warning(sample_messages):
    """Test that warning is logged when using unsupported parameters."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 5

    with patch(
        "api.anthropic.CLIENT.messages.create",
        return_value=mock_response,
    ):
        with patch("api.anthropic.logger.warning") as mock_warning:
            anthropic_request(
                messages=sample_messages,
                frequency_penalty=0.5,
                presence_penalty=0.3,
                max_tokens=2048,
            )

            # Verify warning was logged
            mock_warning.assert_called_once_with(
                "Anthropic API does not support frequency_penalty or "
                "presence_penalty. These parameters will be ignored."
            )


def test_max_tokens_none_defaults_with_warning(sample_messages):
    """Test that max_tokens=None defaults to 1024 with warning logged."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 5

    with patch(
        "api.anthropic.CLIENT.messages.create",
        return_value=mock_response,
    ) as mock_create:
        with patch("api.anthropic.logger.warning") as mock_warning:
            anthropic_request(messages=sample_messages, max_tokens=None)

            # Verify warning was logged about default max_tokens
            mock_warning.assert_called_once_with(
                "Max tokens not provided. Using default of 1024."
                "As Anthropic API requires max_tokens to be set."
            )

            # Verify max_tokens was set to default 1024
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["max_tokens"] == 2048
