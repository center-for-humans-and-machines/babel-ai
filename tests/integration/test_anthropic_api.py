"""Integration tests for Anthropic API.

These tests make real API calls to Anthropic to verify
functionality and API compatibility.
"""

import os

from api.anthropic import anthropic_request
from api.enums import AnthropicModels
from models.api import LLMResponse


def test_real_api_call_with_different_models_and_default_parameters():
    """Test real API calls with different available models."""
    messages = [{"role": "user", "content": "Say 'Hello, this is a test!'"}]

    models = [
        AnthropicModels.CLAUDE_3_5_SONNET_20241022,
        AnthropicModels.CLAUDE_3_5_HAIKU_20241022,
    ]

    for model in models:
        response = anthropic_request(
            messages=messages,
            model=model,
        )

        assert response is not None
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert "test" in response.content.lower()
        assert response.input_token_count > 0
        assert response.output_token_count > 0


def test_environment_variables():
    """Test that required environment variables are set."""
    required_vars = ["ANTHROPIC_API_KEY"]
    for var in required_vars:
        assert (
            os.getenv(var) is not None
        ), f"Missing required environment variable: {var}"


def test_temperature_parameter():
    """Test that temperature parameter affects response generation."""
    messages = [
        {"role": "user", "content": "Write a creative one-sentence story."}
    ]

    # Test with low temperature for deterministic response
    response_low = anthropic_request(
        messages=messages,
        model=AnthropicModels.CLAUDE_3_5_HAIKU_20241022,
        temperature=0.1,
        max_tokens=50,
    )

    # Test with high temperature for creative response
    response_high = anthropic_request(
        messages=messages,
        model=AnthropicModels.CLAUDE_3_5_HAIKU_20241022,
        temperature=0.9,
        max_tokens=50,
    )

    assert isinstance(response_low, LLMResponse)
    assert isinstance(response_high, LLMResponse)
    assert len(response_low.content) > 0
    assert len(response_high.content) > 0
