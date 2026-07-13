"""Integration tests for OpenAI API."""

import os

from api.enums import OpenAIModels
from api.openai import openai_request
from models.api import LLMResponse


def test_real_api_call_with_old_and_new_models_default_parameters():
    """Test real API calls with an old and a newer OpenAI model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, this is a test!'"},
    ]

    models = [
        OpenAIModels.GPT4_1106_PREVIEW,  # old-style parameters
        OpenAIModels.GPT5_MINI_2025_08_07,  # newer parameter style
    ]

    for model in models:
        response = openai_request(
            messages=messages,
            model=model,
        )

        assert response is not None
        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert "test" in response.content.lower()
        assert response.input_token_count > 0
        assert response.output_token_count > 0


def test_environment_variables_present_for_openai():
    """Test that required environment variable is set for OpenAI."""
    required_vars = ["OPENAI_API_KEY"]
    for var in required_vars:
        assert (
            os.getenv(var) is not None
        ), f"Missing required environment variable: {var}"
