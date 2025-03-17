"""Integration tests for Azure OpenAI API.

These tests make real API calls to Azure OpenAI to verify
functionality and API compatibility.
"""

import os

from src.api.azure_openai import azure_openai_request


def test_real_api_call_with_different_models_and_default_parameters():
    """Test real API calls with different available models."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, this is a test!'"},
    ]

    models = ["gpt-4o-2024-08-06"]

    for model in models:
        response = azure_openai_request(
            messages=messages,
            model=model,
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "test" in response.lower()


def test_environment_variables():
    """Test that required environment variables are set."""
    required_vars = ["AZURE_ENDPOINT", "AZURE_KEY"]
    for var in required_vars:
        assert (
            os.getenv(var) is not None
        ), f"Missing required environment variable: {var}"
