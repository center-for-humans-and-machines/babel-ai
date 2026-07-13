"""Unit tests for the vLLM API with mocked responses."""

from unittest.mock import MagicMock, patch

from api.enums import VLLMModels
from api.vllm import vllm_request
from models.api import LLMResponse


def test_vllm_request_returns_llm_response():
    """Return content and token usage from vLLM."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="vLLM test response"))
    ]
    mock_response.usage.prompt_tokens = 25
    mock_response.usage.completion_tokens = 10
    messages = [{"role": "user", "content": "Hello"}]

    with patch(
        "api.vllm.CLIENT.chat.completions.create",
        return_value=mock_response,
    ) as mock_create:
        response = vllm_request(
            messages=messages,
            model=VLLMModels.DEFAULT,
            temperature=0.7,
            max_tokens=100,
        )

    assert isinstance(response, LLMResponse)
    assert response.content == "vLLM test response"
    assert response.input_token_count == 25
    assert response.output_token_count == 10
    mock_create.assert_called_once_with(
        model="default",
        messages=messages,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=1.0,
        max_tokens=100,
    )
