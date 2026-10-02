"""Unit tests for the vLLM API with mocked responses."""

from unittest.mock import MagicMock, patch

import pytest

from api.enums import VLLMModels
from api.vllm import LLAMA3_STOP_TOKEN_IDS, _resolve_model, vllm_request
from models.api import LLMResponse

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA3 = "meta-llama/Meta-Llama-3-70B-Instruct"


@pytest.fixture
def mock_vllm_response():
    """Create a mock vLLM chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="vLLM test response"))
    ]
    mock_response.usage.prompt_tokens = 25
    mock_response.usage.completion_tokens = 10
    return mock_response


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [{"role": "user", "content": "Hello"}]


@pytest.fixture
def mock_client(mock_vllm_response):
    """Create a mock OpenAI-compatible client."""
    client = MagicMock()
    client.chat.completions.create.return_value = mock_vllm_response
    return client


def test_vllm_request_returns_llm_response(
    mock_client, sample_messages, monkeypatch
):
    """Return content and token usage, resolving the model from env."""
    monkeypatch.setenv("VLLM_MODEL", QWEN)

    with patch("api.vllm.CLIENT", mock_client):
        response = vllm_request(
            messages=sample_messages,
            model=VLLMModels.DEFAULT,
            temperature=0.7,
            max_tokens=100,
        )

    assert isinstance(response, LLMResponse)
    assert response.content == "vLLM test response"
    assert response.input_token_count == 25
    assert response.output_token_count == 10
    mock_client.chat.completions.create.assert_called_once_with(
        model=QWEN,
        messages=sample_messages,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=1.0,
        max_tokens=100,
    )


def test_resolve_model_uses_explicit_member():
    """A non-default model member is sent verbatim."""
    explicit = MagicMock(value="some/custom-model")

    assert _resolve_model(explicit) == "some/custom-model"


def test_resolve_model_requires_vllm_model(monkeypatch):
    """The default sentinel needs ``VLLM_MODEL`` to be set."""
    monkeypatch.delenv("VLLM_MODEL", raising=False)

    with pytest.raises(ValueError, match="VLLM_MODEL"):
        _resolve_model(VLLMModels.DEFAULT)


def test_vllm_request_requires_base_url(monkeypatch, sample_messages):
    """Without an endpoint the request fails instead of hitting OpenAI."""
    monkeypatch.setenv("VLLM_MODEL", QWEN)

    with patch("api.vllm.CLIENT", None):
        with pytest.raises(ValueError, match="VLLM_BASE_URL"):
            vllm_request(messages=sample_messages)


def test_vllm_request_requires_model(
    mock_client, sample_messages, monkeypatch
):
    """The default sentinel without ``VLLM_MODEL`` raises."""
    monkeypatch.delenv("VLLM_MODEL", raising=False)

    with patch("api.vllm.CLIENT", mock_client):
        with pytest.raises(ValueError, match="VLLM_MODEL"):
            vllm_request(messages=sample_messages, model=VLLMModels.DEFAULT)


def test_vllm_request_adds_llama3_stop_token_ids(
    mock_client, sample_messages, monkeypatch
):
    """Llama 3 models carry explicit end-of-turn stop ids."""
    monkeypatch.setenv("VLLM_MODEL", LLAMA3)

    with patch("api.vllm.CLIENT", mock_client):
        vllm_request(messages=sample_messages, model=VLLMModels.DEFAULT)

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {
        "stop_token_ids": LLAMA3_STOP_TOKEN_IDS,
    }
    assert kwargs["model"] == LLAMA3


def test_vllm_request_omits_llama3_stop_token_ids(
    mock_client, sample_messages, monkeypatch
):
    """A non-Llama model gets no ``extra_body``."""
    monkeypatch.setenv("VLLM_MODEL", QWEN)

    with patch("api.vllm.CLIENT", mock_client):
        vllm_request(messages=sample_messages, model=VLLMModels.DEFAULT)

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in kwargs


def test_vllm_request_propagates_api_errors(mock_client, sample_messages):
    """A transport or server error is logged and re-raised."""
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    with patch("api.vllm.CLIENT", mock_client):
        with pytest.raises(Exception) as exc_info:
            vllm_request(
                messages=sample_messages,
                model=MagicMock(value=QWEN),
            )

    assert str(exc_info.value) == "API Error"
