"""Tests for the Agent class."""

from unittest.mock import patch

import pytest

from api.llm_interface import Provider
from api.openai import OpenAIModel
from babel_ai.agent import Agent
from babel_ai.models import AgentConfig


@pytest.fixture
def sample_agent_config():
    """Create a sample AgentConfig for testing."""
    return AgentConfig(
        provider=Provider.OPENAI,
        model=OpenAIModel.GPT4_1106_PREVIEW,
        temperature=0.8,
        max_tokens=150,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        top_p=0.9,
    )


@pytest.fixture
def sample_messages():
    """Create sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


class TestAgent:
    """Test the Agent class."""

    def test_agent_initialization(self, sample_agent_config):
        """Test that Agent initializes correctly with AgentConfig."""
        agent = Agent(sample_agent_config)

        assert agent.config == sample_agent_config
        assert agent.config.provider == Provider.OPENAI
        assert agent.config.model == OpenAIModel.GPT4_1106_PREVIEW
        assert agent.config.temperature == 0.8
        assert agent.config.max_tokens == 150
        assert agent.config.frequency_penalty == 0.2
        assert agent.config.presence_penalty == 0.1
        assert agent.config.top_p == 0.9

    @patch("babel_ai.agent.generate_response")
    def test_generate_response_calls_api(
        self, mock_generate_response, sample_agent_config, sample_messages
    ):
        """Test that generate_response calls
        the API with correct parameters."""
        mock_generate_response.return_value = "Test response from API"

        agent = Agent(sample_agent_config)
        response = agent.generate_response(sample_messages)

        # Verify the response
        assert response == "Test response from API"

        # Verify generate_response was called with correct parameters
        mock_generate_response.assert_called_once_with(
            messages=sample_messages,
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT4_1106_PREVIEW,
            temperature=0.8,
            max_tokens=150,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            top_p=0.9,
        )
