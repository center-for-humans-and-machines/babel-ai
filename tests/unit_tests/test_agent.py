"""Tests for the Agent class."""

from unittest.mock import patch

import pytest

from api.llm_interface import Provider
from api.openai import OpenAIModel
from babel_ai.agent import Agent
from models import AgentConfig


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

        # Config attributes
        assert agent.config == sample_agent_config
        assert agent.config.provider == Provider.OPENAI
        assert agent.config.model == OpenAIModel.GPT4_1106_PREVIEW
        assert agent.config.system_prompt is None
        assert agent.config.temperature == 0.8
        assert agent.config.max_tokens == 150
        assert agent.config.frequency_penalty == 0.2
        assert agent.config.presence_penalty == 0.1
        assert agent.config.top_p == 0.9

        # Explicite attributes
        assert agent.id is not None
        assert agent.provider == Provider.OPENAI
        assert agent.model == OpenAIModel.GPT4_1106_PREVIEW
        assert agent.system_prompt is None

    def test_agent_initialization_with_system_prompt(self):
        """Test that Agent initializes correctly with system prompt."""
        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT4_1106_PREVIEW,
            system_prompt="You are a helpful assistant.",
            temperature=0.0,
        )

        agent = Agent(config)

        assert agent.system_prompt == "You are a helpful assistant."
        assert agent.config.system_prompt == "You are a helpful assistant."

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

    @patch("babel_ai.agent.generate_response")
    def test_generate_response_with_system_prompt(
        self, mock_generate_response
    ):
        """Test that system prompt is prepended to messages."""
        mock_generate_response.return_value = "Test response"

        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT4_1106_PREVIEW,
            system_prompt="You are a helpful assistant.",
            temperature=0.5,
        )

        agent = Agent(config)
        messages = [{"role": "user", "content": "Hello!"}]

        response = agent.generate_response(messages)

        # Verify the response
        assert response == "Test response"

        # Verify generate_response was called with system prompt prepended
        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        mock_generate_response.assert_called_once_with(
            messages=expected_messages,
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT4_1106_PREVIEW,
            temperature=0.5,
            max_tokens=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            top_p=1.0,
        )

    @patch("babel_ai.agent.generate_response")
    def test_generate_response_without_system_prompt(
        self, mock_generate_response
    ):
        """Test that messages are passed unchanged when no system prompt."""
        mock_generate_response.return_value = "Test response"

        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT4_1106_PREVIEW,
            temperature=0.5,
        )

        agent = Agent(config)
        messages = [{"role": "user", "content": "Hello!"}]

        response = agent.generate_response(messages)

        # Verify the response
        assert response == "Test response"

        # Verify generate_response was called with original messages unchanged
        mock_generate_response.assert_called_once_with(
            messages=messages,
            provider=Provider.OPENAI,
            model=OpenAIModel.GPT4_1106_PREVIEW,
            temperature=0.5,
            max_tokens=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            top_p=1.0,
        )

    def test_define_msg_tree_empty_list(self):
        """Test _define_msg_tree with empty list input."""
        result = Agent._define_msg_tree([])
        result_list = list(result)  # Convert iterator to list

        assert result_list == []

    def test_define_msg_tree_single_message(self):
        """Test _define_msg_tree with single message."""
        messages = [{"content": "Hello world"}]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        expected = [{"role": "user", "content": "Hello world"}]
        assert result_list == expected

    def test_define_msg_tree_four_messages(self):
        """Test _define_msg_tree with four messages."""
        messages = [
            {"content": "Message one"},
            {"content": "Message two"},
            {"content": "Message three"},
            {"content": "Message four"},
        ]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        expected = [
            {"role": "assistant", "content": "Message one"},
            {"role": "user", "content": "Message two"},
            {"role": "assistant", "content": "Message three"},
            {"role": "user", "content": "Message four"},
        ]
        assert result_list == expected

    def test_define_msg_tree_ignores_original_roles(self):
        """Test that _define_msg_tree ignores original role keys."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "system", "content": "Third message"},
        ]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        # Original roles should be ignored, new roles assigned based on
        # position
        expected = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]
        assert result_list == expected

    def test_define_msg_tree_with_extra_keys(self):
        """Test _define_msg_tree with messages containing extra keys."""
        messages = [
            {"content": "First", "timestamp": "2023-01-01", "id": 1},
            {"content": "Second", "metadata": {"key": "value"}},
        ]
        result = Agent._define_msg_tree(messages)
        result_list = list(result)  # Convert iterator to list

        # Should only have role and content keys
        for msg in result_list:
            assert set(msg.keys()) == {"role", "content"}

        assert result_list[0]["content"] == "First"
        assert result_list[1]["content"] == "Second"
