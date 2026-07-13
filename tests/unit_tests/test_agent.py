"""Tests for the Agent class."""

from unittest.mock import patch

import pytest

from agent import Agent
from api.enums import OpenAIModels, Provider
from models import AgentConfig


@pytest.fixture
def sample_agent_config():
    """Create a sample AgentConfig for testing."""
    return AgentConfig(
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
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
        assert agent.config.model == OpenAIModels.GPT4_1106_PREVIEW
        assert agent.config.system_prompt is None
        assert agent.config.temperature == 0.8
        assert agent.config.max_tokens == 150
        assert agent.config.frequency_penalty == 0.2
        assert agent.config.presence_penalty == 0.1
        assert agent.config.top_p == 0.9

        # Explicite attributes
        assert agent.id is not None
        assert agent.provider == Provider.OPENAI
        assert agent.model == OpenAIModels.GPT4_1106_PREVIEW
        assert agent.system_prompt is None

    def test_agent_initialization_with_system_prompt(self):
        """Test that Agent initializes correctly with system prompt."""
        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            system_prompt="You are a helpful assistant.",
            temperature=0.0,
        )

        agent = Agent(config)

        assert agent.system_prompt == "You are a helpful assistant."
        assert agent.config.system_prompt == "You are a helpful assistant."

    @patch("agent.LLMInterface.generate_response")
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
            model=OpenAIModels.GPT4_1106_PREVIEW,
            temperature=0.8,
            max_tokens=150,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            top_p=0.9,
        )

    @patch("agent.LLMInterface.generate_response")
    def test_generate_response_with_system_prompt(
        self, mock_generate_response
    ):
        """Test generate_response with system prompt."""
        mock_generate_response.return_value = "Test response"

        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
        )
        agent = Agent(config)
        messages = [{"role": "user", "content": "Hello"}]

        response = agent.generate_response(messages)

        # Verify response
        assert response == "Test response"

        # Verify generate_response was called with system prompt prepended
        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        mock_generate_response.assert_called_once_with(
            messages=expected_messages,
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            temperature=0.7,
            max_tokens=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            top_p=1.0,
        )

    @patch("agent.LLMInterface.generate_response")
    def test_generate_response_without_system_prompt(
        self, mock_generate_response
    ):
        """Test generate_response without system prompt."""
        mock_generate_response.return_value = "Test response"

        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            # No system_prompt
            temperature=0.7,
        )
        agent = Agent(config)
        messages = [{"role": "user", "content": "Hello"}]

        response = agent.generate_response(messages)

        # Verify response
        assert response == "Test response"

        # Verify generate_response was called with original messages unchanged
        mock_generate_response.assert_called_once_with(
            messages=messages,  # No system prompt added
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            temperature=0.7,
            max_tokens=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            top_p=1.0,
        )

    def test_define_msg_tree_empty_list(self, sample_agent_config):
        """Test _define_msg_tree with empty list input."""
        agent = Agent(sample_agent_config)
        result = agent._define_msg_tree([])

        assert result == []

    def test_define_msg_tree_single_message(self, sample_agent_config):
        """Test _define_msg_tree with single message."""
        agent = Agent(sample_agent_config)
        messages = [{"content": "Hello world"}]
        result = agent._define_msg_tree(messages)

        expected = [{"role": "user", "content": "Hello world"}]
        assert result == expected

    def test_define_msg_tree_four_messages(self, sample_agent_config):
        """Test _define_msg_tree with four messages."""
        agent = Agent(sample_agent_config)
        messages = [
            {"content": "Message one"},
            {"content": "Message two"},
            {"content": "Message three"},
            {"content": "Message four"},
        ]
        result = agent._define_msg_tree(messages)

        expected = [
            {"role": "assistant", "content": "Message one"},
            {"role": "user", "content": "Message two"},
            {"role": "assistant", "content": "Message three"},
            {"role": "user", "content": "Message four"},
        ]
        assert result == expected

    def test_define_msg_tree_ignores_original_roles(self, sample_agent_config):
        """Test that _define_msg_tree ignores original role keys."""
        agent = Agent(sample_agent_config)
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "system", "content": "Third message"},
        ]
        result = agent._define_msg_tree(messages)

        # Original roles should be ignored, new roles assigned based on
        # position
        expected = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
        ]
        assert result == expected

    def test_define_msg_tree_with_extra_keys(self, sample_agent_config):
        """Test _define_msg_tree with messages containing extra keys."""
        agent = Agent(sample_agent_config)
        messages = [
            {"content": "First", "timestamp": "2023-01-01", "id": 1},
            {"content": "Second", "metadata": {"key": "value"}},
        ]
        result = agent._define_msg_tree(messages)

        # Should only have role and content keys
        for msg in result:
            assert set(msg.keys()) == {"role", "content"}

        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"

    def test_define_msg_tree_with_forgetting(self):
        """Test _define_msg_tree with forgetting configured."""
        config = AgentConfig(
            provider=Provider.OPENAI,
            model=OpenAIModels.GPT4_1106_PREVIEW,
            forgetting=2,
        )
        agent = Agent(config)
        messages = [
            {"content": "Message one"},
            {"content": "Message two"},
            {"content": "Message three"},
            {"content": "Message four"},
        ]
        result = agent._define_msg_tree(messages)

        # Should only use last 2 messages
        expected = [
            {"role": "assistant", "content": "Message three"},
            {"role": "user", "content": "Message four"},
        ]
        assert result == expected

    def test_define_msg_tree_with_forgetting_none(self, sample_agent_config):
        """Test _define_msg_tree with forgetting=None uses all messages."""
        agent = Agent(sample_agent_config)
        messages = [
            {"content": "Message one"},
            {"content": "Message two"},
            {"content": "Message three"},
        ]
        result = agent._define_msg_tree(messages)

        # Should use all messages when forgetting is None
        expected = [
            {"role": "user", "content": "Message one"},
            {"role": "assistant", "content": "Message two"},
            {"role": "user", "content": "Message three"},
        ]
        assert result == expected
