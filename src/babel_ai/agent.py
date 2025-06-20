"""Agent for generating responses using various LLM providers."""

from typing import Dict, List

from api.llm_interface import generate_response
from babel_ai.models import AgentConfig


class Agent:
    """An agent that generates responses using configured parameters."""

    def __init__(self, agent_config: AgentConfig):
        """Initialize the Agent with configuration.

        Args:
            agent_config: Configuration containing provider, model,
                         and generation parameters
        """
        self.config: AgentConfig = agent_config

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the agent.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                     keys representing the conversation history

        Returns:
            Generated response string
        """
        return generate_response(
            messages=messages,
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            top_p=self.config.top_p,
        )
