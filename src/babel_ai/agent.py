"""Agent for generating responses using various LLM providers."""

import itertools
from typing import Dict, Generator, List

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


def round_robin_agent_selection(
    agents: List[Agent],
) -> Generator[Agent, None, None]:
    """Select the next agent in the round-robin sequence.

    Args:
        agents: List of agents to select from

    Yields:
        The next agent in the sequence, cycling infinitely
    """
    if not agents:
        raise ValueError("Cannot select from an empty list of agents")

    for agent in itertools.cycle(agents):
        yield agent
