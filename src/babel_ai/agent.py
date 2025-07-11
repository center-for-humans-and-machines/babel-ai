"""Agent for generating responses using various LLM providers."""

import itertools
import logging
import uuid
from typing import Dict, Generator, List

from api.llm_interface import generate_response
from models import AgentConfig

logger = logging.getLogger(__name__)


class Agent:
    """An agent that generates responses using configured parameters."""

    def __init__(self, agent_config: AgentConfig):
        """Initialize the Agent with configuration.

        Args:
            agent_config: Configuration containing provider, model,
                         and generation parameters
        """
        self.id = uuid.uuid4()
        self.provider = agent_config.provider
        self.model = agent_config.model
        self.system_prompt = agent_config.system_prompt
        self.config: AgentConfig = agent_config
        # TODO: Make this configurable
        # Currently, we only support conversational agents
        # If we change that, we need to think about
        # how to handle the system prompt with alternalting
        # agent types.
        self.is_conversational_agent = True

        logger.info(
            f"Agent {self.id} initialized "
            f"with provider {self.provider.value}, "
            f"model {self.model.value}, "
            f"and system prompt {self.system_prompt}"
        )
        logger.debug(f"Agent {self.id} config: {self.config.model_dump()}")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the agent.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                     keys representing the conversation history

        Returns:
            Generated response string

        Note:
            If a system_prompt is configured, it will be prepended as the
            first message with role 'system' to guide the agent's behavior.
        """
        # Prepare messages with system prompt if configured
        logger.info(f"Generating response for agent {self.id}")
        logger.debug(f"Agent {self.id} messages: {messages}")

        final_messages = []
        if self.system_prompt:
            logger.debug(
                f"Agent {self.id} adding system prompt to messages: "
                f"{self.system_prompt}"
            )
            final_messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        # TODO: Stub for later agent types
        if self.is_conversational_agent:
            logger.debug(
                f"Agent {self.id} is a conversational agent, "
                f"defining message tree."
            )
            message_tree = self._define_msg_tree(messages)
            final_messages.extend(message_tree)

        logger.debug(f"Agent {self.id} final messages: {final_messages}")

        return generate_response(
            messages=final_messages,
            provider=self.provider,
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            top_p=self.config.top_p,
        )

    @staticmethod
    def _define_msg_tree(
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Define a msg tree from the incoming messages,
        fitting the agents model type.
        """
        logger.debug("Defining message tree roles.")
        new_messages = []
        for i, message in enumerate(reversed(messages)):
            role = "user" if i % 2 == 0 else "assistant"
            new_messages.append({"role": role, "content": message["content"]})
        return reversed(new_messages)

    @staticmethod
    def _define_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Define a prompt from the incoming messages,
        fitting the agents model type.
        """
        logger.debug("Defining prompt.")
        raise NotImplementedError(
            """Not implemented we need to wait
            for Bram to implement base models."""
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
    logger.debug("Selecting next agent in round-robin sequence")
    logger.debug(f"Agents: {agents}")

    if not agents:
        raise ValueError("Cannot select from an empty list of agents")

    for agent in itertools.cycle(agents):
        logger.info(f"Yielding next agent in round-robin sequence: {agent.id}")
        yield agent
