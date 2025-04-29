"""LLM interface for drift experiments."""

import logging
from typing import Dict, List

from src.api.azure_openai import azure_openai_request
from src.api.openai import openai_request
from src.api.openai import openai_request
from src.api.ollama import ollama_request

logger = logging.getLogger(__name__)


class LLMInterface:
    """Protocol defining the interface for LLM providers."""

    def __init__(self):
        """Initialize the LLM interface."""
        self.messages: List[Dict[str, str]] = []

    def generate(
        self,
        prompt: str,
        provider: str = "openai",
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.7,
        max_tokens: int = 100,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float = 1.0,
        swap_roles: bool = True,
    ) -> str:
        """Generate text from the given prompt.

        Args:
            prompt: The input prompt
            provider: Azure, Openai, Ollama, etc.
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            frequency_penalty: Penalty for frequency
            presence_penalty: Penalty for presence
            top_p: Top-p sampling parameter

        Returns:
            Generated text response
        """

        if swap_roles and self.messages:
            # Swap roles of previous messages
            logger.debug(
                "Swapping roles of previous messages."
                f" Role of last message: {self.messages[-1]['role']}"
            )
            for message in self.messages:
                message["role"] = (
                    "assistant" if message["role"] == "user" else "user"
                )

        # Add new prompt
        logger.debug(f"Adding new prompt: {prompt}")
        self.messages.append({"role": "user", "content": prompt})
        
        match provider:
            case "openai":
                f = openai_request
            case "ollama":
                f = ollama_request
            case "azure":
                f = azure_openai_request

        return f(
            messages=self.messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
        )
