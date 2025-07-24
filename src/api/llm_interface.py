"""LLM interface for drift experiments."""

import logging
import time
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from api.budget import BudgetTracker
from api.enums import APIModels, Provider

logger = logging.getLogger(__name__)


class LLMInterface:
    """Singleton LLM interface."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def generate_response(
        messages: List[Dict[str, str]],
        provider: Provider,
        model: APIModels,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float = 1.0,
        # TODO: Configurability is missing here.
        max_retries: int = 3,
        initial_delay: float = 3.0,
    ) -> str:
        """Generate text response using the specified provider and model.

        Args:
            messages: List of messages in the conversation
            provider: The LLM provider to use
            model: The model to use (must be a valid model for the provider)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            frequency_penalty: Penalty for frequency
            presence_penalty: Penalty for presence
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before retrying

        Returns:
            Generated text response string

        Raises:
            Exception: If all retry attempts fail
        """
        # Generate a unique request ID
        request_id = str(uuid4())

        request_function = provider.get_request_function()

        # Collect errors for final exception message
        errors = []

        # Try to process request for max_retries
        for attempt in range(max_retries):
            try:
                # Log attempt number
                logger.info(
                    f"Request ID: {request_id}, "
                    f"Attempt {attempt + 1} of {max_retries} for "
                    f"provider: {provider.value}, model: {model.value}"
                )

                # Make the request
                llm_response = request_function(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    top_p=top_p,
                )

                # Log token usage
                logger.debug(
                    f"Request ID: {request_id}, "
                    f"Token usage - Input: {llm_response.input_token_count}, "
                    f"Output: {llm_response.output_token_count}"
                )

                # Track budget usage
                budget_tracker = BudgetTracker()
                usage_summary = budget_tracker.add_usage(
                    provider=provider,
                    model=model,
                    input_tokens=llm_response.input_token_count,
                    output_tokens=llm_response.output_token_count,
                )

                # Log budget information
                logger.info(
                    f"Request ID: {request_id}, "
                    f"Budget - Cost: ${usage_summary['total_cost']:.6f}, "
                    f"Cumulative: ${usage_summary['cumulative_total_cost']:.6f}"  # noqa: E501
                )

                # If successful, return the content string
                logger.info(
                    f"Request ID: {request_id}, "
                    f"Successfully generated response on attempt "
                    f"{attempt + 1}"
                )
                return llm_response.content

            except Exception as e:
                # Log error and prepare for next attempt
                error_msg = (
                    f"Request ID: {request_id}, "
                    f"Attempt {attempt + 1} failed: {str(e)}"
                )
                logger.error(error_msg)
                errors.append(error_msg)

                # If this was the last attempt, don't wait
                if attempt == max_retries - 1:
                    break

                # Wait for next attempt with exponential backoff
                delay = initial_delay**attempt
                logger.info(
                    f"Request ID: {request_id}, "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        # If max retries reached, raise exception with all errors
        error_summary = (
            f"Request ID: {request_id}, "
            f"Max retries ({max_retries}) reached for provider: "
            f"{provider}, model: {model}. Errors: " + " | ".join(errors)
        )
        logger.error(error_summary)
        raise Exception(error_summary)
