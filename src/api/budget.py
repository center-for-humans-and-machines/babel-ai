"""Budget tracking system for LLM API usage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from api.enums import (
    AnthropicModels,
    AzureModels,
    OllamaModels,
    OpenAIModels,
    Provider,
)

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Singleton class for tracking LLM API budget usage.

    Tracks token usage and costs per provider/model combination
    and persists data to file for permanent storage.
    """

    _instance = None
    _lock = Lock()

    # Pricing per 1K tokens (input/output) in USD
    PRICING = {
        Provider.OPENAI: {
            OpenAIModels.GPT4_1106_PREVIEW: {
                # $0.001 per 1K/$10 per 1M input tokens
                "input": 0.001,
                # $0.003 per 1K/$30.00 per 1M output tokens
                "output": 0.003,
            },
            OpenAIModels.GPT4_0125_PREVIEW: {
                # $0.001 per 1K/$10 per 1M input tokens
                "input": 0.001,
                # $0.003 per 1K/$30.00 per 1M output tokens
                "output": 0.003,
            },
            OpenAIModels.O3_2025_04_16: {
                # $0.002 per 1K/$2.00 per 1M input tokens
                "input": 0.002,
                # $0.008 per 1K/$8.00 per 1M output tokens
                "output": 0.008,
            },
            OpenAIModels.O4_MINI_2025_04_16: {
                # $0.0011 per 1K/$1.10 per 1M input tokens
                "input": 0.0011,
                # $0.0044 per 1K/$4.40 per 1M output tokens
                "output": 0.0044,
            },
        },
        Provider.AZURE: {
            AzureModels.GPT4O_2024_08_06: {
                # €0.00234883 per 1K/€2.34883 per 1M input tokens
                "input": 0.00234883,
                # €0.0093953 per 1K/€9.3953 per 1M output tokens
                "output": 0.0093953,
            },
            AzureModels.O3_2025_04_16: {
                # $0.00171 per 1K/$1.71 per 1M input tokens
                "input": 0.00171,
                # $0.00684 per 1K/$6.84 per 1M output tokens
                "output": 0.00684,
            },
            AzureModels.O4_MINI_2025_04_16: {
                # $0.00094 per 1K/$0.94 per 1M input tokens
                "input": 0.00094,
                # $0.00376 per 1K/$3.76 per 1M output tokens
                "output": 0.00376,
            },
        },
        Provider.OLLAMA: {
            # Local models - no cost
            OllamaModels.MISTRAL_7B: {"input": 0.0, "output": 0.0},
            OllamaModels.MISTRAL_7B_TEXT: {"input": 0.0, "output": 0.0},
            OllamaModels.LLAMA3_70B: {"input": 0.0, "output": 0.0},
            OllamaModels.LLAMA3_70B_TEXT: {"input": 0.0, "output": 0.0},
            OllamaModels.DEEPSEEK_R1: {"input": 0.0, "output": 0.0},
            OllamaModels.GPT_2_1_5B: {"input": 0.0, "output": 0.0},
        },
        Provider.RAVEN: {
            # Uses Ollama models - no cost
            OllamaModels.MISTRAL_7B: {"input": 0.0, "output": 0.0},
            OllamaModels.MISTRAL_7B_TEXT: {"input": 0.0, "output": 0.0},
            OllamaModels.LLAMA3_70B: {"input": 0.0, "output": 0.0},
            OllamaModels.LLAMA3_70B_TEXT: {"input": 0.0, "output": 0.0},
            OllamaModels.DEEPSEEK_R1: {"input": 0.0, "output": 0.0},
            OllamaModels.GPT_2_1_5B: {"input": 0.0, "output": 0.0},
        },
        Provider.ANTHROPIC: {
            AnthropicModels.CLAUDE_SONNET_4_20250514: {
                # $0.003 per 1K/$3.00 per 1M input tokens
                "input": 0.003,
                # $0.015 per 1K/$15.00 per 1M output tokens
                "output": 0.015,
            },
            AnthropicModels.CLAUDE_OPUS_4_20250514: {
                # $0.015 per 1K/$15.00 per 1M input tokens
                "input": 0.015,
                # $0.075 per 1K/$75.00 per 1M output tokens
                "output": 0.075,
            },
            AnthropicModels.CLAUDE_3_5_HAIKU_20241022: {
                # $0.0008 per 1K/$0.80 per 1M input tokens
                "input": 0.04,
                # $0.004 per 1K/$4.00 per 1M output tokens
                "output": 0.004,
            },
        },
    }

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize budget tracker."""
        if self._initialized:
            return

        self.budget_file = Path("logs/budget_tracking.json")
        self.budget_file.parent.mkdir(exist_ok=True)

        # Initialize budget data structure
        self.budget_data = {
            "total_cost": 0.0,
            "providers": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

        # Load existing data if available
        self._load_budget_data()
        self._initialized = True

        logger.info(
            f"BudgetTracker initialized, data stored at: "
            f"{self.budget_file.absolute()}"
        )

    def _load_budget_data(self) -> None:
        """Load existing budget data from file."""
        if self.budget_file.exists():
            try:
                with open(self.budget_file, "r") as f:
                    loaded_data = json.load(f)
                # Merge with default structure to handle schema updates
                self.budget_data.update(loaded_data)
                logger.info("Loaded existing budget data from file")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not load budget data: {e}. "
                    f"Starting with fresh data."
                )

    def _save_budget_data(self) -> None:
        """Save current budget data to file."""
        try:
            with open(self.budget_file, "w") as f:
                json.dump(self.budget_data, f, indent=2, ensure_ascii=False)
            logger.debug("Budget data saved to file")
        except IOError as e:
            logger.error(f"Failed to save budget data: {e}")

    def _get_cost_per_1k_tokens(
        self, provider: Provider, model: Any, token_type: str
    ) -> float:
        """Get cost per 1K tokens for a specific provider/model/type.

        Args:
            provider: The LLM provider
            model: The model (enum value)
            token_type: Either "input" or "output"

        Returns:
            Cost per 1K tokens in USD
        """
        try:
            return self.PRICING[provider][model][token_type]
        except KeyError:
            logger.warning(
                f"No pricing data for {provider.value}/"
                f"{model.value}/{token_type}"
            )
            return 0.0

    def add_usage(
        self,
        provider: Provider,
        model: Any,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, Any]:
        """Add token usage and calculate costs.

        Args:
            provider: The LLM provider used
            model: The model used (OpenAIModels, OllamaModels, or AzureModels)
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated

        Returns:
            Dictionary with cost breakdown for this usage
        """
        with self._lock:
            # Calculate costs
            input_cost_per_1k = self._get_cost_per_1k_tokens(
                provider, model, "input"
            )
            output_cost_per_1k = self._get_cost_per_1k_tokens(
                provider, model, "output"
            )

            input_cost = (input_tokens / 1000.0) * input_cost_per_1k
            output_cost = (output_tokens / 1000.0) * output_cost_per_1k
            total_cost = input_cost + output_cost

            # Initialize provider data if needed
            provider_key = provider.value
            if provider_key not in self.budget_data["providers"]:
                self.budget_data["providers"][provider_key] = {
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "models": {},
                }

            # Initialize model data if needed
            model_key = model.value
            provider_data = self.budget_data["providers"][provider_key]
            if model_key not in provider_data["models"]:
                provider_data["models"][model_key] = {
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "usage_count": 0,
                }

            # Update totals
            model_data = provider_data["models"][model_key]
            model_data["total_cost"] += total_cost
            model_data["total_input_tokens"] += input_tokens
            model_data["total_output_tokens"] += output_tokens
            model_data["usage_count"] += 1

            provider_data["total_cost"] += total_cost
            provider_data["total_input_tokens"] += input_tokens
            provider_data["total_output_tokens"] += output_tokens

            self.budget_data["total_cost"] += total_cost
            self.budget_data["last_updated"] = datetime.now().isoformat()

            # Save to file
            self._save_budget_data()

            # Prepare return data
            usage_summary = {
                "provider": provider.value,
                "model": model.value,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
                "cumulative_total_cost": round(
                    self.budget_data["total_cost"], 6
                ),
            }

            logger.info(f"Added usage: {usage_summary}")
            return usage_summary
