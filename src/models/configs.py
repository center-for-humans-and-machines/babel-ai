"""Configuration models for drift experiments."""
import logging
from typing import List, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from api.enums import APIModels, Provider
from babel_ai.enums import AgentSelectionMethod, AnalyzerType, FetcherType

logger = logging.getLogger(__name__)


class AnalyzerConfig(BaseModel):
    """Configuration parameters for text analysis in drift experiments.

    This class defines how the analyzer should process and compare LLM
    responses to detect drift patterns. It controls the scope and methodology
    of the analysis performed on generated text.

    Attributes:
        analyzer (AnalyzerType): Type of analyzer to use for measuring drift
        analyze_window (int): Number of previous responses to include when
            calculating similarity metrics. Larger windows provide more
            context but may dilute recent drift signals. Must be ≥1.

    Example:
        >>> config = AnalyzerConfig(
        ...     analyzer=AnalyzerType.SIMILARITY,
        ...     analyze_window=5
        ... )
        >>> # This will compare each response against the 5 most recent ones
    """

    analyzer: AnalyzerType = Field(description="Type of analyzer to use")
    analyze_window: int = Field(
        description="Number of previous responses to analyze for drift",
        ge=1,
    )


class FetcherConfig(BaseModel):
    """Configuration parameters for conversation prompt fetching.

    This class defines how the prompt fetcher should select and prepare
    conversation data for drift experiments. It controls data source location
    and conversation filtering criteria.

    Attributes:
        fetcher (FetcherType): Type of fetcher to use for getting
            conversation prompts
        data_path (Optional[str]): File path to the conversation dataset.
            Should point to a valid data file in the expected format for the
            fetcher type. Required for SHAREGPT, INFINITE_CONVERSATION, and
            TOPICAL_CHAT fetcher types.
        second_data_path (Optional[str]): File path to the second conversation
            dataset. Only used by TOPICAL_CHAT fetcher type.
        category (Optional[str]): Category for random prompt generation.
            Only used by RANDOM fetcher type. Options: 'creative',
            'analytical', 'conversational'.
        min_messages (Optional[int]): Minimum number of messages required in a
            conversation for it to be selected. Filters out very short
            conversations. Required for SHAREGPT, INFINITE_CONVERSATION, and
            TOPICAL_CHAT fetcher types.
        max_messages (Optional[int]): Maximum number of messages to include
            from a conversation. Longer conversations will be truncated.
            Required for SHAREGPT, INFINITE_CONVERSATION, and TOPICAL_CHAT
            fetcher types.

    Example:
        >>> # For ShareGPT fetcher
        >>> config = FetcherConfig(
        ...     fetcher=FetcherType.SHAREGPT,
        ...     data_path="/data/conversations.json",
        ...     min_messages=3,
        ...     max_messages=20
        ... )
        >>> # For random fetcher
        >>> config = FetcherConfig(
        ...     fetcher=FetcherType.RANDOM,
        ...     category="creative"
        ... )
        >>> # For topical chat fetcher
        >>> config = FetcherConfig(
        ...     fetcher=FetcherType.TOPICAL_CHAT,
        ...     data_path="/data/test_rare.jsonl",
        ...     second_data_path="/data/test_freq.jsonl",
        ...     min_messages=2,
        ...     max_messages=10
        ... )
    """

    fetcher: FetcherType = Field(description="Type of fetcher to use")
    data_path: Optional[str] = Field(
        default=None, description="Path to the data file"
    )
    second_data_path: Optional[str] = Field(
        default=None,
        description="Path to the second data file (for TOPICAL_CHAT)",
    )
    category: Optional[str] = Field(
        default=None,
        description="Category for random prompts (for RANDOM fetcher)",
    )
    min_messages: Optional[int] = Field(
        default=None,
        description="Minimum number of messages in a conversation",
        ge=1,
    )
    max_messages: Optional[int] = Field(
        default=None,
        description="Maximum number of messages in a conversation",
        ge=1,
    )

    @field_validator("max_messages")
    def validate_max_messages(
        cls, v: Optional[int], values: ValidationInfo
    ) -> Optional[int]:
        """Validate that max_messages is >= min_messages."""
        if v is None:
            logger.debug("Initializing FetcherConfig with max_messages = None")
            return v

        min_messages = values.data.get("min_messages")
        if min_messages is not None and v < min_messages:
            raise ValueError(
                f"max_messages ({v}) must be >= min_messages ({min_messages})"
            )
        return v

    @field_validator("category")
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        """Validate that category is one of the allowed values."""
        if v is None:
            logger.debug("Initializing FetcherConfig with category = None")
            return v

        allowed_categories = ["creative", "analytical", "conversational"]
        if v not in allowed_categories:
            raise ValueError(
                f"category must be one of {allowed_categories}, got: {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_fetcher_parameters(self) -> "FetcherConfig":
        """Validate that the parameters match the fetcher type requirements."""
        fetcher_type = self.fetcher

        required_kwargs = fetcher_type.get_kwargs_mapping()
        for key, value in self.model_dump().items():
            if key in required_kwargs and value is None:
                raise ValueError(
                    f"Parameter {key} is required "
                    f"for fetcher type {fetcher_type}"
                )
            if key not in required_kwargs + ["fetcher"] and value is not None:
                raise ValueError(
                    f"Parameter {key} is not allowed "
                    f"for fetcher type {fetcher_type}"
                )

        return self


class AgentConfig(BaseModel):
    """Configuration for an LLM agent used in drift experiments.

    This class defines the parameters for configuring an agent that will
    generate responses during a drift experiment. It includes settings for
    the LLM provider, model selection, and various generation parameters
    that control the behavior and output characteristics of the agent.

    Attributes:
        provider (Provider): The LLM provider to use (e.g., OpenAI, Ollama)
        model (ModelType): The specific model to use from the provider
        system_prompt (Optional[str]): System prompt to guide the agent's
            behavior and responses. Default: None
        temperature (float): Controls randomness in generation (0.0-2.0).
            Higher values make output more random. Default: 0.7
        max_tokens (int): Maximum number of tokens to generate per response.
            Default: 100
        frequency_penalty (float): Penalty for frequently used tokens
            (-2.0 to 2.0). Positive values discourage repetition. Default: 0.0
        presence_penalty (float): Penalty for tokens that have already
            appeared (-2.0 to 2.0). Positive values encourage new topics.
            Default: 0.0
        top_p (float): Nucleus sampling parameter (0.0-1.0). Only consider
            tokens with cumulative probability up to top_p. Default: 1.0

    Example:
        >>> config = AgentConfig(
        ...     provider=Provider.OPENAI,
        ...     model=ModelType.GPT_3_5_TURBO,
        ...     system_prompt="You are a helpful assistant.",
        ...     temperature=0.8,
        ...     max_tokens=150,
        ...     frequency_penalty=0.1,
        ...     presence_penalty=0.1
        ... )
    """

    provider: Provider = Field(description="Name of the provider to use")
    model: APIModels = Field(description="Name of the model to use")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to guide agent behavior"
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("model")
    def validate_model_provider_compatibility(
        cls, v: APIModels, values: ValidationInfo
    ) -> APIModels:
        """Validate that the model is compatible with the provider."""
        provider = values.data["provider"]
        expected_model_enum = provider.get_model_enum()

        if not isinstance(v, expected_model_enum):
            raise ValueError(
                f"Model {v} is not compatible with provider {provider}. "
                f"Expected model type: {expected_model_enum.__name__}"
            )

        return v


class ExperimentConfig(BaseModel):
    """Configuration for running a drift experiment.

    This class defines the complete configuration for running a drift
    experiment, including fetcher settings, analyzer settings, agent
    configurations, and experiment parameters.

    Attributes:
        fetcher_config: Configuration parameters for the fetcher, including
            the fetcher type and its settings
        analyzer_config: Configuration parameters for the analyzer, including
            the analyzer type and its settings
        agent_configs: List of agent configurations to use in the experiment
        agent_selection_method: Method for selecting which agent responds next
        max_iterations: Maximum number of conversation turns to run
        max_total_characters: Maximum total characters across all responses
        output_dir: Directory to save results

    Example:
        >>> config = ExperimentConfig(
        ...     fetcher_config=FetcherConfig(
        ...         fetcher=FetcherType.SHAREGPT,
        ...         data_path="/path/to/sharegpt.json",
        ...         min_messages=2,
        ...         max_messages=10
        ...     ),
        ...     analyzer_config=AnalyzerConfig(
        ...         analyzer=AnalyzerType.SIMILARITY,
        ...         analyze_window=5
        ...     ),
        ...     agent_configs=[
        ...         AgentConfig(
        ...             provider=Provider.OPENAI,
        ...             model=ModelType.GPT_3_5_TURBO,
        ...             temperature=0.7
        ...         ),
        ...         AgentConfig(
        ...             provider=Provider.OLLAMA,
        ...             model=ModelType.LLAMA2_7B,
        ...             temperature=0.8
        ...         )
        ...     ],
        ...     agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        ...     max_iterations=50,
        ...     max_total_characters=500000,
        ...     output_dir="results",
        ... )
    """

    fetcher_config: FetcherConfig = Field(
        description="Configuration for the fetcher"
    )
    analyzer_config: AnalyzerConfig = Field(
        description="Configuration for the analyzer"
    )
    agent_configs: List[AgentConfig] = Field(
        description="Configurations for the agents"
    )
    agent_selection_method: AgentSelectionMethod = Field(
        description="Method to select the next agent"
    )
    max_iterations: int = Field(default=100, ge=1)
    max_total_characters: int = Field(default=1000000, ge=1)
    output_dir: Optional[str] = Field(
        default=None, description="Directory to save results"
    )
