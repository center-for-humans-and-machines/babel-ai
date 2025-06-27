"""Pydantic models for analysis results."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from api.llm_interface import ModelType, Provider
from babel_ai.enums import AgentSelectionMethod, AnalyzerType, FetcherType


class AnalysisResult(BaseModel):
    """Comprehensive analysis results for LLM-generated text outputs.

    This class contains various metrics computed from analyzing LLM responses
    to measure text quality, coherence, similarity, and complexity. It's used
    in drift experiments to track how model outputs change over time.

    Attributes:
        word_count (int): Total number of words in the analyzed text
        unique_word_count (int): Number of unique/distinct words used
        coherence_score (float): Ratio of unique words to total words
            (0.0-1.0). Higher values indicate more diverse vocabulary usage
        lexical_similarity (Optional[float]): Jaccard similarity coefficient
            between current and previous text (0.0-1.0). Measures word overlap
        semantic_similarity (Optional[float]): Cosine similarity between
            sentence embeddings (-1.0 to 1.0). Measures meaning similarity
        lexical_similarity_window (Optional[float]): Average Jaccard similarity
            across analysis window (0.0-1.0)
        semantic_similarity_window (Optional[float]): Average cosine similarity
            across analysis window (-1.0 to 1.0)
        token_perplexity (Optional[float]): Average perplexity of tokens
            (≥1.0). Lower values indicate more predictable/coherent text

    Note:
        Optional fields (lexical_similarity, semantic_similarity,
        lexical_similarity_window, semantic_similarity_window,
        token_perplexity) may be None if comparison data is unavailable
        or if the analysis couldn't be performed.

    Example:
        >>> result = AnalysisResult(
        ...     word_count=45,
        ...     unique_word_count=38,
        ...     coherence_score=0.844,
        ...     lexical_similarity=0.23,
        ...     semantic_similarity=0.78,
        ...     lexical_similarity_window=0.3,
        ...     semantic_similarity_window=0.8,
        ...     token_perplexity=12.5
        ... )
    """

    word_count: int = Field(description="Total number of words in the text")
    unique_word_count: int = Field(description="Number of unique words")
    coherence_score: float = Field(
        description="Ratio of unique words to total words", ge=0.0, le=1.0
    )
    lexical_similarity: Optional[float] = Field(
        None,
        description="Jaccard similarity between current and previous text",
        ge=0.0,
        le=1.0,
    )
    semantic_similarity: Optional[float] = Field(
        None,
        description="Cosine similarity between sentence embeddings",
        ge=-1.0,
        le=1.0,
    )
    lexical_similarity_window: Optional[float] = Field(
        None,
        description="Average Jaccard similarity across analysis window",
        ge=0.0,
        le=1.0,
    )
    semantic_similarity_window: Optional[float] = Field(
        None,
        description="Average cosine similarity across analysis window",
        ge=-1.0,
        le=1.0,
    )
    token_perplexity: Optional[float] = Field(
        None,
        description="Average perplexity of all tokens in the text",
        ge=1.0,
    )


class AnalyzerConfig(BaseModel):
    """Configuration parameters for text analysis in drift experiments.

    This class defines how the analyzer should process and compare LLM
    responses to detect drift patterns. It controls the scope and methodology
    of the analysis performed on generated text.

    Attributes:
        analyze_window (int): Number of previous responses to include when
            calculating similarity metrics. Larger windows provide more
            context but may dilute recent drift signals. Must be ≥1.

    Example:
        >>> config = AnalyzerConfig(analyze_window=5)
        >>> # This will compare each response against the 5 most recent ones
    """

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
        data_path (str): File path to the conversation dataset. Should point
            to a valid data file in the expected format for the fetcher type
        min_messages (int): Minimum number of messages required in a
            conversation for it to be selected. Filters out very short
            conversations
        max_messages (int): Maximum number of messages to include from a
            conversation. Longer conversations will be truncated

    Example:
        >>> config = FetcherConfig(
        ...     data_path="/data/conversations.json",
        ...     min_messages=3,
        ...     max_messages=20
        ... )
        >>> # Will load conversations with 3-20 messages from the JSON file
    """

    data_path: str = Field(description="Path to the data file")
    min_messages: int = Field(
        description="Minimum number of messages in a conversation",
        ge=1,
    )
    max_messages: int = Field(
        description="Maximum number of messages in a conversation",
        ge=1,
    )

    @field_validator("max_messages")
    def validate_max_messages(cls, v: int, values: ValidationInfo) -> int:
        """Validate that max_messages is >= min_messages."""
        if "min_messages" not in values.data.keys():
            raise ValueError(
                "min_messages is required to validate max_messages"
            )
        min_messages = values.data["min_messages"]
        if v < min_messages:
            raise ValueError(
                f"max_messages ({v}) must be >= min_messages ({min_messages})"
            )
        return v


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
    model: ModelType = Field(description="Name of the model to use")
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
        cls, v: ModelType, values: ValidationInfo
    ) -> ModelType:
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
        fetcher: Type of fetcher to use for getting conversation prompts
        fetcher_config: Dictionary of configuration parameters for the fetcher
        analyzer: Type of analyzer to use for measuring drift
        analyzer_config: Configuration parameters for the analyzer
        agent_configs: List of agent configurations to use in the experiment
        agent_selection_method: Method for selecting which agent responds next
        max_iterations: Maximum number of conversation turns to run
        max_total_characters: Maximum total characters across all responses
        output_dir: Directory to save results

    Example:
        >>> config = ExperimentConfig(
        ...     fetcher=FetcherType.SHAREGPT,
        ...     fetcher_config={
        ...         "data_path": "/path/to/sharegpt.json",
        ...         "min_messages": 2,
        ...         "max_messages": 10
        ...     },
        ...     analyzer=AnalyzerType.SIMILARITY,
        ...     analyzer_config=AnalyzerConfig(analyze_window=5),
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

    fetcher: FetcherType = Field(description="Type of fetcher to use")
    fetcher_config: FetcherConfig = Field(
        description="Configuration for the fetcher"
    )
    analyzer: AnalyzerType = Field(description="Type of analyzer to use")
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


class Metric(BaseModel):
    """A single data point collected during a drift experiment.

    This class represents one measurement taken during an experiment,
    containing both the raw response data and its computed analysis metrics.
    It serves as the fundamental unit of data collection for tracking model
    drift over time.

    Attributes:
        iteration (int): Sequential number of this measurement in the
            experiment. Starts from 0 and increments with each new response
        timestamp (datetime): Exact time when this metric was recorded.
            Used for temporal analysis and experiment tracking
        role (str): The role/type of the message in the conversation
            (typically "system", "user", or "assistant")
        response (str): The actual text content generated by the LLM.
            This is the raw output being analyzed for drift
        analysis (AnalysisResult): Computed metrics for this response
            including word counts, similarity scores, and perplexity measures
        config (Optional[ExperimentConfig]): The experiment configuration
            used when this metric was collected. May be None for space
            efficiency

    Example:
        >>> metric = Metric(
        ...     iteration=42,
        ...     timestamp=datetime.now(),
        ...     role="assistant",
        ...     response="This is a sample response from the model.",
        ...     analysis=AnalysisResult(
        ...         word_count=9,
        ...         unique_word_count=9,
        ...         coherence_score=1.0
        ...     )
        ... )
    """

    iteration: int = Field(description="Iteration number in the experiment")
    timestamp: datetime = Field(description="When this metric was recorded")
    role: str = Field(
        description="Role of the message (system/user/assistant)"
    )
    content: str = Field(description="The actual text response")
    analysis: Optional[AnalysisResult] = Field(
        None, description="Analysis results for this response"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to a dictionary format suitable for CSV export.

        Returns:
            Dictionary with flattened structure for CSV export
        """
        result = {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "analysis": self.analysis.model_dump() if self.analysis else None,
        }

        return result


class FetcherMetric(Metric):
    """A single data point collected during a drift experiment."""

    fetcher_type: FetcherType = Field(
        description="Type of fetcher that generated the response"
    )
    fetcher_config: FetcherConfig = Field(
        description="Configuration of the fetcher that generated the response"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to a dictionary format suitable for CSV export."""
        result = super().to_dict()
        result.update(
            {
                "fetcher_type": self.fetcher_type,
                "fetcher_config": self.fetcher_config.model_dump(),
            }
        )

        return result


class AgentMetric(Metric):
    """A single data point collected during a drift experiment."""

    agent_id: str = Field(
        description="ID of the agent that generated the response"
    )
    agent_config: AgentConfig = Field(
        description="Configuration of the agent that generated the response"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to a dictionary format suitable for CSV export."""
        result = super().to_dict()
        result.update(
            {
                "agent_id": self.agent_id,
                "agent_config": self.agent_config.model_dump(),
            }
        )

        return result


class ExperimentMetadata(BaseModel):
    """Metadata container for LLM drift experiments.

    This class stores comprehensive metadata about an experiment run,
    including timing information, configuration details, and execution
    statistics. It provides a structured way to track and persist
    experiment information for analysis and reproducibility.

    Attributes:
        timestamp: When the experiment was initiated
        config: Complete experiment configuration used
        num_iterations_total: Total number of iterations completed
            (includes both fetcher and agent-generated messages)
        num_fetcher_messages: Number of initial messages from fetcher
        total_characters: Total character count across all messages

    Example:
        >>> config = ExperimentConfig(
        ...     max_iterations=100,
        ...     max_total_characters=10000,
        ...     # ... other config fields
        ... )
        >>>
        >>> metadata = ExperimentMetadata(
        ...     timestamp=datetime.now(),
        ...     config=config,
        ...     num_iterations_total=87,
        ...     num_fetcher_messages=5,
        ...     total_characters=8543
        ... )
    """

    timestamp: datetime
    config: ExperimentConfig
    num_iterations_total: Optional[int] = None
    num_fetcher_messages: Optional[int] = None
    total_characters: Optional[int] = None
