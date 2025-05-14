"""Pydantic models for analysis results."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class WordStats(BaseModel):
    """Statistics about word usage in text."""

    word_count: int = Field(description="Total number of words in the text")
    unique_word_count: int = Field(description="Number of unique words")
    coherence_score: float = Field(
        description="Ratio of unique words to total words", ge=0.0, le=1.0
    )


class LexicalMetrics(BaseModel):
    """Metrics for lexical similarity analysis."""

    similarity: Optional[float] = Field(
        None,
        description="Jaccard similarity between current and previous text",
        ge=0.0,
        le=1.0,
    )
    is_repetitive: bool = Field(
        default=False, description="Whether the text is lexically repetitive"
    )


class SemanticMetrics(BaseModel):
    """Metrics for semantic similarity analysis."""

    similarity: Optional[float] = Field(
        None,
        description="Cosine similarity between sentence embeddings",
        ge=-1.0,
        le=1.0,
    )
    is_repetitive: bool = Field(
        default=False,
        description="Whether the text is semantically repetitive",
    )


class SurpriseMetrics(BaseModel):
    """Metrics for semantic surprise analysis."""

    semantic_surprise: float = Field(
        description="Average KL divergence from previous texts", ge=0.0
    )
    max_semantic_surprise: float = Field(
        description="Maximum KL divergence from previous texts", ge=0.0
    )
    is_surprising: bool = Field(
        description="Whether the text is considered surprising"
    )


class TokenLikelihoodMetrics(BaseModel):
    """Metrics for token likelihood analysis."""

    avg_token_likelihood: float = Field(
        description="Average likelihood of all tokens in the text",
        ge=0.0,
        le=1.0,
    )
    context_avg_likelihood: Optional[float] = Field(
        description="Average likelihood when considering previous context",
        ge=0.0,
        le=1.0,
        default=None,
    )


class AnalysisResult(BaseModel):
    """Combined analysis results for text outputs."""

    word_stats: WordStats
    lexical: Optional[LexicalMetrics] = None
    semantic: Optional[SemanticMetrics] = None
    surprise: Optional[SurpriseMetrics] = None
    token_likelihood: Optional[TokenLikelihoodMetrics] = None


class ExperimentConfig(BaseModel):
    """Configuration for the drift experiment."""

    provider: str = (Field(description="Name of the provider to use"),)
    model: str = (Field(description="Name of the model to use"),)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=100, ge=1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_iterations: int = Field(default=100, ge=1)
    max_total_characters: int = Field(default=1000000, ge=1)
    analyze_window: int = Field(default=20, ge=1)


class Metric(BaseModel):
    """A single metric entry from a drift experiment."""

    iteration: int = Field(description="Iteration number in the experiment")
    timestamp: datetime = Field(description="When this metric was recorded")
    role: str = Field(
        description="Role of the message (system/user/assistant)"
    )
    response: str = Field(description="The actual text response")
    analysis: AnalysisResult = Field(
        description="Analysis results for this response"
    )
    config: Optional[ExperimentConfig] = Field(
        default=None, description="Configuration used for this experiment"
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
            "response": self.response,
            # Word stats
            "word_count": self.analysis.word_stats.word_count,
            "unique_word_count": self.analysis.word_stats.unique_word_count,
            "coherence_score": self.analysis.word_stats.coherence_score,
        }

        # Add lexical metrics if available
        if self.analysis.lexical:
            result.update(
                {
                    "lexical_similarity": self.analysis.lexical.similarity,
                    "is_repetitive": self.analysis.lexical.is_repetitive,
                }
            )

        # Add semantic metrics if available
        if self.analysis.semantic:
            result.update(
                {
                    "semantic_similarity": self.analysis.semantic.similarity,
                    "is_semantically_repetitive": self.analysis.semantic.is_repetitive,  # noqa: E501
                }
            )

        # Add surprise metrics if available
        if self.analysis.surprise:
            result.update(
                {
                    "semantic_surprise": self.analysis.surprise.semantic_surprise,  # noqa: E501
                    "max_semantic_surprise": self.analysis.surprise.max_semantic_surprise,  # noqa: E501
                    "is_surprising": self.analysis.surprise.is_surprising,
                }
            )

        # Add token likelihood metrics if available
        if self.analysis.token_likelihood:
            result.update(
                {
                    "avg_token_likelihood": self.analysis.token_likelihood.avg_token_likelihood,  # noqa: E501
                    "context_avg_likelihood": self.analysis.token_likelihood.context_avg_likelihood,  # noqa: E501
                }
            )

        # Add config if available
        if self.config:
            result.update(self.config.dict())

        return result
