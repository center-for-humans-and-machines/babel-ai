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


class TokenPerplexityMetrics(BaseModel):
    """Metrics for token perplexity analysis."""

    avg_token_perplexity: float = Field(
        description="Average perplexity of all tokens in the text",
        ge=1.0,  # Perplexity is always >= 1
    )


class AnalysisResult(BaseModel):
    """Combined analysis results for text outputs."""

    word_stats: WordStats
    lexical: Optional[LexicalMetrics] = None
    semantic: Optional[SemanticMetrics] = None
    token_perplexity: Optional[TokenPerplexityMetrics] = None


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

        # Add token perplexity metrics if available
        if self.analysis.token_perplexity:
            result.update(
                {
                    "avg_token_perplexity": self.analysis.token_perplexity.avg_token_perplexity,  # noqa: E501
                }
            )

        # Add config if available
        if self.config:
            result.update(self.config.dict())

        return result
