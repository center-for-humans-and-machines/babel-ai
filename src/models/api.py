"""Pydantic models for LLM API requests and responses."""

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Response model for LLM API calls."""

    content: str = Field(..., description="The generated text response")
    input_token_count: int = Field(
        ..., description="Number of tokens in the input"
    )
    output_token_count: int = Field(
        ..., description="Number of tokens in the output"
    )
