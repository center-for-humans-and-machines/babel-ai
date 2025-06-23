"""Analyzer classes for LLM drift experiments."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer

from babel_ai.enums import AnalyzerType
from babel_ai.models import AnalysisResult

logger = logging.getLogger(__name__)


class Analyzer(ABC):
    """Abstract base class for analyzers."""

    @abstractmethod
    def analyze(self, outputs: List[str]) -> AnalysisResult:
        """Analyze the outputs."""
        pass

    @classmethod
    def create_analyzer(
        cls, analyzer_type: "AnalyzerType", **kwargs
    ) -> "Analyzer":
        """Create an analyzer from a analyzer type."""
        analyzer_class = analyzer_type.get_class()
        return analyzer_class(**kwargs)


class SimilarityAnalyzer(Analyzer):
    """Analyzes similarity patterns in LLM outputs."""

    # Semantic similarity model
    semantic_model_name = "all-MiniLM-L6-v2"
    semantic_model = SentenceTransformer(semantic_model_name)

    # Token model and tokenizer
    token_model_name = "gpt2"
    token_model = AutoModelForCausalLM.from_pretrained(token_model_name)
    tokenizer = AutoTokenizer.from_pretrained(token_model_name)

    max_context_length = token_model.config.max_position_embeddings

    token_model.eval()

    def __init__(
        self,
        analyze_window: int = 20,
    ):
        """Initialize the SimilarityAnalyzer.

        Args:
            semantic_model: Name of the sentence-transformer model to use
            analyze_window: Number of previous texts to analyze
            token_model: Name of the model to use for token likelihood
        """
        self.analyze_window = analyze_window

    def _analyze_word_stats(self, text: str) -> Tuple[int, int, float]:
        """Analyze basic word statistics of the text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (word_count, unique_word_count, coherence_score)
        """
        words = text.lower().split()
        unique_words = set(words)

        word_count = len(words)
        unique_word_count = len(unique_words)
        coherence_score = len(unique_words) / len(words) if words else 0.0

        return word_count, unique_word_count, coherence_score

    def _analyze_lexical_similarity(
        self, outputs: List[str]
    ) -> Optional[float]:
        """Analyze lexical similarity with previous outputs.

        Args:
            outputs: List of all outputs including the current one

        Returns:
            Jaccard similarity score or None if no comparison possible
        """
        if len(outputs) < 2:
            return None

        current_text = outputs[-1]
        previous_text = outputs[-2]

        current_words = set(current_text.lower().split())
        prev_words = set(previous_text.lower().split())

        if prev_words and current_words:
            intersection = current_words.intersection(prev_words)
            union = current_words.union(prev_words)
            return len(intersection) / len(union)

        return None

    def _analyze_semantic_similarity(
        self, outputs: List[str]
    ) -> Optional[float]:
        """Analyze semantic similarity using Sentence-BERT.

        Args:
            outputs: List of all outputs including the current one

        Returns:
            Cosine similarity score or None if no comparison possible
        """
        if len(outputs) < 2:
            return None

        current_text = outputs[-1]
        previous_text = outputs[-2]

        # Encode current and previous text
        current_embedding = self.semantic_model.encode(
            current_text, convert_to_tensor=True
        )
        prev_embedding = self.semantic_model.encode(
            previous_text, convert_to_tensor=True
        )

        # Calculate cosine similarity and clamp to valid range
        similarity = cos_sim(current_embedding, prev_embedding).item()
        similarity = max(-1.0, min(1.0, similarity))  # Clamp to [-1, 1]

        if similarity <= 0:
            logger.warning(
                f"Semantic similarity is <= 0. "
                f"Similarity: {similarity} "
                f"Current text: {current_text} "
                f"Previous text: {previous_text} "
            )

        return similarity

    def _analyze_token_perplexity(self, text: str) -> Optional[float]:
        """Analyze the perplexity of tokens in the text.

        Args:
            text: The text to analyze

        Returns:
            Average token perplexity or None if calculation not possible
        """
        # Ensure input text is not empty
        if not text:
            logger.warning("Input text is empty. Returning max perplexity.")
            return float("inf")

        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Check if we have enough tokens for perplexity calculation
        if input_ids.shape[1] < 2:
            logger.warning(
                "Input text has only one token. Perplexity calculation "
                "requires at least two tokens. Returning max perplexity."
            )
            return float("inf")

        if len(text) > self.max_context_length:
            logger.warning(
                f"Input text exceeds maximum context length of "
                f"{self.max_context_length} tokens. Splitting into chunks."
            )
            first_block = text[: self.max_context_length]
            second_block = text[self.max_context_length :]

            perplexities = []
            for block in [first_block, second_block]:
                if block.strip():  # Only analyze non-empty blocks
                    result = self._analyze_token_perplexity(block)
                    if result is not None:
                        perplexities.append(result)

            return np.mean(perplexities) if perplexities else None

        # Get model predictions
        with torch.no_grad():
            outputs_model = self.token_model(**inputs)
            logits = outputs_model.logits[
                0, :-1, :
            ]  # Shape: [seq_len-1, vocab_size]

            # Get target tokens (shifted by 1)
            target_ids = input_ids[0, 1:]

            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[
                torch.arange(len(target_ids)), target_ids
            ]

            # Calculate perplexity: exp(-mean(log_probs))
            avg_log_prob = token_log_probs.mean().item()
            perplexity = np.exp(-avg_log_prob)

        return perplexity

    def analyze(self, outputs: List[str]) -> AnalysisResult:
        """Orchestrate different analysis methods on the outputs.

        Args:
            outputs: List of all outputs including the current one

        Returns:
            AnalysisResult containing combined analysis metrics
        """

        # Get word statistics and token perplexity
        # of the current text
        current_text = outputs[-1]

        (
            word_count,
            unique_word_count,
            coherence_score,
        ) = self._analyze_word_stats(current_text)
        token_perplexity = self._analyze_token_perplexity(current_text)

        # Get similarity metrics (will return None if not enough outputs)
        lexical_similarity = self._analyze_lexical_similarity(outputs)
        semantic_similarity = self._analyze_semantic_similarity(outputs)

        return AnalysisResult(
            word_count=word_count,
            unique_word_count=unique_word_count,
            coherence_score=coherence_score,
            token_perplexity=token_perplexity,
            lexical_similarity=lexical_similarity,
            semantic_similarity=semantic_similarity,
        )
