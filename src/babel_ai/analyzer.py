"""Analyzer classes for LLM drift experiments."""

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import (
    AnalysisResult,
    LexicalMetrics,
    SemanticMetrics,
    SurpriseMetrics,
    TokenPerplexityMetrics,
    WordStats,
)

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Analyzes similarity patterns in LLM outputs."""

    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        analyze_window: int = 20,
        token_model: str = "gpt2",
    ):
        """Initialize the SimilarityAnalyzer.

        Args:
            semantic_model: Name of the sentence-transformer model to use
            analyze_window: Number of previous texts to analyze
            token_model: Name of the model to use for token likelihood
        """
        self._model_name = semantic_model
        self.semantic_model = SentenceTransformer(self._model_name)
        self.analyze_window = analyze_window

        # Initialize token model and tokenizer
        self.token_model = AutoModelForCausalLM.from_pretrained(token_model)
        self.tokenizer = AutoTokenizer.from_pretrained(token_model)
        self.token_model.eval()  # Set to evaluation mode

    def _analyze_word_stats(self, text: str) -> WordStats:
        """Analyze basic word statistics of the text.

        Args:
            text: Input text to analyze

        Returns:
            WordStats containing word-level statistics
        """
        words = text.lower().split()
        unique_words = set(words)

        return WordStats(
            word_count=len(words),
            unique_word_count=len(unique_words),
            coherence_score=len(unique_words) / len(words) if words else 0.0,
        )

    def _analyze_similarity(
        self,
        current_text: str,
        previous_texts: List[str],
        similarity_threshold: float = 0.8,
    ) -> LexicalMetrics:
        """Analyze lexical similarity with previous outputs.

        Args:
            current_text: Current text to analyze
            previous_texts: List of previous texts to compare against
            similarity_threshold: Threshold for considering text repetitive

        Returns:
            LexicalMetrics containing similarity analysis results
        """
        current_words = set(current_text.lower().split())
        prev_text = previous_texts[-1]
        prev_words = set(prev_text.lower().split())

        similarity = None
        is_repetitive = False

        if prev_words and current_words:
            intersection = current_words.intersection(prev_words)
            union = current_words.union(prev_words)
            similarity = len(intersection) / len(union)
            is_repetitive = similarity > similarity_threshold

        return LexicalMetrics(
            similarity=similarity, is_repetitive=is_repetitive
        )

    def _analyze_semantic_similarity(
        self,
        current_text: str,
        previous_texts: List[str],
        semantic_threshold: float = 0.9,
    ) -> SemanticMetrics:
        """Analyze semantic similarity using Sentence-BERT.

        Args:
            current_text: The current output text
            previous_texts: List of previous outputs
            semantic_threshold: Threshold for semantic similarity

        Returns:
            SemanticMetrics containing semantic similarity analysis
        """
        previous_text = previous_texts[-1]

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
                f"Previous text: {previous_texts[-1]} "
            )

        return SemanticMetrics(
            similarity=similarity,
            is_repetitive=similarity > semantic_threshold,
        )

    def _get_semantic_distribution(self, text: str) -> np.ndarray:
        """Get probability distribution from sentence embedding.

        Args:
            text: Input text to encode

        Returns:
            Normalized probability distribution from embedding
        """
        # Get embedding and convert to numpy
        embedding = (
            self.semantic_model.encode(text, convert_to_tensor=True)
            .cpu()
            .numpy()
        )

        # Normalize to create a probability distribution
        # Using softmax to ensure positive values that sum to 1
        prob_dist = F.softmax(torch.tensor(embedding), dim=0).numpy()

        return prob_dist

    def _analyze_semantic_surprise(
        self,
        current_text: str,
        previous_texts: List[str],
        surprise_threshold: float = 2.0,
    ) -> SurpriseMetrics:
        """Analyze semantic surprise using KL divergence of embeddings.

        Args:
            current_text: The current output text
            previous_texts: List of previous outputs
            surprise_threshold: Threshold for considering output surprising

        Returns:
            SurpriseMetrics containing surprise analysis results
        """
        # Get probability distribution for current text
        current_dist = self._get_semantic_distribution(current_text)

        # Initialize surprise metrics
        surprise_values = []

        # Use all history if window_size is None
        if self.analyze_window is None:
            texts_to_compare = previous_texts
        else:
            # Calculate surprise against previous texts in window
            start_idx = max(0, len(previous_texts) - self.analyze_window)
            texts_to_compare = previous_texts[start_idx:]

        # Calculate surprise against previous texts in window
        for prev_text in texts_to_compare:
            prev_dist = self._get_semantic_distribution(prev_text)
            # Calculate KL divergence (semantic surprise)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            surprise = entropy(current_dist + epsilon, prev_dist + epsilon)
            surprise_values.append(surprise)

        # Calculate average surprise if we have values
        if surprise_values:
            avg_surprise = np.mean(surprise_values)
            max_surprise = np.max(surprise_values)
        else:
            avg_surprise = 0.0
            max_surprise = 0.0

        return SurpriseMetrics(
            semantic_surprise=avg_surprise,
            max_semantic_surprise=max_surprise,
            is_surprising=avg_surprise > surprise_threshold,
        )

    def _analyze_token_perplexity(
        self,
        text: str,
    ) -> TokenPerplexityMetrics:
        """Analyze the perplexity of tokens in the text.

        Args:
            text: The text to analyze

        Returns:
            TokenPerplexityMetrics containing perplexity analysis
        """
        # Function to get token perplexity for a given text
        def get_token_perplexity(input_text: str) -> float:
            # Ensure input text is not empty
            if not input_text:
                logger.warning(
                    "Input text is empty. Returning max perplexity."
                )
                return float("inf")

            # Tokenize the text
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # Get model predictions
            with torch.no_grad():
                outputs = self.token_model(**inputs)
                logits = outputs.logits[
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

        # Get perplexity for current text
        avg_perplexity = get_token_perplexity(text)

        return TokenPerplexityMetrics(
            avg_token_perplexity=avg_perplexity,
        )

    def analyze(self, outputs: List[str]) -> AnalysisResult:
        """Orchestrate different analysis methods on the outputs.

        Args:
            outputs: List of all outputs including the current one

        Returns:
            AnalysisResult containing combined analysis metrics
        """
        current_text = outputs[-1]
        previous_texts = outputs[:-1] if len(outputs) > 1 else []

        # Get word statistics
        word_stats = self._analyze_word_stats(current_text)

        # Initialize optional metrics
        lexical_metrics = None
        semantic_metrics = None
        surprise_metrics = None
        token_perplexity_metrics = None

        if previous_texts:
            # Get lexical metrics
            lexical_metrics = self._analyze_similarity(
                current_text, previous_texts
            )

            # Get semantic metrics
            semantic_metrics = self._analyze_semantic_similarity(
                current_text, previous_texts
            )

            # Get surprise metrics
            surprise_metrics = self._analyze_semantic_surprise(
                current_text, previous_texts
            )

        # Get token perplexity metrics
        token_perplexity_metrics = self._analyze_token_perplexity(current_text)

        return AnalysisResult(
            word_stats=word_stats,
            lexical=lexical_metrics,
            semantic=semantic_metrics,
            surprise=surprise_metrics,
            token_perplexity=token_perplexity_metrics,
        )
