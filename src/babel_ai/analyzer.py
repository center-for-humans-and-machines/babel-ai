"""Analyzer classes for LLM drift experiments."""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Analyzes similarity patterns in LLM outputs."""

    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        analyze_window: int = 20,
    ):
        """Initialize the SimilarityAnalyzer.

        Args:
            semantic_model: Name of the sentence-transformer model to use
        """
        self.semantic_model = SentenceTransformer(semantic_model)
        self.analyze_window = analyze_window

    def analyze(self, outputs: List[str]) -> Dict[str, Any]:
        """Orchestrate different analysis methods on the outputs.

        Args:
            outputs: List of all outputs including the current one

        Returns:
            Dictionary containing combined analysis metrics
        """
        current_text = outputs[-1]
        previous_texts = outputs[:-1] if len(outputs) > 1 else []

        analysis = {}

        # Combine results from all analysis methods
        analysis.update(self._analyze_word_stats(current_text))
        if previous_texts:
            analysis.update(
                self._analyze_similarity(current_text, previous_texts)
            )
            analysis.update(
                self._analyze_semantic_similarity(current_text, previous_texts)
            )
            analysis.update(
                self._analyze_semantic_surprise(current_text, previous_texts)
            )

        return analysis

    def _analyze_word_stats(self, text: str) -> Dict[str, Any]:
        """Analyze basic word statistics of the text."""
        words = text.lower().split()
        unique_words = set(words)

        return {
            "word_count": len(words),
            "unique_word_count": len(unique_words),
            "coherence_score": (
                len(unique_words) / len(words) if words else 0.0
            ),
        }

    def _analyze_similarity(
        self,
        current_text: str,
        previous_texts: List[str],
        similarity_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """Analyze lexical similarity with previous outputs."""
        current_words = set(current_text.lower().split())
        prev_text = previous_texts[-1]
        prev_words = set(prev_text.lower().split())

        analysis = {"is_repetitive": False}

        if prev_words and current_words:
            intersection = current_words.intersection(prev_words)
            union = current_words.union(prev_words)
            similarity = len(intersection) / len(union)

            analysis.update(
                {
                    "lexical_similarity": similarity,
                    "is_repetitive": similarity > similarity_threshold,
                }
            )

        return analysis

    def _analyze_semantic_similarity(
        self,
        current_text: str,
        previous_texts: List[str],
        semantic_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """Analyze semantic similarity using Sentence-BERT.

        Args:
            current_text: The current output text
            previous_texts: List of previous outputs
            semantic_threshold: Threshold for semantic similarity

        Returns:
            Dictionary containing semantic similarity metrics
        """
        # Encode current and previous text
        current_embedding = self.semantic_model.encode(
            current_text, convert_to_tensor=True
        )
        prev_embedding = self.semantic_model.encode(
            previous_texts[-1], convert_to_tensor=True
        )

        # Calculate cosine similarity
        similarity = cos_sim(current_embedding, prev_embedding).item()

        return {
            "semantic_similarity": similarity,
            "is_semantically_repetitive": similarity > semantic_threshold,
        }

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
    ) -> Dict[str, Any]:
        """Analyze semantic surprise using KL divergence of embeddings.

        Args:
            current_text: The current output text
            previous_texts: List of previous outputs
            surprise_threshold: Threshold for considering output surprising
            window_size: Number of previous texts to consider for surprise
                        If None, uses all available history

        Returns:
            Dictionary containing surprise metrics
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

        return {
            "semantic_surprise": avg_surprise,
            "max_semantic_surprise": max_surprise,
            "is_surprising": avg_surprise > surprise_threshold,
        }
