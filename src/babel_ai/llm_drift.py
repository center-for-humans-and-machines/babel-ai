"""LLM Drift Experiment.

This module implements an experiment to analyze long-term
behavior of Large Language Models when they operate in a
self-loop without external input.
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.babel_ai.azure_open_ai import azure_openai_request

logger = logging.getLogger(__name__)


class LLMProvider:
    """Protocol defining the interface for LLM providers."""

    def generate(
        self,
        prompt: str,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.7,
        max_tokens: int = 100,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate text from the given prompt."""

        messages: list = [
            {"role": "user", "content": prompt},
        ]

        return azure_openai_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
        )


@dataclass
class ExperimentConfig:
    """Configuration for the drift experiment."""

    temperature: float = 0.7
    max_tokens: int = 100
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_p: float = 1.0
    max_iterations: int = 100
    max_total_characters: int = 1000000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }


class DriftAnalyzer:
    """Analyzes drift patterns in LLM outputs."""

    def __init__(self, semantic_model: str = "all-MiniLM-L6-v2"):
        """Initialize the DriftAnalyzer.

        Args:
            semantic_model: Name of the sentence-transformer model to use
        """
        self.semantic_model = SentenceTransformer(semantic_model)

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
        window_size: int = 3,
    ) -> Dict[str, Any]:
        """Analyze semantic surprise using KL divergence of embeddings.

        Args:
            current_text: The current output text
            previous_texts: List of previous outputs
            surprise_threshold: Threshold for considering output surprising
            window_size: Number of previous texts to consider for surprise

        Returns:
            Dictionary containing surprise metrics
        """
        # Get probability distribution for current text
        current_dist = self._get_semantic_distribution(current_text)

        # Initialize surprise metrics
        surprise_values = []

        # Calculate surprise against previous texts in window
        start_idx = max(0, len(previous_texts) - window_size)
        for prev_text in previous_texts[start_idx:]:
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


class PromptFetcher:
    """Fetches random prompts from various online sources."""

    REDDIT_WRITING_PROMPTS = "https://www.reddit.com/r/WritingPrompts/new.json"
    RANDOM_WORD_API = "https://random-word-api.herokuapp.com/word"

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PromptFetcher/1.0)"
        }

    def get_random_prompt(self, category: Optional[str] = None) -> str:
        """Get a random prompt from available sources.

        Args:
            category: Optional category
            ('creative', 'analytical', 'conversational')

        Returns:
            A random prompt string
        """
        methods = {
            "creative": self._get_writing_prompt,
            "analytical": self._get_analytical_prompt,
            "conversational": self._get_conversational_prompt,
        }

        if category and category in methods:
            method = methods[category]
        else:
            method = random.choice(list(methods.values()))

        try:
            return method()
        except Exception as e:
            logger.warning(f"Failed to fetch prompt: {e}")
            return self._get_fallback_prompt()

    def _get_writing_prompt(self) -> str:
        """Fetch a writing prompt from Reddit."""
        response = requests.get(
            self.REDDIT_WRITING_PROMPTS, headers=self.headers
        )
        response.raise_for_status()

        data = response.json()
        posts = data["data"]["children"]
        prompts = [
            post["data"]["title"]
            for post in posts
            if not post["data"]["title"].startswith("[WP]")
        ]

        return (
            random.choice(prompts) if prompts else self._get_fallback_prompt()
        )

    def _get_analytical_prompt(self) -> str:
        """Generate an analytical prompt using random words."""
        words = self._get_random_words(2)
        templates = [
            "Analyze the relationship between {} and {}.",
            "Compare and contrast {} with {}.",
            "Explain how {} influences {}.",
            "What are the implications of {} on {}?",
        ]
        return random.choice(templates).format(*words)

    def _get_conversational_prompt(self) -> str:
        """Generate a conversational prompt using random words."""
        word = self._get_random_words(1)[0]
        templates = [
            "What are your thoughts on {}?",
            "How does {} affect our daily lives?",
            "Why is {} important in modern society?",
            "Share your perspective on {}.",
        ]
        return random.choice(templates).format(word)

    def _get_random_words(self, count: int = 1) -> List[str]:
        """Fetch random words from the random word API."""
        response = requests.get(
            f"{self.RANDOM_WORD_API}?number={count}", headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def _get_fallback_prompt(self) -> str:
        """Return a fallback prompt when online fetching fails."""
        fallback_prompts = [
            "Share your thoughts on an interesting topic.",
            "Tell me about something you find fascinating.",
            "Describe a concept that intrigues you.",
            "What's on your mind?",
        ]
        return random.choice(fallback_prompts)


class DriftExperiment:
    """Main class for running LLM drift experiments."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[ExperimentConfig] = None,
        analyzer: Optional[DriftAnalyzer] = None,
        prompt_fetcher: Optional[PromptFetcher] = None,
    ):
        self.llm = llm_provider or LLMProvider()
        self.config = config or ExperimentConfig()
        self.analyzer = analyzer or DriftAnalyzer()
        self.prompt_fetcher = prompt_fetcher or PromptFetcher()
        self.results = []

    def _save_results_to_csv(
        self, metrics: List[Dict[str, Any]], timestamp: datetime
    ) -> None:
        """Save experiment results to a CSV file.

        Args:
            metrics: List of metric dictionaries from the experiment
            timestamp: Timestamp to use in filename
        """
        # Flatten metrics by expanding analysis dict into separate columns
        flattened_metrics = []
        for metric in metrics:
            flat_metric = {
                "iteration": metric["iteration"],
                "timestamp": metric["timestamp"],
                "response": metric["response"],
            }
            # Add analysis metrics as separate columns
            flat_metric.update(metric["analysis"])
            # Add configuration as columns with 'config_' prefix
            flat_metric.update({k: v for k, v in metric["config"].items()})
            flattened_metrics.append(flat_metric)

        # Convert to DataFrame and save
        df = pd.DataFrame(flattened_metrics)
        filename = (
            f"drift_experiment_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(filename, index=False)
        logger.info(f"Saved experiment results to {filename}")

    def fetch_prompt(self, category: Optional[str] = None) -> str:
        """Fetch a random prompt from the prompt fetcher.

        Args:
            category: Optional category for the prompt
                     ('creative', 'analytical', 'conversational')

        Returns:
            A random prompt string
        """
        return self.prompt_fetcher.get_random_prompt(category)

    def run(self, initial_prompt: str) -> List[Dict[str, Any]]:
        """Run the drift experiment with the given or fetched initial prompt.

        Args:
            initial_prompt: Optional initial prompt. If None, a random prompt
                          will be fetched.

        Returns:
            List of dictionaries containing experiment metrics
        """

        current_prompt = initial_prompt
        outputs = [initial_prompt]
        metrics = [
            {
                "iteration": 0,
                "timestamp": datetime.now(),
                "response": initial_prompt,
                "analysis": self.analyzer.analyze(outputs),
                "config": self.config.to_dict(),
            }
        ]

        for i in range(1, self.config.max_iterations + 1):
            # Generate next response
            response = self.llm.generate(
                prompt=current_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                top_p=self.config.top_p,
            )

            # Store output
            outputs.append(response)

            # Analyze all outputs
            analysis = self.analyzer.analyze(outputs)

            # Store results
            metrics.append(
                {
                    "iteration": i,
                    "timestamp": datetime.now(),
                    "analysis": analysis,
                    "response": response,
                    "config": self.config.to_dict(),
                }
            )

            # Check stop conditions
            # if analysis['is_repetitive']:
            #    break

            # if analysis['is_semantically_repetitive']:
            #    break

            # Check total length of outputs
            if (
                sum(len(output) for output in outputs)
                > self.config.max_total_characters
            ):
                break

            # Update prompt for next iteration
            current_prompt = response

        # Save results to CSV
        self._save_results_to_csv(metrics, metrics[0]["timestamp"])

        return metrics
