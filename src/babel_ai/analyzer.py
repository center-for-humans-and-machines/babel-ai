"""Analyzer classes for LLM drift experiments."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from babel_ai.enums import AnalyzerType
from models import AnalysisResult

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
        logger.info(f"Creating analyzer of type {analyzer_type}")
        logger.debug(f"Analyzer kwargs: {kwargs}")

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

    # Configure tokenizer padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    max_context_length = token_model.config.max_position_embeddings

    token_model.eval()

    logger.info(
        f"Initialized SimilarityAnalyzer with "
        f"semantic model: {semantic_model_name}, "
        f"token model: {token_model_name}, "
        f"token model max context length: {max_context_length}."
    )

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
        logger.debug(
            "Initializing SimilarityAnalyzer "
            f"with analyze_window: {analyze_window}"
        )
        self.analyze_window = analyze_window

    def _analyze_word_stats(self, text: str) -> Tuple[int, int, float]:
        """Analyze basic word statistics of the text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (word_count, unique_word_count, coherence_score)
        """
        logger.info("Analyzing word stats")
        logger.debug(f"Word stats analysis text: {text[:50]}")

        if not text:
            logger.warning("Input text is empty.")

        words = text.lower().split()
        unique_words = set(words)

        word_count = len(words)
        unique_word_count = len(unique_words)
        coherence_score = len(unique_words) / len(words) if words else 0.0

        logger.debug(
            f"Word stats: word_count: {word_count}, "
            f"unique_word_count: {unique_word_count}, "
            f"coherence_score: {coherence_score}"
        )

        return word_count, unique_word_count, coherence_score

    def _analyze_lexical_similarity(
        self, outputs: List[str], window_size: int = 1
    ) -> Optional[float]:
        """Analyze lexical similarity with previous outputs.

        Args:
            outputs: List of all outputs including the current one
            window_size: Number of previous outputs to compare against

        Returns:
            Average Jaccard similarity score or None if no comparison possible
        """
        logger.info("Analyzing lexical similarity")
        logger.debug(f"Number of input texts: {len(outputs)}")
        logger.debug(
            f"Lexical similarity analysis input: "
            f"{[o[:50] for o in outputs]}"
        )
        logger.debug(f"Lexical similarity analysis window_size: {window_size}")

        if len(outputs) < 2:
            logger.warning(
                "Not enough outputs to analyze lexical similarity."
                f"Number of input texts: {len(outputs)}, "
                f"required: 2."
            )
            return None

        current_text = outputs[-1]
        logger.debug(f"Current text: {current_text[:50]}")
        current_words = set(current_text.lower().split())

        # Calculate similarities within the specified window
        similarities = []
        start_idx = max(0, len(outputs) - 1 - window_size)
        logger.debug(f"Comparing with {len(outputs) - start_idx} outputs")

        for i in range(start_idx, len(outputs) - 1):
            compare_text = outputs[i]
            compare_words = set(compare_text.lower().split())
            logger.debug(f"Comparing with {compare_text[:50]}")

            if compare_words and current_words:
                intersection = current_words.intersection(compare_words)
                union = current_words.union(compare_words)
                similarity = len(intersection) / len(union)
                similarities.append(similarity)

            logger.debug(f"Intersection: {intersection}")
            logger.debug(f"Union: {union}")
            logger.debug(f"Similarity: {similarity}")

        if similarities:
            logger.debug(f"Similarities: {similarities}")
            logger.debug(
                f"Average similarity: {sum(similarities) / len(similarities)}"
            )
            return sum(similarities) / len(similarities)

        logger.warning("No similarities found. Returning None.")
        return None

    def _analyze_semantic_similarity(
        self, outputs: List[str], window_size: int = 1
    ) -> Optional[float]:
        """Analyze semantic similarity using Sentence-BERT.

        Args:
            outputs: List of all outputs including the current one
            window_size: Number of previous outputs to compare against

        Returns:
            Average cosine similarity score or None if no comparison possible
        """
        logger.info("Analyzing semantic similarity")
        logger.debug(f"Number of input texts: {len(outputs)}")
        logger.debug(
            f"Semantic similarity analysis input: "
            f"{[o[:50] for o in outputs]}"
        )
        logger.debug(
            f"Semantic similarity analysis window_size: {window_size}"
        )
        logger.debug(f"Semantic model: {self.semantic_model_name}")

        if len(outputs) < 2:
            logger.warning(
                "Not enough outputs to analyze lexical similarity."
                f"Number of input texts: {len(outputs)}, "
                f"required: 2."
            )
            return None

        current_text = outputs[-1]
        logger.debug(f"Current text: {current_text[:50]}")
        current_embedding = self.semantic_model.encode(
            current_text, convert_to_tensor=True
        )

        # Calculate similarities within the specified window
        similarities = []
        start_idx = max(0, len(outputs) - 1 - window_size)
        logger.debug(f"Comparing with {len(outputs) - start_idx} outputs")

        for i in range(start_idx, len(outputs) - 1):
            compare_text = outputs[i]
            logger.debug(f"Comparing with {compare_text[:50]}")

            compare_embedding = self.semantic_model.encode(
                compare_text, convert_to_tensor=True
            )

            logger.debug("Calculating cosine similarity")
            similarity = cos_sim(current_embedding, compare_embedding).item()
            logger.debug(f"Cosine similarity: {similarity}")
            if similarity < -1.0 or similarity > 1.0:
                logger.warning(
                    f"Cosine similarity is {similarity}, "
                    "which is not -1.0 or 1.0. "
                    "Similarity will be clamped to [-1, 1]."
                )
                similarity = max(
                    -1.0, min(1.0, similarity)
                )  # Clamp to [-1, 1]
                logger.debug(f"Clamped cosine similarity: {similarity}.")
            similarities.append(similarity)

            # Log warning for the direct comparison case (window_size=1)
            if similarity <= 0:
                logger.warning(
                    f"Semantic similarity is <= 0. "
                    f"Similarity: {similarity} "
                    f"Current text: {current_text[:50]} "
                    f"Previous text: {compare_text[:50]} "
                )

        if similarities:
            logger.debug(f"Similarities: {similarities}")
            logger.debug(
                f"Average similarity: {sum(similarities) / len(similarities)}"
            )
            return sum(similarities) / len(similarities)

        logger.warning("No similarities found. Returning None.")
        return None

    def _analyze_token_perplexity(self, text: str) -> Optional[float]:
        """Analyze the perplexity of tokens in the text.

        Args:
            text: The text to analyze

        Returns:
            Average token perplexity or None if calculation not possible
        """
        logger.info("Analyzing token perplexity")
        logger.debug(f"Token perplexity analysis text: {text[:50]}")

        # Ensure input text is not empty
        if not text:
            logger.warning("Input text is empty. Returning max perplexity.")
            return float("inf")

        tokenized_inputs = self._text_to_tokenizer_encoding(text)

        perplexities = []

        for inputs in tokenized_inputs:
            logger.debug(f"Tokenized input: {inputs}")

            input_ids = inputs["input_ids"]

            # Validate token IDs are within vocabulary range
            vocab_size = self.token_model.config.vocab_size

            if torch.any(input_ids >= vocab_size):
                logger.warning(
                    f"Token IDs exceed vocabulary size: {vocab_size}"
                )
                continue

            # Get model predictions
            with torch.no_grad():
                outputs_model = self.token_model(**inputs)

                logger.debug("Computing logits")
                logits = outputs_model.logits[
                    0, :-1, :
                ]  # Shape: [seq_len-1, vocab_size]

                logger.debug("Getting target tokens")
                # Get target tokens (shifted by 1)
                target_ids = input_ids[0, 1:]

                logger.debug("Calculating log probabilities")
                # Calculate log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[
                    torch.arange(len(target_ids)), target_ids
                ]

                logger.debug("Calculating average log probability")
                # Calculate perplexity: exp(-mean(log_probs))
                avg_log_prob = token_log_probs.mean().item()

                logger.debug("Calculating perplexity by exp(-avg_log_prob)")
                perplexity = np.exp(-avg_log_prob)

                logger.debug(f"Perplexity: {perplexity}")
                perplexities.append(perplexity)

        if perplexities:
            return np.mean(perplexities)
        else:
            logger.warning("No valid perplexities from tokenized inputs")
            return float("inf")

    def _text_to_tokenizer_encoding(self, text: str) -> List[BatchEncoding]:
        """
        Convert text to a list of BatchEncoding objects
        within model context length.

        Recursively splits long text into smaller chunks that fit within the
        model's maximum context length. Each chunk is tokenized and returned
        as a BatchEncoding.

        Args:
            text: Input text to tokenize and split if needed

        Returns:
            List[BatchEncoding]: List of tokenized
                text chunks, each within the model's
                context length limit
        """
        logger.info("Converting text to tokenizer encoding")
        logger.debug(f"Text: {text[:50]}")

        encoding_list = []

        # Tokenize the text with proper truncation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length,
            padding=False,
        )
        input_ids = inputs["input_ids"]
        logger.debug(f"Input IDs: {input_ids.tolist()}")

        # Return the tensor if the length is fitting
        if input_ids.shape[1] >= self.max_context_length:
            # Split text by character length approximation
            logger.warning(
                f"Input text has {input_ids.shape[1]} tokens, "
                "which exceeds the maximum context length of "
                f"{self.max_context_length}. "
                f"Splitting into two halves."
            )
            mid_point = len(text) // 2
            first_half = text[:mid_point]
            second_half = text[mid_point:]

            encoding_list.extend(self._text_to_tokenizer_encoding(first_half))
            encoding_list.extend(self._text_to_tokenizer_encoding(second_half))

        elif input_ids.shape[1] < 2:
            logger.warning(
                "Input text has only one token. Perplexity calculation "
                "requires at least two tokens. Returning empty tensor list."
            )
        else:
            encoding_list.append(inputs)

        logger.debug(
            f"Returning encoding list with {len(encoding_list)} elements."
        )
        return encoding_list

    def analyze(self, outputs: List[str]) -> AnalysisResult:
        """Orchestrate different analysis methods on the outputs.

        Args:
            outputs: List of all outputs including the current one

        Returns:
            AnalysisResult containing combined analysis metrics
        """

        # Get word statistics and token perplexity
        # of the current text
        logger.info("Analyzing input texts")
        logger.debug(f"Input texts: {[o[:10] for o in outputs]}")

        current_text = outputs[-1]
        logger.debug(f"Current text: {current_text[:10]}")

        (
            word_count,
            unique_word_count,
            coherence_score,
        ) = self._analyze_word_stats(current_text)
        token_perplexity = self._analyze_token_perplexity(current_text)

        # Get similarity metrics (will return None if not enough outputs)
        # Direct similarity with previous output (window_size=1)
        lexical_similarity = self._analyze_lexical_similarity(
            outputs, window_size=1
        )
        semantic_similarity = self._analyze_semantic_similarity(
            outputs, window_size=1
        )

        # Rolling window similarity using the configured analyze_window
        lexical_similarity_window = self._analyze_lexical_similarity(
            outputs, window_size=self.analyze_window
        )
        semantic_similarity_window = self._analyze_semantic_similarity(
            outputs, window_size=self.analyze_window
        )

        logger.info("Analysis complete.")
        logger.debug(f"Word count: {word_count}")
        logger.debug(f"Unique word count: {unique_word_count}")
        logger.debug(f"Coherence score: {coherence_score}")
        logger.debug(f"Token perplexity: {token_perplexity}")
        logger.debug(f"Lexical similarity: {lexical_similarity}")
        logger.debug(f"Semantic similarity: {semantic_similarity}")
        logger.debug(f"Lexical similarity window: {lexical_similarity_window}")
        logger.debug(
            f"Semantic similarity window: {semantic_similarity_window}"
        )

        return AnalysisResult(
            word_count=word_count,
            unique_word_count=unique_word_count,
            coherence_score=coherence_score,
            token_perplexity=token_perplexity,
            lexical_similarity=lexical_similarity,
            semantic_similarity=semantic_similarity,
            lexical_similarity_window=lexical_similarity_window,
            semantic_similarity_window=semantic_similarity_window,
        )
