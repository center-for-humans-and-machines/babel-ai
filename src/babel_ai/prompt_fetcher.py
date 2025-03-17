"""Prompt fetcher classes for LLM drift experiments."""

import json
import logging
import random
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


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


class ShareGPTPromptFetcher:
    """Fetches random conversations from ShareGPT conversations dataset."""

    def __init__(
        self,
        data_path: str,
        min_messages: int = 2,
        max_messages: Optional[int] = None,
    ):
        """Initialize the ShareGPT prompt fetcher.

        Args:
            data_path: Path to sharegpt_clean.json file
            min_messages:
                Minimum number of messages in conversation to consider
            max_messages:
                Maximum number of messages in conversation to consider.
                    If None, no upper limit is applied.
        """
        self.data_path = data_path
        self.min_messages = min_messages
        self.max_messages = max_messages
        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess the ShareGPT dataset."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        # Extract items and filter for conversation length
        self.conversations = [d["items"] for d in data]
        self.conversations = [
            conv
            for conv in self.conversations
            if len(conv) >= self.min_messages
            and (self.max_messages is None or len(conv) <= self.max_messages)
        ]

    def get_random_prompt(self) -> List[Dict[str, str]]:
        """Get a random conversation thread from ShareGPT conversations.

        Returns:
            List of messages in the format expected by LLMProvider
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        """
        # Select random conversation
        conversation = random.choice(self.conversations)

        # Convert ShareGPT format to LLMProvider format
        messages = []
        for msg in conversation:
            role = "user" if msg["from"] == "human" else "assistant"
            messages.append({"role": role, "content": msg["value"]})

        return messages
