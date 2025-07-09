"""Prompt fetcher classes for LLM drift experiments."""

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import requests

from babel_ai.enums import FetcherType

logger = logging.getLogger(__name__)


class BasePromptFetcher(ABC):
    """Abstract base class for prompt fetchers."""

    @abstractmethod
    def get_conversation(self) -> List[Dict[str, str]]:
        """Get a conversation.

        Returns:
            List of message dictionaries in the format:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        """
        pass

    @classmethod
    def create_fetcher(
        cls, fetcher_type: "FetcherType", **kwargs
    ) -> "BasePromptFetcher":
        """Factory method to create appropriate fetcher instance.

        Args:
            fetcher_type: Type of fetcher to create
            **kwargs: Fetcher-specific arguments

        Returns:
            Initialized fetcher instance
        """
        fetcher_class = fetcher_type.get_fetcher_class()
        return fetcher_class(**kwargs)


class RandomPromptFetcher(BasePromptFetcher):
    """Fetches random prompts from various online sources."""

    REDDIT_WRITING_PROMPTS = "https://www.reddit.com/r/WritingPrompts/new.json"
    RANDOM_WORD_API = "https://random-word-api.herokuapp.com/word"

    def __init__(self, category: Optional[str] = None):
        """Initialize the RandomPromptFetcher.

        Args:
            category: Optional category
                ('creative', 'analytical', 'conversational')
        """
        self.category = category
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PromptFetcher/1.0)"
        }

    def get_conversation(self) -> List[Dict[str, str]]:
        """Get a random prompt from available sources
        as single-message conversation.

        Returns:
            A single-message conversation as List[Dict[str, str]]
        """
        methods = {
            "creative": self._get_writing_prompt,
            "analytical": self._get_analytical_prompt,
            "conversational": self._get_conversational_prompt,
        }

        if self.category and self.category in methods:
            method = methods[self.category]
        else:
            method = random.choice(list(methods.values()))

        try:
            prompt_text = method()
        except Exception as e:
            logger.warning(f"Failed to fetch prompt: {e}")
            prompt_text = self._get_fallback_prompt()

        # Return as single-message conversation
        return [{"role": "user", "content": prompt_text}]

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


class ShareGPTConversationFetcher(BasePromptFetcher):
    """Fetches random conversations from ShareGPT conversations dataset."""

    def __init__(
        self,
        data_path: str,
        min_messages: int = 2,
        max_messages: Optional[int] = None,
    ):
        """Initialize the ShareGPT conversation fetcher.

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

    def get_conversation(self) -> List[Dict[str, str]]:
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


class InfiniteConversationFetcher(BasePromptFetcher):
    """Fetches random conversations from the Infinite Conversation dataset."""

    def __init__(
        self,
        data_path: str,
        min_messages: int = 2,
        max_messages: Optional[int] = None,
    ):
        """Initialize the Infinite Conversation fetcher.

        Args:
            data_dir: Path to directory containing conversation JSON files
            min_messages: Minimum number of messages in conversation
            max_messages: Maximum number of messages. If None, no upper limit.
        """
        self.data_dir = Path(data_path)
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.conversations = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess the Infinite Conversation dataset."""
        # Find all JSON files in the directory
        json_files = list(self.data_dir.glob("conversation_*.json"))

        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extract messages from the conversation
                messages = self._extract_messages(data)

                # Filter based on message count
                if len(messages) >= self.min_messages and (
                    self.max_messages is None
                    or len(messages) <= self.max_messages
                ):
                    self.conversations.append(
                        {"id": file_path.stem, "messages": messages}
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load conversation from {file_path}: {e}"
                )

        logger.info(
            f"Loaded {len(self.conversations)} "
            f"conversations from {self.data_dir}"
        )

    def _extract_messages(self, conversation_data: Dict) -> List[Dict]:
        """Extract and format messages from a conversation.

        Args:
            conversation_data: Raw conversation data from JSON file

        Returns:
            List of formatted message dictionaries
        """
        messages = []
        for key, value in conversation_data.items():
            # Extract speaker name from the text
            if ":" in value:
                speaker, text = value.split(":", 1)
            else:
                speaker = "Unknown"
                text = value

            # Extract timestamp from the key (MP3 filename)
            # Format is like: /slavoj_1704311301.66552.mp3
            try:
                # Extract the numeric part after the underscore and before .mp3
                timestamp_str = key.split("_")[1].split(".mp3")[0]
                timestamp = float(timestamp_str)
            except (IndexError, ValueError):
                # If we can't parse the timestamp, use a default
                timestamp = 0

            messages.append(
                {
                    "key": key,  # MP3 filename
                    "speaker": speaker.strip(),
                    "text": text.strip(),
                    "timestamp": timestamp,
                }
            )

        # Sort messages by timestamp to ensure chronological order
        messages.sort(key=lambda x: x["timestamp"])

        return messages

    def get_conversation(self) -> List[Dict[str, str]]:
        """Get a random conversation from the Infinite Conversation dataset.

        Returns:
            List of messages in the format expected by LLMProvider
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        """
        if not self.conversations:
            logger.warning("No conversations loaded")
            return []

        # Select a random conversation
        conversation = random.choice(self.conversations)

        # Convert to LLMProvider format
        messages = []
        current_role = "user"  # Start with user

        for msg in conversation["messages"]:
            # Alternate between user and assistant roles
            messages.append(
                {
                    "role": current_role,
                    "content": f"{msg['speaker']}: {msg['text']}",
                }
            )

            # Toggle role for next message
            current_role = "assistant" if current_role == "user" else "user"

        return messages


class TopicalChatConversationFetcher(BasePromptFetcher):
    """Fetches random conversations from the Topical-Chat dataset."""

    def __init__(
        self,
        data_path: str,
        second_data_path: Optional[str] = None,
        min_messages: int = 2,
        max_messages: Optional[int] = None,
    ):
        """Initialize the Topical-Chat conversation fetcher.

        Args:
            data_path: Path to first data file (e.g., test_rare.jsonl)
            second_data_path: Path to second data file (e.g., test_freq.jsonl)
            min_messages: Minimum number of messages in conversation
            max_messages: Maximum number of messages. If None, no upper limit.
        """
        self.data_path = Path(data_path)
        self.second_data_path = (
            Path(second_data_path) if second_data_path is not None else None
        )
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.conversations = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess the Topical-Chat dataset."""
        files_to_load = [
            path
            for path in [self.data_path, self.second_data_path]
            if path is not None
        ]

        for file_path in files_to_load:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        # Extract conversation from the nested structure
                        conversation_data = data[1]["content"]

                        # Convert to standard message format
                        messages = self._extract_messages(conversation_data)

                        # Filter based on message count
                        if len(messages) >= self.min_messages and (
                            self.max_messages is None
                            or len(messages) <= self.max_messages
                        ):
                            self.conversations.append(
                                {
                                    "id": data[0],
                                    "messages": messages,
                                    "article_url": data[1].get(
                                        "article_url", ""
                                    ),
                                    "config": data[1].get("config", ""),
                                    "conversation_rating": data[1].get(
                                        "conversation_rating", {}
                                    ),
                                }
                            )
            except Exception as e:
                logger.warning(
                    f"Failed to load conversations from {file_path}: {e}"
                )

        logger.info(
            f"Loaded {len(self.conversations)} "
            f"conversations from Topical-Chat dataset"
        )

    def _extract_messages(
        self, conversation_data: List[Dict]
    ) -> List[Dict[str, str]]:
        """Extract and format messages from a Topical-Chat conversation.

        Args:
            conversation_data: List of message dictionaries from JSON

        Returns:
            List of formatted message dictionaries
        """
        messages = []
        for msg in conversation_data:
            messages.append(
                {
                    "role": msg["agent"],
                    "content": msg["message"],
                }
            )

        return messages

    def get_conversation(self) -> List[Dict[str, str]]:
        """Get a random conversation from the Topical-Chat dataset.

        Returns:
            List of messages in the format expected by LLMProvider
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        """
        if not self.conversations:
            logger.warning("No conversations loaded")
            return []

        # Select a random conversation
        conversation = random.choice(self.conversations)
        # Remove the last two messages
        # TODO: Make this configurable
        messages = conversation["messages"][:-2]
        return messages
