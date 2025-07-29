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
        logger.info(f"Creating fetcher instance for {fetcher_type.value}")
        logger.debug(f"Fetcher kwargs: {kwargs}")

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
        logger.info(
            f"Initializing RandomPromptFetcher with category: {category}"
        )
        self.category = category
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PromptFetcher/1.0)"
        }
        logger.debug(f"RandomPromptFetcher headers: {self.headers}")

    def get_conversation(self) -> List[Dict[str, str]]:
        """Get a random prompt from available sources
        as single-message conversation.

        Returns:
            A single-message conversation as List[Dict[str, str]]
        """
        logger.info("Getting conversation from RandomPromptFetcher")

        methods = {
            "creative": self._get_writing_prompt,
            "analytical": self._get_analytical_prompt,
            "conversational": self._get_conversational_prompt,
        }

        if self.category and self.category in methods:
            logger.debug(f"Using category: {self.category}")
            method = methods[self.category]
        else:
            logger.warning("No category provided. Using random category.")
            method = random.choice(list(methods.values()))

        try:
            prompt_text = method()
        except Exception as e:
            logger.warning(f"Failed to fetch prompt: {e}")
            prompt_text = self._get_fallback_prompt()

        # Return as single-message conversation
        logger.info(
            f"RandomPromptFetcher returning prompt: {prompt_text[:50]}"
        )
        return [{"role": "user", "content": prompt_text}]

    def _get_writing_prompt(self) -> str:
        """Fetch a writing prompt from Reddit."""
        logger.info("Fetching writing prompt from Reddit")
        logger.debug(
            f"Reddit writing prompts URL: {self.REDDIT_WRITING_PROMPTS}"
        )

        response = requests.get(
            self.REDDIT_WRITING_PROMPTS, headers=self.headers
        )
        logger.debug(f"Reddit writing prompts response: {response.json()}")

        response.raise_for_status()

        data = response.json()
        posts = data["data"]["children"]
        prompts = [
            post["data"]["title"]
            for post in posts
            if not post["data"]["title"].startswith("[WP]")
        ]
        logger.debug(f"Possible prompts: {[p[:50] for p in prompts]}")
        prompt = (
            random.choice(prompts) if prompts else self._get_fallback_prompt()
        )
        logger.debug(f"Returning writing prompt: {prompt[:50]}")

        return prompt

    def _get_analytical_prompt(self) -> str:
        """Generate an analytical prompt using random words."""
        logger.info("Generating analytical prompt using random words")

        words = self._get_random_words(2)
        templates = [
            "Analyze the relationship between {} and {}.",
            "Compare and contrast {} with {}.",
            "Explain how {} influences {}.",
            "What are the implications of {} on {}?",
        ]
        prompt = random.choice(templates).format(*words)
        logger.debug(f"Returning analytical prompt: {prompt[:50]}")

        return prompt

    def _get_conversational_prompt(self) -> str:
        """Generate a conversational prompt using random words."""
        logger.info("Generating conversational prompt using random words")

        word = self._get_random_words(1)[0]
        logger.debug(f"Random word: {word}")
        templates = [
            "What are your thoughts on {}?",
            "How does {} affect our daily lives?",
            "Why is {} important in modern society?",
            "Share your perspective on {}.",
        ]
        prompt = random.choice(templates).format(word)
        logger.debug(f"Returning conversational prompt: {prompt[:50]}")

        return prompt

    def _get_random_words(self, count: int = 1) -> List[str]:
        """Fetch random words from the random word API."""
        logger.info(f"Fetching {count} random words from random word API")
        logger.debug(f"Random word API URL: {self.RANDOM_WORD_API}")

        response = requests.get(
            f"{self.RANDOM_WORD_API}?number={count}", headers=self.headers
        )
        logger.debug(f"Random word API response: {response.json()}")

        response.raise_for_status()
        return response.json()

    def _get_fallback_prompt(self) -> str:
        """Return a fallback prompt when online fetching fails."""
        logger.warning("Returning fallback prompt")
        fallback_prompts = [
            "Share your thoughts on an interesting topic.",
            "Tell me about something you find fascinating.",
            "Describe a concept that intrigues you.",
            "What's on your mind?",
        ]
        prompt = random.choice(fallback_prompts)
        logger.debug(f"Returning fallback prompt: {prompt[:50]}")
        return prompt


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
        logger.info(
            "Initializing ShareGPTConversationFetcher "
            f"from data path: {data_path}"
        )
        logger.debug(
            f"ShareGPTConversationFetcher min_messages: {min_messages}"
        )
        logger.debug(
            f"ShareGPTConversationFetcher max_messages: {max_messages}"
        )

        self.data_path = data_path
        self.min_messages = min_messages
        self.max_messages = max_messages
        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess the ShareGPT dataset."""
        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, "r") as f:
            data = json.load(f)

        logger.debug(f"Loaded data from {self.data_path}")

        # Extract items and filter for conversation length
        self.conversations = [d["items"] for d in data]
        self.conversations = [
            conv
            for conv in self.conversations
            if len(conv) >= self.min_messages
            and (self.max_messages is None or len(conv) <= self.max_messages)
        ]
        logger.debug(f"Loaded {len(self.conversations)} conversations")

    def get_conversation(self) -> List[Dict[str, str]]:
        """Get a random conversation thread from ShareGPT conversations.

        Returns:
            List of messages in the format expected by LLMProvider
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        """
        logger.info("Getting conversation from ShareGPTConversationFetcher")
        logger.debug(f"Available conversations: {len(self.conversations)}")
        # Select random conversation
        conversation = random.choice(self.conversations)

        # Convert ShareGPT format to LLMProvider format
        messages = []
        for msg in conversation:
            messages.append({"role": msg["from"], "content": msg["value"]})
        logger.debug("Converted conversation to LLMProvider format.")

        logger.debug(f"Returning conversation with {len(messages)} messages")
        logger.debug(
            "Conversation head:\n"
            + "\n".join(
                [f"{m['role']}: {m['content'][:50]}" for m in messages[:5]]
            )
        )
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
        logger.info(
            "Initializing InfiniteConversationFetcher "
            f"from data path: {data_path}"
        )
        logger.debug(
            f"InfiniteConversationFetcher min_messages: {min_messages}"
        )
        logger.debug(
            f"InfiniteConversationFetcher max_messages: {max_messages}"
        )

        self.data_path = Path(data_path)
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.conversations = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess the Infinite Conversation dataset."""
        logger.info(
            "Loading data from 'conversation_*.json' files "
            f"in {self.data_path}"
        )

        # Find all JSON files in the directory
        json_files = list(self.data_path.glob("conversation_*.json"))

        for file_path in json_files:
            try:
                logger.debug(f"Loading conversation from {file_path}")

                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extract messages from the conversation
                messages = self._extract_messages(data)

                # Filter based on message count
                if len(messages) >= self.min_messages and (
                    self.max_messages is None
                    or len(messages) <= self.max_messages
                ):
                    logger.debug(
                        f"Adding conversation with {len(messages)} messages"
                    )
                    logger.debug(
                        f"Adding conversation under id: {file_path.stem}"
                    )
                    self.conversations.append(
                        {"id": file_path.stem, "messages": messages}
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load conversation from {file_path}: {e}"
                )

        logger.debug(
            f"Loaded {len(self.conversations)} "
            f"conversations from {self.data_path}"
        )

    def _extract_messages(self, conversation_data: Dict) -> List[Dict]:
        """Extract and format messages from a conversation.

        Args:
            conversation_data: Raw conversation data from JSON file

        Returns:
            List of formatted message dictionaries
        """
        logger.info("Extracting messages from conversation data")
        logger.debug(f"Conversation dict length: {len(conversation_data)}")

        messages = []
        for key, value in conversation_data.items():
            # Extract speaker name from the text
            if ":" in value:
                speaker, text = value.split(":", 1)
            else:
                speaker = "Unknown"
                logger.warning(f"No speaker found in entry: {value[:50]}")
                text = value

            logger.debug(f"Extracted speaker: {speaker} and text: {text[:50]}")

            logger.debug("Extracting timestamp from key.")
            logger.debug(f"Key: {key}")
            # Extract timestamp from the key (MP3 filename)
            # Format is like: /slavoj_1704311301.66552.mp3
            try:
                # Extract the numeric part after the underscore and before .mp3
                timestamp_str = key.split("_")[1].split(".mp3")[0]
                logger.debug(f"Timestamp string: {timestamp_str}")
                timestamp = float(timestamp_str)
                logger.debug(f"Timestamp: {timestamp}")
            except (IndexError, ValueError):
                # If we can't parse the timestamp, use a default
                timestamp = 0
                logger.warning(f"Failed to extract timestamp from key: {key}")
                logger.debug(f"Timestamp: {timestamp}")

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

        logger.debug(f"Extracted {len(messages)} messages")
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
        logger.info("Getting conversation from InfiniteConversationFetcher")
        logger.debug(f"Available conversations: {len(self.conversations)}")

        if not self.conversations:
            logger.warning("No conversations loaded")
            return []

        # Select a random conversation
        conversation = random.choice(self.conversations)
        logger.debug(f"Selected conversation: {conversation['id']}")

        # Convert to LLMProvider format
        messages = []
        for msg in conversation["messages"]:
            # Alternate between user and assistant roles
            messages.append(
                {
                    "role": msg["speaker"],
                    "content": msg["text"],
                }
            )

        logger.debug(f"Returning conversation with {len(messages)} messages")
        logger.debug(
            "Conversation head:\n"
            + "\n".join(
                [f"{m['role']}: {m['content'][:50]}" for m in messages[:5]]
            )
        )

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
        logger.info(
            "Initializing TopicalChatConversationFetcher "
            f"from data path: {data_path}"
        )
        if second_data_path is not None:
            logger.info(
                "Initializing TopicalChatConversationFetcher "
                f"from second data path: {second_data_path}"
            )
        else:
            logger.debug("No second data path provided.")
        logger.debug(
            f"TopicalChatConversationFetcher min_messages: {min_messages}"
        )
        logger.debug(
            f"TopicalChatConversationFetcher max_messages: {max_messages}"
        )

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
        logger.info(
            f"Loading data from {self.data_path} and {self.second_data_path}"
        )

        files_to_load = [
            path
            for path in [self.data_path, self.second_data_path]
            if path is not None
        ]

        for file_path in files_to_load:
            logger.debug(f"Loading data from {file_path}")
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            try:
                with open(file_path, "r") as f:
                    for i, line in enumerate(f):
                        logger.debug(f"Loading line {i} of {file_path}")

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
                            logger.debug(
                                "Adding conversation with "
                                f"{len(messages)} messages"
                            )
                            logger.debug(
                                f"Adding conversation under id: {data[0]}"
                            )
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

        logger.debug(
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
        logger.debug("Extracting messages from conversation data")

        messages = []
        for msg in conversation_data:
            messages.append(
                {
                    "role": msg["agent"],
                    "content": msg["message"],
                }
            )

        logger.debug(f"Extracted {len(messages)} messages")
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
        logger.info("Getting conversation from TopicalChatConversationFetcher")
        logger.debug(f"Available conversations: {len(self.conversations)}")

        if not self.conversations:
            logger.warning("No conversations loaded")
            return []

        # Select a random conversation
        conversation = random.choice(self.conversations)
        logger.debug(f"Selected conversation: {conversation['id']}")
        logger.debug(
            "Selected conversation with "
            f"{len(conversation['messages'])} messages"
        )

        # Remove the last two messages
        # TODO: Make this configurable
        logger.debug("Removing last two messages from conversation")
        messages = conversation["messages"][:-2]
        logger.debug(f"Returning conversation with {len(messages)} messages")
        logger.debug(
            "Messages head:\n"
            + "\n".join(
                [f"{m['role']}: {m['content'][:50]}" for m in messages[:5]]
            )
        )

        return messages
