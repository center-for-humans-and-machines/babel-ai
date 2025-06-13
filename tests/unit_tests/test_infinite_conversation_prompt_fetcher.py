"""Tests for InfiniteConversationPromptFetcher class."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, main

from babel_ai.prompt_fetcher import InfiniteConversationPromptFetcher


class TestInfiniteConversationPromptFetcher(TestCase):
    """Test cases for InfiniteConversationPromptFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test data directory with sample conversations
        self.temp_dir = tempfile.mkdtemp()

        # Create sample conversation files
        self.conversations = [
            # Conversation 1 (3 messages)
            {
                "/slavoj_1704311301.66552.mp3": "Slavoj Zizek: This is a test message 1",  # noqa: E501
                "/werner_1704311332.325497.mp3": "Werner Herzog: This is a test response 1",  # noqa: E501
                "/slavoj_1704311359.906828.mp3": "Slavoj Zizek: This is a test message 2",  # noqa: E501
            },
            # Conversation 2 (5 messages)
            {
                "/slavoj_1704311401.12345.mp3": "Slavoj Zizek: This is another test message 1",  # noqa: E501
                "/werner_1704311432.67890.mp3": "Werner Herzog: This is another test response 1",  # noqa: E501
                "/slavoj_1704311459.54321.mp3": "Slavoj Zizek: This is another test message 2",  # noqa: E501
                "/werner_1704311486.98765.mp3": "Werner Herzog: This is another test response 2",  # noqa: E501
                "/slavoj_1704311522.13579.mp3": "Slavoj Zizek: This is another test message 3",  # noqa: E501
            },
            # Conversation 3 (7 messages)
            {
                "/slavoj_1704311601.24680.mp3": "Slavoj Zizek: This is a third test message 1",  # noqa: E501
                "/werner_1704311632.86420.mp3": "Werner Herzog: This is a third test response 1",  # noqa: E501
                "/slavoj_1704311659.97531.mp3": "Slavoj Zizek: This is a third test message 2",  # noqa: E501
                "/werner_1704311686.75319.mp3": "Werner Herzog: This is a third test response 2",  # noqa: E501
                "/slavoj_1704311722.86420.mp3": "Slavoj Zizek: This is a third test message 3",  # noqa: E501
                "/werner_1704311736.97531.mp3": "Werner Herzog: This is a third test response 3",  # noqa: E501
                "/slavoj_1704311768.24680.mp3": "Slavoj Zizek: This is a third test message 4",  # noqa: E501
            },
        ]

        # Write conversations to files
        for i, conv in enumerate(self.conversations):
            file_path = (
                Path(self.temp_dir) / f"conversation_2024010{i+1}-2058.json"
            )
            with open(file_path, "w") as f:
                json.dump(conv, f)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files and directory
        for i in range(len(self.conversations)):
            file_path = (
                Path(self.temp_dir) / f"conversation_2024010{i+1}-2058.json"
            )
            if file_path.exists():
                os.remove(file_path)
        os.rmdir(self.temp_dir)

    def test_initialization(self):
        """Test basic initialization."""
        fetcher = InfiniteConversationPromptFetcher(self.temp_dir)
        # Should include all conversations
        self.assertEqual(len(fetcher.conversations), 3)

        # Check that each conversation has the correct number of messages
        message_counts = [
            len(conv["messages"]) for conv in fetcher.conversations
        ]
        self.assertIn(3, message_counts)  # First conversation has 3 messages
        self.assertIn(5, message_counts)  # Second conversation has 5 messages
        self.assertIn(7, message_counts)  # Third conversation has 7 messages

    def test_get_random_prompt(self):
        """Test getting random prompts."""
        fetcher = InfiniteConversationPromptFetcher(self.temp_dir)
        prompt = fetcher.get_random_prompt()

        # Verify prompt structure
        self.assertIsInstance(prompt, list)
        # Length should match one of our test conversations
        self.assertIn(len(prompt), [3, 5, 7])

        # Verify message format
        first_message = prompt[0]
        self.assertIn("role", first_message)
        self.assertIn("content", first_message)
        self.assertIn(first_message["role"], ["user", "assistant"])

    def test_timestamp_sorting(self):
        """Test that messages are properly sorted by timestamp."""
        fetcher = InfiniteConversationPromptFetcher(self.temp_dir)

        # Check each conversation's messages are sorted by timestamp
        for conv in fetcher.conversations:
            messages = conv["messages"]
            timestamps = [msg["timestamp"] for msg in messages]
            self.assertEqual(timestamps, sorted(timestamps))

    def test_speaker_extraction(self):
        """Test that speaker names are properly extracted."""
        fetcher = InfiniteConversationPromptFetcher(self.temp_dir)

        # Check that speakers are correctly extracted
        for conv in fetcher.conversations:
            for msg in conv["messages"]:
                self.assertIn(
                    msg["speaker"], ["Slavoj Zizek", "Werner Herzog"]
                )


if __name__ == "__main__":
    main()
