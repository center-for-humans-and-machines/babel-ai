"""Tests for InfiniteConversationFetcher class."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, main

from babel_ai.prompt_fetcher import InfiniteConversationFetcher


class TestInfiniteConversationFetcher(TestCase):
    """Test cases for InfiniteConversationFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test data directory with sample conversations
        self.temp_dir = tempfile.mkdtemp()

        # Create sample conversation files
        self.conversations = [
            # Conversation 1 (3 messages)
            {
                "/slavoj_1704311301.66552.mp3": "Slavoj Zizek: This is test message 1",  # noqa: E501
                "/werner_1704311332.325497.mp3": "Werner Herzog: This is test response 1",  # noqa: E501
                "/slavoj_1704311359.906828.mp3": "Slavoj Zizek: This is test message 2",  # noqa: E501
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

    def test_initialization_default(self):
        """Test basic initialization with default parameters."""
        fetcher = InfiniteConversationFetcher(self.temp_dir)
        # Should include all conversations by default
        self.assertEqual(len(fetcher.conversations), 3)

        # Check that each conversation has the correct number of messages
        message_counts = [
            len(conv["messages"]) for conv in fetcher.conversations
        ]
        self.assertIn(3, message_counts)  # First conversation has 3 messages
        self.assertIn(5, message_counts)  # Second conversation has 5 messages
        self.assertIn(7, message_counts)  # Third conversation has 7 messages

    def test_initialization_with_min_messages(self):
        """Test initialization with different min_messages values."""
        # Should include all conversations (all have >= 2 messages)
        fetcher = InfiniteConversationFetcher(self.temp_dir, min_messages=2)
        self.assertEqual(len(fetcher.conversations), 3)

        # Should include conversations with 5 or more messages
        fetcher = InfiniteConversationFetcher(self.temp_dir, min_messages=5)
        self.assertEqual(len(fetcher.conversations), 2)

        # Should include only the longest conversation
        fetcher = InfiniteConversationFetcher(self.temp_dir, min_messages=7)
        self.assertEqual(len(fetcher.conversations), 1)

        # Should include no conversations
        fetcher = InfiniteConversationFetcher(self.temp_dir, min_messages=10)
        self.assertEqual(len(fetcher.conversations), 0)

    def test_initialization_with_max_messages(self):
        """Test initialization with different max_messages values."""
        # Should include only conversations with <= 3 messages
        fetcher = InfiniteConversationFetcher(
            self.temp_dir, min_messages=2, max_messages=3
        )
        self.assertEqual(len(fetcher.conversations), 1)

        # Should include conversations with <= 5 messages
        fetcher = InfiniteConversationFetcher(
            self.temp_dir, min_messages=2, max_messages=5
        )
        self.assertEqual(len(fetcher.conversations), 2)

        # Should include all conversations
        fetcher = InfiniteConversationFetcher(
            self.temp_dir, min_messages=2, max_messages=10
        )
        self.assertEqual(len(fetcher.conversations), 3)

    def test_initialization_with_min_and_max_messages(self):
        """Test initialization with both min and max message constraints."""
        # Should include conversations with 5-6 messages
        fetcher = InfiniteConversationFetcher(
            self.temp_dir, min_messages=5, max_messages=6
        )
        self.assertEqual(len(fetcher.conversations), 1)

        # Should include no conversations (impossible range)
        fetcher = InfiniteConversationFetcher(
            self.temp_dir, min_messages=6, max_messages=4
        )
        self.assertEqual(len(fetcher.conversations), 0)

    def test_get_conversation(self):
        """Test getting random conversations."""
        fetcher = InfiniteConversationFetcher(self.temp_dir)
        conversation = fetcher.get_conversation()

        # Verify conversation structure
        self.assertIsInstance(conversation, list)
        # Length should match one of our test conversations
        self.assertIn(len(conversation), [3, 5, 7])

        # Verify message format
        first_message = conversation[0]
        self.assertIn("role", first_message)
        self.assertIn("content", first_message)
        self.assertIn(first_message["role"], ["user", "assistant"])

    def test_get_conversation_empty_dataset(self):
        """Test getting conversation from empty dataset."""
        # Create empty directory
        empty_dir = tempfile.mkdtemp()

        try:
            fetcher = InfiniteConversationFetcher(empty_dir)

            # Should return empty list when no conversations loaded
            conversation = fetcher.get_conversation()
            self.assertEqual(conversation, [])
        finally:
            os.rmdir(empty_dir)

    def test_get_conversation_with_filtering(self):
        """Test getting conversations with message length filtering."""
        # Test with only short conversations
        fetcher = InfiniteConversationFetcher(
            self.temp_dir, min_messages=3, max_messages=3
        )
        conversation = fetcher.get_conversation()

        # Should only return 3-message conversations
        self.assertEqual(len(conversation), 3)

    def test_timestamp_sorting(self):
        """Test that messages are properly sorted by timestamp."""
        fetcher = InfiniteConversationFetcher(self.temp_dir)

        # Check each conversation's messages are sorted by timestamp
        for conv in fetcher.conversations:
            messages = conv["messages"]
            timestamps = [msg["timestamp"] for msg in messages]
            self.assertEqual(timestamps, sorted(timestamps))

    def test_speaker_extraction(self):
        """Test that speaker names are properly extracted."""
        fetcher = InfiniteConversationFetcher(self.temp_dir)

        # Check that speakers are correctly extracted
        for conv in fetcher.conversations:
            for msg in conv["messages"]:
                self.assertIn(
                    msg["speaker"], ["Slavoj Zizek", "Werner Herzog"]
                )

    def test_role_alternation(self):
        """Test that roles alternate properly in get_conversation."""
        fetcher = InfiniteConversationFetcher(self.temp_dir)
        conversation = fetcher.get_conversation()

        # Check that roles alternate between user and assistant
        for i, message in enumerate(conversation):
            if i % 2 == 0:
                self.assertEqual(message["role"], "user")
            else:
                self.assertEqual(message["role"], "assistant")

    def test_content_format(self):
        """Test that content includes speaker names."""
        fetcher = InfiniteConversationFetcher(self.temp_dir)
        conversation = fetcher.get_conversation()

        # Check that each message content includes speaker name
        for message in conversation:
            content = message["content"]
            self.assertTrue(
                "Slavoj Zizek:" in content or "Werner Herzog:" in content
            )


if __name__ == "__main__":
    main()
