"""Tests for ShareGPTConversationFetcher class."""

import json
import os
import tempfile
from unittest import TestCase, main

from babel_ai.prompt_fetcher import ShareGPTConversationFetcher


class TestShareGPTConversationFetcher(TestCase):
    """Test cases for ShareGPTConversationFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test data file with varying conversation lengths
        self.test_data = [
            {
                "items": [  # 2 messages
                    {"from": "human", "value": "Hi"},
                    {"from": "assistant", "value": "Hello"},
                ]
            },
            {
                "items": [  # 4 messages
                    {"from": "human", "value": "Hello"},
                    {"from": "assistant", "value": "Hi there"},
                    {"from": "human", "value": "How are you?"},
                    {"from": "assistant", "value": "I'm doing well!"},
                ]
            },
            {
                "items": [  # 6 messages
                    {"from": "human", "value": "What is Python?"},
                    {"from": "assistant", "value": "A programming language"},
                    {"from": "human", "value": "Tell me more"},
                    {"from": "assistant", "value": "Python is versatile..."},
                    {"from": "human", "value": "What can I build with it?"},
                    {"from": "assistant", "value": "Many things..."},
                ]
            },
        ]

        # Create temporary file
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.temp_dir, "test_data.json")
        with open(self.data_path, "w") as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary file and directory
        os.remove(self.data_path)
        os.rmdir(self.temp_dir)

    def test_initialization_with_min_messages(self):
        """Test initialization with different min_messages values."""
        # Should include all conversations (all have >= 2 messages)
        fetcher = ShareGPTConversationFetcher(self.data_path, min_messages=2)
        self.assertEqual(len(fetcher.conversations), 3)

        # Should include conversations with 4 or more messages
        fetcher = ShareGPTConversationFetcher(self.data_path, min_messages=4)
        self.assertEqual(len(fetcher.conversations), 2)

        # Should include only the longest conversation
        fetcher = ShareGPTConversationFetcher(self.data_path, min_messages=6)
        self.assertEqual(len(fetcher.conversations), 1)

        # Should include no conversations
        fetcher = ShareGPTConversationFetcher(self.data_path, min_messages=8)
        self.assertEqual(len(fetcher.conversations), 0)

    def test_initialization_with_max_messages(self):
        """Test initialization with different max_messages values."""
        # Should include only conversations with 2-3 messages
        fetcher = ShareGPTConversationFetcher(
            self.data_path, min_messages=2, max_messages=3
        )
        self.assertEqual(len(fetcher.conversations), 1)

        # Should include conversations with 2-4 messages
        fetcher = ShareGPTConversationFetcher(
            self.data_path, min_messages=2, max_messages=4
        )
        self.assertEqual(len(fetcher.conversations), 2)

        # Should include all conversations (none have > 6 messages)
        fetcher = ShareGPTConversationFetcher(
            self.data_path, min_messages=2, max_messages=6
        )
        self.assertEqual(len(fetcher.conversations), 3)

    def test_initialization_with_min_and_max_messages(self):
        """Test initialization with both min and max message constraints."""
        # Should include conversations with 4-5 messages
        fetcher = ShareGPTConversationFetcher(
            self.data_path, min_messages=4, max_messages=5
        )
        self.assertEqual(len(fetcher.conversations), 1)

        # Should include no conversations (impossible range)
        fetcher = ShareGPTConversationFetcher(
            self.data_path, min_messages=5, max_messages=3
        )
        self.assertEqual(len(fetcher.conversations), 0)

    def test_get_conversation(self):
        """Test getting random conversations."""
        fetcher = ShareGPTConversationFetcher(
            self.data_path, min_messages=2, max_messages=4
        )
        conversation = fetcher.get_conversation()

        # Verify conversation structure
        self.assertIsInstance(conversation, list)
        self.assertTrue(2 <= len(conversation) <= 4)

        # Verify message format
        first_message = conversation[0]
        self.assertIn("role", first_message)
        self.assertIn("content", first_message)
        self.assertIn(first_message["role"], ["human", "assistant"])

    def test_get_conversation_empty_dataset(self):
        """Test getting conversation from empty dataset."""
        # Create empty dataset
        empty_data = []
        empty_path = os.path.join(self.temp_dir, "empty_data.json")
        with open(empty_path, "w") as f:
            json.dump(empty_data, f)

        fetcher = ShareGPTConversationFetcher(empty_path, min_messages=2)

        # Should raise an exception when trying to get conversation
        with self.assertRaises(IndexError):
            fetcher.get_conversation()

        # Clean up
        os.remove(empty_path)


if __name__ == "__main__":
    main()
