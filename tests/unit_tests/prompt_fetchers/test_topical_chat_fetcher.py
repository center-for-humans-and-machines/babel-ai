"""Tests for TopicalChatConversationFetcher class."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, main

from babel_ai.prompt_fetcher import TopicalChatConversationFetcher


class TestTopicalChatConversationFetcher(TestCase):
    """Test cases for TopicalChatConversationFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test data directory
        self.temp_dir = tempfile.mkdtemp()

        # Create sample test data in TopicalChat format
        self.rare_data = [
            [
                "t_conversation_1",
                {
                    "article_url": "https://example.com/article1",
                    "config": "A",
                    "content": [
                        {
                            "message": "Hello, how are you?",
                            "agent": "agent_1",
                            "sentiment": "Happy",
                            "knowledge_source": ["FS1"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "I'm doing well, thanks!",
                            "agent": "agent_2",
                            "sentiment": "Happy",
                            "knowledge_source": ["FS1"],
                            "turn_rating": "Excellent",
                        },
                    ],
                    "conversation_rating": {
                        "agent_1": "Good",
                        "agent_2": "Excellent",
                    },
                },
            ],
            [
                "t_conversation_2",
                {
                    "article_url": "https://example.com/article2",
                    "config": "B",
                    "content": [
                        {
                            "message": "What do you think about AI?",
                            "agent": "agent_1",
                            "sentiment": "Curious",
                            "knowledge_source": ["FS2"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "AI is fascinating and powerful.",
                            "agent": "agent_2",
                            "sentiment": "Positive",
                            "knowledge_source": ["FS2"],
                            "turn_rating": "Excellent",
                        },
                        {
                            "message": "I agree, it has many applications.",
                            "agent": "agent_1",
                            "sentiment": "Positive",
                            "knowledge_source": ["FS2"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "Yes, from healthcare to entertainment.",  # noqa: E501
                            "agent": "agent_2",
                            "sentiment": "Informative",
                            "knowledge_source": ["FS2"],
                            "turn_rating": "Excellent",
                        },
                    ],
                    "conversation_rating": {
                        "agent_1": "Good",
                        "agent_2": "Excellent",
                    },
                },
            ],
        ]

        self.freq_data = [
            [
                "t_conversation_3",
                {
                    "article_url": "https://example.com/article3",
                    "config": "C",
                    "content": [
                        {
                            "message": "Do you like music?",
                            "agent": "agent_1",
                            "sentiment": "Curious",
                            "knowledge_source": ["FS3"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "Yes, I love all kinds of music.",
                            "agent": "agent_2",
                            "sentiment": "Happy",
                            "knowledge_source": ["FS3"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "What's your favorite genre?",
                            "agent": "agent_1",
                            "sentiment": "Curious",
                            "knowledge_source": ["FS3"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "I really enjoy jazz and classical.",
                            "agent": "agent_2",
                            "sentiment": "Happy",
                            "knowledge_source": ["FS3"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "Those are great choices!",
                            "agent": "agent_1",
                            "sentiment": "Happy",
                            "knowledge_source": ["FS3"],
                            "turn_rating": "Good",
                        },
                        {
                            "message": "Thank you! What about you?",
                            "agent": "agent_2",
                            "sentiment": "Curious",
                            "knowledge_source": ["FS3"],
                            "turn_rating": "Good",
                        },
                    ],
                    "conversation_rating": {
                        "agent_1": "Good",
                        "agent_2": "Good",
                    },
                },
            ]
        ]

        # Create temporary JSONL files
        self.rare_file_path = Path(self.temp_dir) / "test_rare.jsonl"
        self.freq_file_path = Path(self.temp_dir) / "test_freq.jsonl"

        # Write rare data
        with open(self.rare_file_path, "w") as f:
            for item in self.rare_data:
                f.write(json.dumps(item) + "\n")

        # Write freq data
        with open(self.freq_file_path, "w") as f:
            for item in self.freq_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files and directory
        os.remove(self.rare_file_path)
        os.remove(self.freq_file_path)
        os.rmdir(self.temp_dir)

    def test_initialization_both_files(self):
        """Test initialization with both rare and freq files."""
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=2
        )
        # Should load 2 from rare + 1 from freq = 3 total
        self.assertEqual(len(fetcher.conversations), 3)

    def test_initialization_with_min_messages(self):
        """Test initialization with different min_messages values."""
        # Should include conversations with >= 2 messages
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=2
        )
        self.assertEqual(len(fetcher.conversations), 3)

        # Should include conversations with >= 4 messages
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=4
        )
        self.assertEqual(
            len(fetcher.conversations), 2
        )  # 4-msg and 6-msg convs

        # Should include conversations with >= 6 messages
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=6
        )
        self.assertEqual(len(fetcher.conversations), 1)  # Only 6-msg conv

        # Should include no conversations
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=10
        )
        self.assertEqual(len(fetcher.conversations), 0)

    def test_initialization_with_max_messages(self):
        """Test initialization with different max_messages values."""
        # Should include conversations with <= 2 messages
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path,
            self.freq_file_path,
            min_messages=2,
            max_messages=2,
        )
        self.assertEqual(len(fetcher.conversations), 1)  # Only 2-msg conv

        # Should include conversations with <= 4 messages
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path,
            self.freq_file_path,
            min_messages=2,
            max_messages=4,
        )
        self.assertEqual(
            len(fetcher.conversations), 2
        )  # 2-msg and 4-msg convs

        # Should include all conversations
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path,
            self.freq_file_path,
            min_messages=2,
            max_messages=10,
        )
        self.assertEqual(len(fetcher.conversations), 3)

    def test_get_conversation(self):
        """Test getting random conversations."""
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=2
        )
        conversation = fetcher.get_conversation()

        # Verify conversation structure
        self.assertIsInstance(conversation, list)
        self.assertIn(len(conversation), [2, 4, 6])  # Our test data lengths

        # Verify message format
        first_message = conversation[0]
        self.assertIn("role", first_message)
        self.assertIn("content", first_message)
        self.assertIn(first_message["role"], ["agent_1", "agent_2"])

    def test_get_conversation_empty_dataset(self):
        """Test getting conversation from empty dataset."""
        # Create empty files
        empty_rare = Path(self.temp_dir) / "empty_rare.jsonl"
        empty_freq = Path(self.temp_dir) / "empty_freq.jsonl"

        with open(empty_rare, "w"):
            pass  # Empty file
        with open(empty_freq, "w"):
            pass  # Empty file

        try:
            fetcher = TopicalChatConversationFetcher(
                empty_rare, empty_freq, min_messages=2
            )

            # Should return empty list when no conversations loaded
            conversation = fetcher.get_conversation()
            self.assertEqual(conversation, [])
        finally:
            os.remove(empty_rare)
            os.remove(empty_freq)

    def test_message_extraction(self):
        """Test that messages are properly extracted and formatted."""
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path,
            self.freq_file_path,
            min_messages=2,
            max_messages=2,
        )
        conversation = fetcher.get_conversation()

        # Should be the 2-message conversation
        self.assertEqual(len(conversation), 2)
        self.assertEqual(conversation[0]["role"], "agent_1")
        self.assertEqual(conversation[0]["content"], "Hello, how are you?")
        self.assertEqual(conversation[1]["role"], "agent_2")
        self.assertEqual(conversation[1]["content"], "I'm doing well, thanks!")

    def test_conversation_ratings_stored(self):
        """Test that conversation ratings are properly stored."""
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=2
        )

        # Check that conversation ratings are stored in the loaded data
        for conv in fetcher.conversations:
            self.assertIn("conversation_rating", conv)
            self.assertIsInstance(conv["conversation_rating"], dict)
            self.assertIn("agent_1", conv["conversation_rating"])
            self.assertIn("agent_2", conv["conversation_rating"])

    def test_metadata_stored(self):
        """Test that metadata is properly stored."""
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=2
        )

        # Check that metadata is stored
        for conv in fetcher.conversations:
            self.assertIn("id", conv)
            self.assertIn("article_url", conv)
            self.assertIn("config", conv)
            self.assertTrue(conv["id"].startswith("t_conversation_"))
            self.assertTrue(conv["article_url"].startswith("https://"))
            self.assertIn(conv["config"], ["A", "B", "C"])

    def test_role_preservation(self):
        """Test that agent roles are preserved as specified."""
        fetcher = TopicalChatConversationFetcher(
            self.rare_file_path, self.freq_file_path, min_messages=2
        )
        conversation = fetcher.get_conversation()

        # Check that roles are preserved as agent_1 and agent_2
        for message in conversation:
            self.assertIn(message["role"], ["agent_1", "agent_2"])


if __name__ == "__main__":
    main()
