"""Comprehensive test runner for all prompt fetchers."""

import unittest

# Import all test classes
from .test_base_fetcher import TestBasePromptFetcher, TestFetcherType
from .test_infinite_conversation_fetcher import TestInfiniteConversationFetcher
from .test_random_fetcher import TestRandomPromptFetcher
from .test_sharegpt_fetcher import TestShareGPTConversationFetcher
from .test_topical_chat_fetcher import TestTopicalChatConversationFetcher


def create_test_suite():
    """Create a test suite with all prompt fetcher tests."""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestBasePromptFetcher,
        TestFetcherType,
        TestRandomPromptFetcher,
        TestShareGPTConversationFetcher,
        TestInfiniteConversationFetcher,
        TestTopicalChatConversationFetcher,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    return test_suite


def run_all_tests():
    """Run all prompt fetcher tests."""
    test_suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return result


if __name__ == "__main__":
    run_all_tests()
