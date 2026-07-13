"""Conversation lifecycle status."""

from enum import Enum


class ConversationStatus(str, Enum):
    """High-level state of a managed conversation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
