"""Tests for ConversationManager."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from api.enums import OpenAIModels, Provider
from conversation.agents import LLMConversationAgent
from conversation.checkpoint import CheckpointWriter, ConversationState
from conversation.manager import ConversationManager
from conversation.messages import (
    ContextStack,
    ConversationMessage,
    MessageSource,
)
from conversation.settings import (
    AnalysisPolicy,
    ConversationSettings,
)
from conversation.status import ConversationStatus
from enums import FetcherType
from models import AgentConfig, FetcherConfig


@pytest.fixture
def mock_analyzer():
    analyzer = Mock()
    analyzer.analyze.return_value = Mock(
        word_count=1,
        unique_word_count=1,
        coherence_score=1.0,
        lexical_similarity=None,
        semantic_similarity=None,
        lexical_similarity_window=None,
        semantic_similarity_window=None,
        token_perplexity=None,
    )
    return analyzer


@pytest.fixture
def mock_llm_agent():
    agent = Mock()
    agent.id = "abc-123"
    agent.config = AgentConfig(
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
    )
    agent.generate_response.return_value = "reply"
    return agent


def test_manager_runs_until_max_iterations(
    tmp_path, mock_analyzer, mock_llm_agent
):
    settings = ConversationSettings(
        max_iterations=4,
        max_total_characters=10_000,
        checkpoint_enabled=False,
        analysis_policy=AnalysisPolicy.PER_TURN,
    )
    conv = LLMConversationAgent(mock_llm_agent, speaker="agent_0")
    manager = ConversationManager(
        agents=[conv],
        settings=settings,
        analyzer=mock_analyzer,
        run_dir=tmp_path,
        fetcher_config=FetcherConfig(
            fetcher=FetcherType.RANDOM,
            category="creative",
        ),
    )
    seed = [{"role": "user", "content": "hi"}]
    metrics = manager.run(seed)
    assert len(metrics) == 4
    assert mock_llm_agent.generate_response.call_count == 3


def test_checkpoint_save_and_restore_stack(tmp_path):
    writer = CheckpointWriter(tmp_path)
    stack = ContextStack()
    stack.append(
        ConversationMessage(
            turn_index=0,
            role="user",
            speaker="seed",
            content="hello",
            source=MessageSource.SEED,
            timestamp=datetime(2025, 1, 1),
        )
    )
    state = ConversationState(
        run_id="run-1",
        status=ConversationStatus.RUNNING,
    )
    writer.save(
        state=state,
        stack=stack,
        metrics=[],
        turn_taking_data={"next_agent_index": 0},
        settings={"max_iterations": 10},
    )
    data = CheckpointWriter.load(tmp_path / "checkpoint.json")
    restored = CheckpointWriter.restore_stack(data)
    assert len(restored.messages) == 1
    assert restored.messages[0].content == "hello"
