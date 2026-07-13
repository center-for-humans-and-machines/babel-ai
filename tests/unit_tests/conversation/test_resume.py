"""Tests for checkpoint resume (E4)."""

from datetime import datetime
from unittest.mock import Mock

from api.enums import OpenAIModels, Provider
from conversation.agents import LLMConversationAgent
from conversation.checkpoint import CheckpointWriter, ConversationState
from conversation.manager import ConversationManager
from conversation.messages import (
    ContextStack,
    ConversationMessage,
    MessageSource,
)
from conversation.settings import ConversationSettings
from conversation.status import ConversationStatus
from models import AgentConfig


def test_resume_from_continues_conversation(tmp_path):
    """Resume should preserve stack and append new agent turns."""
    mock_analyzer = Mock()
    mock_analyzer.analyze.return_value = Mock(
        word_count=1,
        unique_word_count=1,
        coherence_score=1.0,
    )
    mock_agent = Mock()
    mock_agent.id = "agent-1"
    mock_agent.config = AgentConfig(
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT4_1106_PREVIEW,
    )
    mock_agent.generate_response.side_effect = ["t2", "t3", "t4", "t5"]
    conv = LLMConversationAgent(mock_agent, speaker="agent_0")

    run_path = tmp_path / "results" / "run-1"
    run_path.mkdir(parents=True)
    stack = ContextStack()
    stack.append(
        ConversationMessage(
            turn_index=0,
            role="user",
            speaker="seed",
            content="seed",
            source=MessageSource.SEED,
        )
    )
    stack.append(
        ConversationMessage(
            turn_index=1,
            role="assistant",
            speaker="agent_0",
            content="mid-run",
            source=MessageSource.AGENT,
        )
    )
    writer = CheckpointWriter(run_path)
    writer.save(
        state=ConversationState(
            run_id="run-1",
            status=ConversationStatus.RUNNING,
            agent_turn_count=1,
        ),
        stack=stack,
        metrics=[],
        turn_taking_data={"next_agent_index": 0},
        settings=ConversationSettings(max_iterations=6).model_dump(),
        agents=[conv],
    )

    resumed = ConversationManager.resume_from(
        run_path / "checkpoint.json",
        agents=[conv],
        analyzer=mock_analyzer,
    )
    assert len(resumed.stack.messages) == 2
    resumed.continue_run()
    assert len(resumed.stack.messages) > 2
    assert mock_agent.generate_response.call_count >= 1


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
        agents=[],
    )
    data = CheckpointWriter.load(tmp_path / "checkpoint.json")
    restored = CheckpointWriter.restore_stack(data)
    assert len(restored.messages) == 1
    assert restored.messages[0].content == "hello"
