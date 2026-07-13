"""Tests for turn-taking algorithms."""

import pytest

from conversation.turn_taking import (
    FixedOrderTurnTaking,
    RoundRobinTurnTaking,
    TurnTakingState,
)


def test_round_robin_cycles_two_agents():
    algo = RoundRobinTurnTaking()
    state = TurnTakingState()
    assert algo.next_agent_index(2, state) == 0
    assert algo.next_agent_index(2, state) == 1
    assert algo.next_agent_index(2, state) == 0


def test_fixed_order_follows_sequence():
    algo = FixedOrderTurnTaking([1, 1, 0])
    state = TurnTakingState()
    assert algo.next_agent_index(2, state) == 1
    assert algo.next_agent_index(2, state) == 1
    assert algo.next_agent_index(2, state) == 0


def test_round_robin_checkpoint_roundtrip():
    algo = RoundRobinTurnTaking()
    state = TurnTakingState()
    algo.next_agent_index(3, state)
    snap = algo.snapshot(state)
    restored = TurnTakingState()
    algo.restore(snap, restored)
    assert restored.next_agent_index == state.next_agent_index


def test_fixed_order_requires_non_empty():
    with pytest.raises(ValueError):
        FixedOrderTurnTaking([])
