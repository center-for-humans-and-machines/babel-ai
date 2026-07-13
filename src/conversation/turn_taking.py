"""Deterministic turn-taking algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TurnTakingState:
    """Serializable turn-taking cursor."""

    next_agent_index: int = 0
    fixed_order_cursor: int = 0


class TurnTakingAlgorithm(ABC):
    """Pick which agent speaks next."""

    @abstractmethod
    def next_agent_index(
        self,
        num_agents: int,
        state: TurnTakingState,
    ) -> int:
        """Return index into the agent list."""
        ...

    @abstractmethod
    def snapshot(self, state: TurnTakingState) -> dict:
        """Export state for checkpoint."""
        ...

    @abstractmethod
    def restore(self, data: dict, state: TurnTakingState) -> None:
        """Import state from checkpoint."""
        ...


class RoundRobinTurnTaking(TurnTakingAlgorithm):
    """Deterministic round-robin over agent indices."""

    def next_agent_index(
        self,
        num_agents: int,
        state: TurnTakingState,
    ) -> int:
        if num_agents < 1:
            raise ValueError("Need at least one agent")
        index = state.next_agent_index % num_agents
        state.next_agent_index = (index + 1) % num_agents
        return index

    def snapshot(self, state: TurnTakingState) -> dict:
        return {"next_agent_index": state.next_agent_index}

    def restore(self, data: dict, state: TurnTakingState) -> None:
        state.next_agent_index = data["next_agent_index"]


class FixedOrderTurnTaking(TurnTakingAlgorithm):
    """Cycle through an explicit agent index list."""

    def __init__(self, order: List[int]):
        if not order:
            raise ValueError("fixed_order must not be empty")
        self._order = order

    def next_agent_index(
        self,
        num_agents: int,
        state: TurnTakingState,
    ) -> int:
        index = self._order[state.fixed_order_cursor % len(self._order)]
        if index < 0 or index >= num_agents:
            raise IndexError(
                f"fixed_order index {index} out of range for "
                f"{num_agents} agents"
            )
        state.fixed_order_cursor += 1
        return index

    def snapshot(self, state: TurnTakingState) -> dict:
        return {
            "fixed_order_cursor": state.fixed_order_cursor,
            "order": self._order,
        }

    def restore(self, data: dict, state: TurnTakingState) -> None:
        state.fixed_order_cursor = data["fixed_order_cursor"]
        if data.get("order") != self._order:
            raise ValueError("fixed_order mismatch on checkpoint restore")


def build_turn_taking(
    method: str,
    fixed_order: Optional[List[int]] = None,
) -> TurnTakingAlgorithm:
    """Factory from settings enum value."""
    if method == "round_robin":
        return RoundRobinTurnTaking()
    if method == "fixed_order":
        if not fixed_order:
            raise ValueError("fixed_order required for fixed_order method")
        return FixedOrderTurnTaking(fixed_order)
    raise ValueError(f"Unknown turn-taking method: {method}")
