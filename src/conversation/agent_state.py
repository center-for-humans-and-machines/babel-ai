"""Snapshot and restore mutable per-agent session state."""

from typing import Any, List


def snapshot_agents(agents: List[Any]) -> dict[str, Any]:
    """Collect optional ``export_state`` payloads keyed by ``agent_id``."""
    states: dict[str, Any] = {}
    for agent in agents:
        export = getattr(agent, "export_state", None)
        if callable(export):
            states[agent.agent_id] = export()
    return states


def restore_agents(agents: List[Any], states: dict[str, Any]) -> None:
    """Apply saved state to agents that implement ``import_state``."""
    for agent in agents:
        payload = states.get(agent.agent_id)
        if payload is None:
            continue
        import_state = getattr(agent, "import_state", None)
        if callable(import_state):
            import_state(payload)
