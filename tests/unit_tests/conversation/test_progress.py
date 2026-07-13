"""Tests for live conversation progress rendering."""

import io

from conversation.progress import TurnProgress


def test_turn_progress_disabled_writes_only_completion():
    stream = io.StringIO()
    progress = TurnProgress(10, enabled=False, stream=stream)
    progress.update(3, "eliza", "In what way?")
    progress.complete("Saved run.")
    assert stream.getvalue() == "Saved run.\n"


def test_turn_progress_renders_bar_and_preview():
    stream = io.StringIO()
    progress = TurnProgress(4, enabled=True, stream=stream)
    progress.update(2, "agent_0", "Hello there.")
    output = stream.getvalue()
    assert "Turn 2/4" in output
    assert "[######" in output
    assert "agent_0: Hello there." in output


def test_turn_progress_overwrites_previous_render():
    stream = io.StringIO()
    progress = TurnProgress(4, enabled=True, stream=stream)
    progress.update(1, "eliza", "First reply.")
    progress.update(2, "agent_0", "Second reply.")
    assert stream.getvalue().count("Turn 2/4") == 1
    assert "Second reply." in stream.getvalue()
