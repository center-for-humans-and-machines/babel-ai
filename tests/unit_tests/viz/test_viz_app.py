"""Tests for the local trajectory viewer."""

import pandas as pd
from fastapi.testclient import TestClient

from persistence import save_run
from viz.app import create_app


def test_health_returns_ok(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runs_lists_completed_run(tmp_path):
    _save_run(tmp_path, "run-1")
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs")

    assert response.status_code == 200
    assert "run-1" in response.text
    assert "2 turns" in response.text


def test_run_detail_shows_similarity_trajectory_and_transcript(tmp_path):
    _save_run(tmp_path, "run-1")
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert "Semantic similarity (window)" in response.text
    assert "Hello, ELIZA." in response.text
    assert "How does that make you feel?" in response.text


def test_run_detail_falls_back_to_turn_index(tmp_path):
    _save_run(tmp_path, "run-1", similarity=None)
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert "Turn index" in response.text


def test_run_detail_returns_not_found_for_unknown_run(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/missing")

    assert response.status_code == 404


def test_run_detail_returns_not_found_for_incomplete_run(tmp_path):
    (tmp_path / "incomplete").mkdir()
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/incomplete")

    assert response.status_code == 404


def _save_run(
    results_root, run_id: str, similarity: float | None = 0.75
) -> None:
    """Save a two-turn run used by viewer requests."""
    turns = pd.DataFrame(
        {
            "turn_index": [0, 1],
            "role": ["user", "assistant"],
            "speaker": ["user", "eliza"],
            "content": ["Hello, ELIZA.", "How does that make you feel?"],
            "analysis": [
                {"semantic_similarity_window": similarity},
                {"semantic_similarity_window": similarity},
            ],
        }
    )
    save_run(results_root / run_id, turns, {"run_id": run_id})
