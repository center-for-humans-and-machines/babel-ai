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


def test_run_detail_shows_eliza_branch_columns(tmp_path):
    _save_run(tmp_path, "run-1", eliza_branch="keyword:alike")
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert "ELIZA branch" in response.text
    assert "keyword:alike" in response.text
    assert "ELIZA branch counts" in response.text


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


def test_compare_overlay_and_aggregate(tmp_path):
    _save_run(tmp_path, "run-a", similarity=0.5, index=3)
    _save_run(tmp_path, "run-b", similarity=0.8, index=5)
    client = TestClient(create_app(tmp_path))

    response = client.get("/compare?runs=run-a,run-b&baseline=run-a")

    assert response.status_code == 200
    assert "Overlay" in response.text
    assert "Aggregate mean" in response.text
    assert "Config diff" in response.text


def test_runs_page_links_to_compare(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs")

    assert response.status_code == 200
    assert "/compare" in response.text


def _save_run(
    results_root,
    run_id: str,
    similarity: float | None = 0.75,
    index: int = 0,
    eliza_branch: str | None = None,
) -> None:
    """Save a two-turn run used by viewer requests."""
    turns = pd.DataFrame(
        {
            "turn_index": [0, 1],
            "role": ["user", "assistant"],
            "speaker": ["user", "eliza"],
            "content": ["Hello, ELIZA.", "How does that make you feel?"],
            "eliza_branch": ["", eliza_branch or ""],
            "eliza_reassembly": ["", "WHAT DOES THAT SUGGEST TO YOU"],
            "analysis": [
                {"semantic_similarity_window": similarity},
                {"semantic_similarity_window": similarity},
            ],
        }
    )
    save_run(
        results_root / run_id,
        turns,
        {
            "run_id": run_id,
            "run_slug": "eliza_sharegpt_2turns",
            "config": {"max_iterations": index},
        },
    )
