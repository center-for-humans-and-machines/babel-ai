"""Tests for the local trajectory viewer."""

import pandas as pd
from fastapi.testclient import TestClient

from persistence import save_run
from trajectory.artifacts import save_projection
from trajectory.tsne import ProjectionResult
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
    assert "3 turns" in response.text
    assert "Tracked" in response.text


def test_run_detail_shows_eliza_branch_columns(tmp_path):
    _save_run(tmp_path, "run-1", eliza_branch="keyword:like")
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert "ELIZA setup" in response.text
    assert "Started:" in response.text
    assert "Monday, 13 July 2026, 17:46" in response.text
    assert "ELIZA decision paths" in response.text
    assert "keyword:like" in response.text
    assert "Branch counts" in response.text


def test_run_detail_shows_grouped_analysis_trajectories_and_transcript(
    tmp_path,
):
    _save_run(tmp_path, "run-1")
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert "Analysis trajectories" in response.text
    assert "Semantic similarity" in response.text
    assert "Lexical similarity" in response.text
    assert "Token perplexity" in response.text
    assert "Semantic similarity (window)" in response.text
    assert "Semantic similarity (direct)" in response.text
    assert "Hello, ELIZA." in response.text
    assert "How does that make you feel?" in response.text


def test_run_detail_falls_back_to_turn_index(tmp_path):
    _save_run(tmp_path, "run-1", similarity=None)
    client = TestClient(create_app(tmp_path))

    response = client.get("/runs/run-1")

    assert response.status_code == 200
    assert "Turn index" in response.text


def test_run_detail_shows_saved_tsne_topic_cloud(tmp_path):
    _save_run(tmp_path, "run-1")
    trajectory = pd.DataFrame(
        {
            "turn_index": [1, 3, 5],
            "trajectory_index": [0, 1, 2],
            "speaker": ["agent_0"] * 3,
            "content": ["alpha", "beta", "gamma"],
            "tsne_x": [0.0, 1.0, 0.5],
            "tsne_y": [0.0, 0.5, 1.0],
        }
    )
    save_projection(
        tmp_path / "run-1",
        ProjectionResult(
            trajectory=trajectory,
            meta={"status": "complete", "point_count": 3},
        ),
    )
    client = TestClient(create_app(tmp_path))
    response = client.get("/runs/run-1")
    assert response.status_code == 200
    assert "LLM topic cloud and trajectory" in response.text
    assert "Projected 3 generated LLM turns." in response.text


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
    """Save a short multi-turn run used by viewer requests."""
    turns = pd.DataFrame(
        {
            "turn_index": [0, 1, 2],
            "role": ["user", "assistant", "assistant"],
            "speaker": ["seed", "eliza", "agent_0"],
            "content": [
                "Hello, ELIZA.",
                "How does that make you feel?",
                "I understand.",
            ],
            "agent_config": [
                None,
                None,
                '{"provider": "azure", "model": "gpt-4o-2024-08-06"}',
            ],
            "eliza_branch": ["", eliza_branch or "", ""],
            "eliza_reassembly": [
                "",
                "WHAT DOES THAT SUGGEST TO YOU",
                "",
            ],
            "semantic_similarity_window": [None, None, similarity],
            "semantic_similarity": [None, None, _scaled(similarity, 0.9)],
            "lexical_similarity_window": [
                None,
                None,
                _scaled(similarity, 0.8),
            ],
            "lexical_similarity": [None, None, _scaled(similarity, 0.7)],
            "token_perplexity": [
                None,
                None,
                12.5 if similarity is not None else None,
            ],
        }
    )
    save_run(
        results_root / run_id,
        turns,
        {
            "run_id": run_id,
            "run_slug": "eliza_sharegpt_2turns",
            "timestamp_human": "Monday, 13 July 2026, 17:46",
            "config": {
                "max_iterations": index,
                "agents": [
                    {"type": "llm", "provider": "azure", "model": "gpt-4o"},
                    {
                        "type": "rule_based",
                        "partner": "eliza",
                        "generic_intervention": "passthrough",
                    },
                ],
                "analyzer_config": {
                    "analyzer": "similarity",
                    "analyze_window": 5,
                },
            },
        },
    )


def _scaled(value: float | None, factor: float) -> float | None:
    if value is None:
        return None
    return round(value * factor, 3)
