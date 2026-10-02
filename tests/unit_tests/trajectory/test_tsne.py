"""Tests for independent LLM t-SNE projection."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis_scope import AnalysisScope
from trajectory.tsne import (
    adaptive_perplexity,
    analyze_tsne,
    select_llm_turns,
    SentenceTransformerEmbedder,
)


class FakeEmbedder:
    """Return stable vectors without loading an embedding model."""

    model_name = "fake-embeddings"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [index, index**2, len(text), index + len(text)]
                for index, text in enumerate(texts)
            ],
            dtype=float,
        )


def _turns(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn_index": list(range(count)),
            "speaker": ["agent_0"] * count,
            "content": [f"Generated turn {index}" for index in range(count)],
            "agent_config": ['{"model": "test"}'] * count,
        }
    )


def test_select_llm_turns_excludes_seed_and_rule_agents():
    turns = pd.DataFrame(
        {
            "turn_index": [3, 0, 2, 1],
            "speaker": ["agent_0", "seed", "eliza", "agent_0"],
            "content": ["later", "seed", "prompt", "earlier"],
            "agent_config": ["{}", None, None, "{}"],
        }
    )
    selected = select_llm_turns(turns)
    assert selected["turn_index"].tolist() == [1, 3]
    assert selected["trajectory_index"].tolist() == [0, 1]


def test_select_llm_turns_handles_missing_schema():
    selected = select_llm_turns(pd.DataFrame({"content": ["missing"]}))
    assert selected.empty
    assert "trajectory_index" in selected.columns


def test_adaptive_perplexity_is_valid_and_bounded():
    assert adaptive_perplexity(3) == 2.0
    assert adaptive_perplexity(1000) == 30.0
    with pytest.raises(ValueError, match="at least three"):
        adaptive_perplexity(2)


def test_projection_is_deterministic_and_ordered():
    first = analyze_tsne(_turns(), embedder=FakeEmbedder(), random_state=9)
    second = analyze_tsne(_turns(), embedder=FakeEmbedder(), random_state=9)
    assert first.meta["status"] == "complete"
    assert first.meta["embedding_model"] == "fake-embeddings"
    assert first.trajectory["trajectory_index"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(
        first.trajectory[["tsne_x", "tsne_y"]],
        second.trajectory[["tsne_x", "tsne_y"]],
    )


def test_projection_reports_insufficient_points_without_embedding():
    class FailingEmbedder(FakeEmbedder):
        def encode(self, texts: list[str]) -> np.ndarray:
            raise AssertionError("encode must not run")

    result = analyze_tsne(_turns(2), embedder=FailingEmbedder())
    assert result.meta["status"] == "unavailable"
    assert result.meta["point_count"] == 2
    assert result.trajectory.empty


def test_projection_rejects_invalid_embedding_shape():
    class InvalidEmbedder(FakeEmbedder):
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.ones((1, 2))

    with pytest.raises(ValueError, match="matrix shape"):
        analyze_tsne(_turns(), embedder=InvalidEmbedder())


def test_projection_all_turns_includes_partner_rows():
    turns = pd.DataFrame(
        {
            "turn_index": [0, 1, 2, 3],
            "speaker": ["seed", "eliza", "agent_0", "agent_0"],
            "content": ["seed", "partner", "one", "two"],
            "agent_config": [None, None, "{}", "{}"],
        }
    )
    result = analyze_tsne(
        turns,
        embedder=FakeEmbedder(),
        scope=AnalysisScope.ALL_TURNS,
    )
    assert result.meta["status"] == "complete"
    assert result.meta["analysis_scope"] == "all_turns"
    assert len(result.trajectory) == 4


def test_sentence_transformer_embedder_loads_model_lazily(monkeypatch):
    loads = []

    class FakeModel:
        def __init__(self, model_name):
            loads.append(model_name)

        def encode(self, texts, **kwargs):
            return [[len(text), 1.0] for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    embedder = SentenceTransformerEmbedder("fake-model")
    first = embedder.encode(["one", "three"])
    second = embedder.encode(["again"])
    assert loads == ["fake-model"]
    assert first.shape == (2, 2)
    assert second.shape == (1, 2)
