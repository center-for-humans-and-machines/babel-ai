"""Per-run t-SNE projection of analysis-scoped turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from analysis_scope import AnalysisScope, select_analysis_turns

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_OUTPUT_COLUMNS = (
    "turn_index",
    "trajectory_index",
    "speaker",
    "content",
    "tsne_x",
    "tsne_y",
)


class EmbeddingProvider(Protocol):
    """Encode text into a dense vector space."""

    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one embedding row per input text."""
        ...


class SentenceTransformerEmbedder:
    """Lazy sentence-transformer embedding provider."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with the configured sentence transformer."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float)


@dataclass(frozen=True)
class ProjectionResult:
    """Coordinates and reproducibility metadata for one run."""

    trajectory: pd.DataFrame
    meta: dict[str, object]


def select_llm_turns(turns: pd.DataFrame) -> pd.DataFrame:
    """Return generated LLM turns in stable conversation order."""
    return select_analysis_turns(turns, AnalysisScope.LLM_ONLY)


def adaptive_perplexity(point_count: int) -> float:
    """Choose a conservative perplexity below the sample count."""
    if point_count < 3:
        raise ValueError("t-SNE requires at least three analysis turns")
    return min(30.0, max(2.0, (point_count - 1) / 3))


def analyze_tsne(
    turns: pd.DataFrame,
    *,
    embedder: EmbeddingProvider | None = None,
    random_state: int = 0,
    scope: AnalysisScope = AnalysisScope.LLM_ONLY,
) -> ProjectionResult:
    """Embed scoped turns and project them independently."""
    selected = select_analysis_turns(turns, scope)
    point_count = len(selected)
    provider = embedder or SentenceTransformerEmbedder()
    base_meta: dict[str, object] = {
        "embedding_model": provider.model_name,
        "random_state": random_state,
        "point_count": point_count,
        "analysis_scope": scope.value,
    }
    if point_count < 3:
        reason = (
            "t-SNE requires at least three generated LLM turns"
            if scope is AnalysisScope.LLM_ONLY
            else "t-SNE requires at least three turns"
        )
        return ProjectionResult(
            trajectory=_empty_trajectory(),
            meta={
                **base_meta,
                "status": "unavailable",
                "reason": reason,
                "perplexity": None,
            },
        )

    embeddings = provider.encode(selected["content"].astype(str).tolist())
    if embeddings.ndim != 2 or embeddings.shape[0] != point_count:
        raise ValueError("embedder returned an invalid matrix shape")
    perplexity = adaptive_perplexity(point_count)
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    ).fit_transform(embeddings)
    trajectory = selected.copy()
    trajectory["tsne_x"] = coordinates[:, 0]
    trajectory["tsne_y"] = coordinates[:, 1]
    return ProjectionResult(
        trajectory=trajectory.loc[:, list(_OUTPUT_COLUMNS)],
        meta={
            **base_meta,
            "status": "complete",
            "reason": None,
            "perplexity": perplexity,
        },
    )


def _empty_trajectory() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
