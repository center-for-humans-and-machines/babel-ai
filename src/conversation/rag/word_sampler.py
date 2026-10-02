"""Random word retrieval from a pretrained word-vector space.

The sampler draws random vectors in the embedding space and returns the
nearest vocabulary words. This deliberately surfaces arbitrary, loosely
related terms that later seed a web search.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Protocol, Sequence


class _KeyedVectors(Protocol):
    """Minimal surface shared by ``gensim`` ``KeyedVectors`` objects."""

    vectors: Any
    index_to_key: Sequence[str]


class RandomWordSampler:
    """Draw random vocabulary words from a word2vec-style embedding."""

    def __init__(
        self,
        model_name: str = "glove-wiki-gigaword-100",
        *,
        seed: int = 0,
        vectors: Optional[_KeyedVectors] = None,
    ) -> None:
        if not model_name and vectors is None:
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self._rng = random.Random(seed)
        self._vectors = vectors
        self._unit_matrix: Any | None = None

    def _ensure_loaded(self) -> None:
        """Lazily load the pretrained vectors from ``gensim``."""
        if self._vectors is not None:
            return
        import gensim.downloader as api  # imported lazily: heavy optional dep

        self._vectors = api.load(self.model_name)

    def _ensure_matrix(self) -> tuple[Any, Sequence[str]]:
        if self._unit_matrix is None:
            import numpy as np

            assert self._vectors is not None
            matrix = np.asarray(self._vectors.vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._unit_matrix = matrix / np.clip(norms, 1e-12, None)
        return self._unit_matrix, list(self._vectors.index_to_key)

    def sample(self, count: int) -> list[str]:
        """Return ``count`` vocabulary words nearest to random vectors."""
        if count < 1:
            raise ValueError("count must be >= 1")
        self._ensure_loaded()
        unit, keys = self._ensure_matrix()
        dim = int(unit.shape[1])
        words: list[str] = []
        for _ in range(count):
            vector = self._random_unit_vector(dim)
            index = int((unit @ vector).argmax())
            words.append(keys[index])
        return words

    def _random_unit_vector(self, dim: int) -> Any:
        import numpy as np

        vector = np.fromiter(
            (self._rng.gauss(0.0, 1.0) for _ in range(dim)),
            dtype=np.float32,
            count=dim,
        )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # pragma: no cover - vanishingly unlikely
            vector = np.ones(dim, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
        return vector / norm

    def export_state(self) -> dict[str, Any]:
        """Serialize RNG state for exact checkpoint recovery."""
        return {"rng_state": _encode_rng_state(self._rng.getstate())}

    def import_state(self, data: dict[str, Any]) -> None:
        """Restore RNG state from a checkpoint."""
        if "rng_state" in data:
            self._rng.setstate(_decode_rng_state(data["rng_state"]))


def _encode_rng_state(state: tuple) -> dict[str, Any]:
    version, internal, gauss_next = state
    return {
        "version": int(version),
        "internal": [int(value) for value in internal],
        "gauss_next": gauss_next,
    }


def _decode_rng_state(payload: Any) -> tuple:
    if isinstance(payload, dict):
        return (
            int(payload["version"]),
            tuple(int(value) for value in payload["internal"]),
            payload.get("gauss_next"),
        )
    version, internal, gauss_next = payload
    return int(version), tuple(int(value) for value in internal), gauss_next
