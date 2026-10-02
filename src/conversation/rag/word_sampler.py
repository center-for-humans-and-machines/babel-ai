"""Random word retrieval from a pretrained word-vector space.

The sampler draws random vectors in the embedding space and returns the
nearest vocabulary words. This deliberately surfaces arbitrary, loosely
related terms that later seed a web search.

Vectors are loaded lazily from one of three sources:

* an injected ``vectors`` object (tests),
* a local ``model_path`` (a word2vec/GloVe file, optionally gzipped),
* otherwise downloaded from the gensim-data release via ``requests``.

``gensim.downloader`` is intentionally avoided: it uses ``urllib``, which
fails behind TLS-intercepting proxies where ``requests`` succeeds.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "glove-wiki-gigaword-100"
_DOWNLOAD_BASE = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download"
)


class _KeyedVectors(Protocol):
    """Minimal surface shared by ``gensim`` ``KeyedVectors`` objects."""

    vectors: Any
    index_to_key: Sequence[str]


class RandomWordSampler:
    """Draw random vocabulary words from a word2vec-style embedding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        seed: int = 0,
        vectors: Optional[_KeyedVectors] = None,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_timeout: float = 120.0,
        session: Any | None = None,
    ) -> None:
        if not model_name and vectors is None and not model_path:
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name or DEFAULT_MODEL
        self._rng = random.Random(seed)
        self._vectors = vectors
        self._model_path = model_path
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._download_timeout = download_timeout
        self._session = session
        self._unit_matrix: Any | None = None

    def _ensure_loaded(self) -> None:
        """Lazily load vectors without relying on ``gensim.downloader``."""
        if self._vectors is not None:
            return
        from gensim.models import KeyedVectors

        if self._model_path:
            self._vectors = _load_local_vectors(
                Path(self._model_path).expanduser()
            )
            return
        cache_dir = self._cache_dir or Path.home() / "gensim-data"
        path = download_vectors(
            self.model_name,
            cache_dir=cache_dir,
            timeout=self._download_timeout,
            session=self._session,
        )
        self._vectors = KeyedVectors.load_word2vec_format(
            str(path), binary=False
        )

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


def download_vectors(
    model_name: str = DEFAULT_MODEL,
    *,
    cache_dir: Path | str | None = None,
    timeout: float = 120.0,
    session: Any | None = None,
) -> Path:
    """Ensure a gensim-data vector file is cached locally; return its path."""
    import requests

    root = (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path.home() / "gensim-data"
    )
    target = root / model_name / f"{model_name}.gz"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_DOWNLOAD_BASE}/{model_name}/{model_name}.gz"
    logger.info("Downloading word vectors from %s", url)
    client = session or requests.Session()
    response = client.get(url, stream=True, timeout=timeout)
    if response.status_code != 200:
        raise OSError(
            f"vector download failed with status {response.status_code}"
        )
    temporary = target.with_suffix(".gz.part")
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def _load_local_vectors(path: Path) -> _KeyedVectors:
    """Load a local word2vec/GloVe file; ``gensim`` handles ``.gz``."""
    if not path.is_file():
        raise FileNotFoundError(f"word vector file not found: {path}")
    from gensim.models import KeyedVectors

    return KeyedVectors.load_word2vec_format(str(path), binary=False)


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
