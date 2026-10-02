"""Tests for the RAG word-vector loader."""

import gzip

import pytest

from conversation.rag.word_sampler import (
    RandomWordSampler,
    download_vectors,
)


class _FakeStreamResponse:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]


class _FakeDownloadSession:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self._payload = payload
        self._status = status_code
        self.urls: list[str] = []

    def get(self, url, stream=False, timeout=None):
        self.urls.append(url)
        return _FakeStreamResponse(self._payload, self._status)


def _vectors_file(tmp_path):
    path = tmp_path / "vectors.txt"
    path.write_text(
        "3 3\nnorth 1 0 0\neast 0 1 0\nup 0 0 1\n",
        encoding="utf-8",
    )
    return path


def test_sampler_loads_local_model_path(tmp_path):
    sampler = RandomWordSampler(
        model_path=str(_vectors_file(tmp_path)), seed=5
    )
    words = sampler.sample(4)
    assert len(words) == 4
    assert set(words) <= {"north", "east", "up"}


def test_sampler_loads_gzipped_local_model_path(tmp_path):
    path = tmp_path / "vectors.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("3 3\nnorth 1 0 0\neast 0 1 0\nup 0 0 1\n")
    sampler = RandomWordSampler(model_path=str(path), seed=1)
    assert set(sampler.sample(3)) <= {"north", "east", "up"}


def test_sampler_missing_local_model_raises(tmp_path):
    sampler = RandomWordSampler(
        model_path=str(tmp_path / "missing.txt"), seed=0
    )
    with pytest.raises(FileNotFoundError):
        sampler.sample(1)


def test_download_vectors_writes_cache_and_reuses_it(tmp_path):
    payload = gzip.compress(b"2 2\nnorth 1 0\neast 0 1\n")
    session = _FakeDownloadSession(payload)
    path = download_vectors("tiny", cache_dir=tmp_path, session=session)
    assert path.is_file()
    assert path.stat().st_size > 0
    assert session.urls and "tiny" in session.urls[0]

    cached = _FakeDownloadSession(b"ignored")
    again = download_vectors("tiny", cache_dir=tmp_path, session=cached)
    assert again == path
    assert cached.urls == []


def test_download_vectors_raises_on_error_status(tmp_path):
    session = _FakeDownloadSession(b"", status_code=404)
    with pytest.raises(OSError):
        download_vectors("missing", cache_dir=tmp_path, session=session)


def test_downloaded_vectors_load_into_sampler(tmp_path):
    payload = gzip.compress(b"3 3\nnorth 1 0 0\neast 0 1 0\nup 0 0 1\n")
    download_vectors(
        "tiny",
        cache_dir=tmp_path,
        session=_FakeDownloadSession(payload),
    )
    sampler = RandomWordSampler(
        model_name="tiny",
        cache_dir=str(tmp_path),
        seed=2,
    )
    assert set(sampler.sample(3)) <= {"north", "east", "up"}
