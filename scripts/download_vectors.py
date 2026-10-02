"""Download pretrained word vectors for the RAG scaffolder.

Uses ``requests`` (not ``gensim.downloader``) so it works behind
TLS-intercepting proxies. Vectors are cached under ``~/gensim-data``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from conversation.rag.word_sampler import (
    DEFAULT_MODEL,
    download_vectors,
)  # noqa: E402


def main() -> None:
    """Cache a vector file and print its path."""
    parser = argparse.ArgumentParser(
        description="Download word vectors for the RAG scaffolder",
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: ~/gensim-data)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    path = download_vectors(
        args.model,
        cache_dir=args.cache_dir,
        timeout=args.timeout,
    )
    print(f"{args.model}: {path}")


if __name__ == "__main__":
    main()
