"""Parse analysis blobs stored in legacy CSV cells."""

from __future__ import annotations

import json
from ast import literal_eval
from typing import Any


def parse_analysis_cell(raw: Any) -> dict[str, Any]:
    """Parse a serialized analysis dictionary without using ``eval``."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}

    text = raw.strip()
    text = text.replace("None", "null").replace("inf", "Infinity")
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    try:
        data = literal_eval(raw)
        if isinstance(data, dict):
            return data
    except (SyntaxError, ValueError):
        return {}
    return {}
