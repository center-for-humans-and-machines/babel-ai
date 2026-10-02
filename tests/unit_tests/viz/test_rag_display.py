"""Tests for RAG scaffolder display helpers."""

import json

import pandas as pd

from viz.rag_display import (
    extract_rag_agent_config,
    has_rag_data,
    rag_source_domains,
    rag_status,
    rag_turn_rows,
)


def _frame(**overrides):
    base = {
        "turn_index": [0, 1, 2, 3],
        "speaker": ["agent_0", "rag_scaffolder", "rag_scaffolder", "llm"],
        "content": ["Hello", "Grounded nudge", "Fallback nudge", "World"],
        "rag_used": [None, True, False, None],
        "rag_mode": [None, "novelty", "topic", None],
        "rag_words": [
            None,
            json.dumps(["ocean", "heat"]),
            None,
            None,
        ],
        "rag_query": [None, "ocean heat", None, None],
        "rag_source_url": [
            None,
            "https://en.wikipedia.org/wiki/Ocean",
            None,
            None,
        ],
        "rag_source_title": [None, "Ocean", None, None],
        "rag_fallback_reason": [
            None,
            None,
            "search returned no results",
            None,
        ],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_extract_rag_agent_config_reads_rag_scaffolder():
    meta = {
        "config": {
            "agents": [
                {"type": "llm", "provider": "azure", "model": "gpt-4o"},
                {
                    "type": "rag_scaffolder",
                    "word_model": "glove-wiki-gigaword-100",
                    "num_words": 5,
                    "search_backend": "auto",
                    "search_results": 5,
                    "novelty_nudge_rate": 0.2,
                },
            ]
        }
    }
    config = extract_rag_agent_config(meta)
    assert config["word_model"] == "glove-wiki-gigaword-100"
    assert config["num_words"] == "5"
    assert config["search_backend"] == "auto"


def test_has_rag_data_requires_values():
    assert has_rag_data(_frame())
    without = _frame(
        rag_used=[None, None, None, None],
        rag_mode=[None] * 4,
        rag_words=[None] * 4,
        rag_query=[None] * 4,
        rag_source_url=[None] * 4,
        rag_source_title=[None] * 4,
        rag_fallback_reason=[None] * 4,
    )
    assert not has_rag_data(without)


def test_rag_status_counts_grounded_and_fallbacks():
    status = rag_status(_frame())
    assert "Grounded 1 of 2 RAG turns" in status
    assert "1 fallbacks" in status
    assert "search returned no results" in status


def test_rag_status_without_rag_turns():
    frame = pd.DataFrame({"speaker": ["agent_0"], "content": ["hi"]})
    assert "No RAG scaffolder turns" in rag_status(frame)


def test_rag_turn_rows_parses_words_and_status():
    rows = rag_turn_rows(_frame())
    assert len(rows) == 2
    grounded, fallback = rows
    assert grounded["status"] == "grounded"
    assert grounded["rag_words"] == "ocean, heat"
    assert grounded["rag_source_url"].startswith("https://")
    assert fallback["status"] == "fallback"
    assert fallback["rag_words"] == ""


def test_rag_source_domains_counts_hosts():
    domains = rag_source_domains(_frame())
    assert domains == [{"domain": "en.wikipedia.org", "count": 1}]
