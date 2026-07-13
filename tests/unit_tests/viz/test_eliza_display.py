"""Tests for ELIZA branch display helpers."""

import pandas as pd

from viz.eliza_display import (
    eliza_branch_status,
    eliza_turn_rows,
    extract_eliza_agent_config,
    has_eliza_branch_data,
)


def test_extract_eliza_agent_config_reads_rule_based_agent():
    meta = {
        "config": {
            "agents": [
                {"type": "llm", "provider": "azure", "model": "gpt-4o"},
                {
                    "type": "rule_based",
                    "partner": "eliza",
                    "generic_intervention": "passthrough",
                },
            ]
        }
    }
    assert extract_eliza_agent_config(meta) == {
        "partner": "eliza",
        "generic_intervention": "passthrough",
    }


def test_has_eliza_branch_data_requires_eliza_turn_values():
    frame = pd.DataFrame(
        {
            "speaker": ["eliza", "agent_0"],
            "eliza_branch": ["keyword:like", None],
        }
    )
    assert has_eliza_branch_data(frame)


def test_eliza_branch_status_warns_when_column_missing():
    frame = pd.DataFrame(
        {
            "speaker": ["eliza", "agent_0"],
            "content": ["Hi", "Hello"],
        }
    )
    status = eliza_branch_status(frame)
    assert "unavailable" in status


def test_eliza_turn_rows_returns_only_eliza_entries():
    frame = pd.DataFrame(
        {
            "turn_index": [1, 2],
            "speaker": ["agent_0", "eliza"],
            "eliza_branch": [None, "generic:$"],
            "eliza_keyword": [None, ""],
            "eliza_reassembly": [None, ""],
            "used_generic_fallback": [None, True],
            "content": ["Hello", "Please go on."],
        }
    )
    rows = eliza_turn_rows(frame)
    assert len(rows) == 1
    assert rows[0]["eliza_branch"] == "generic:$"
