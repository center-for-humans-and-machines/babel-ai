"""Tests for shared analysis cell parsing."""

from persistence.analysis_parse import parse_analysis_cell


def test_parse_analysis_cell_json():
    raw = '{"word_count": 3, "coherence_score": 1.0}'
    assert parse_analysis_cell(raw)["word_count"] == 3


def test_parse_analysis_cell_python_repr():
    raw = "{'word_count': 5}"
    assert parse_analysis_cell(raw)["word_count"] == 5
