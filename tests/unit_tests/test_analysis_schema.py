"""Tests for the flat analysis schema."""

from analysis_schema import AnalysisResult, flatten_analysis, parquet_columns


def test_parquet_columns_match_model_fields():
    assert parquet_columns() == [
        name
        for name in AnalysisResult.model_fields
        if name != "analysis_extra"
    ]


def test_flatten_analysis_namespaces_extra_fields():
    result = flatten_analysis(
        {
            "word_count": 3,
            "analysis_extra": {"novel_metric": 0.7},
            "future_metric": 1,
        }
    )

    assert result["word_count"] == 3
    assert result["semantic_similarity"] is None
    assert result["analysis_extra_novel_metric"] == 0.7
    assert result["analysis_extra_future_metric"] == 1


def test_flatten_analysis_accepts_model_instances():
    result = flatten_analysis(AnalysisResult(word_count=4))

    assert result["word_count"] == 4
    assert set(result) == set(parquet_columns())
