"""Tests for descriptive run directory naming."""

from datetime import datetime

from api.enums import OpenAIModels, Provider
from conversation.agent_config import LLMAgentConfig, RuleBasedAgentConfig
from enums import AgentSelectionMethod, AnalyzerType, FetcherType
from models.configs import (
    AgentConfig,
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
)
from persistence.run_naming import (
    build_run_id,
    build_run_slug,
    enrich_run_meta,
    format_run_timestamp,
    timestamp_from_meta,
)


def _eliza_llm_config() -> ExperimentConfig:
    return ExperimentConfig(
        fetcher_config=FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="data/human-ai_datasets/sharegpt_clean.json",
            min_messages=2,
            max_messages=6,
        ),
        analyzer_config=AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY,
            analyze_window=5,
        ),
        agents=[
            LLMAgentConfig(provider="azure", model="gpt-4o-2024-08-06"),
            RuleBasedAgentConfig(),
        ],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=50,
    )


def test_build_run_slug_includes_agents_fetcher_and_turns():
    slug = build_run_slug(_eliza_llm_config())
    assert "llm-azure-gpt-4o-2024-08-06" in slug
    assert "eliza" in slug
    assert "sharegpt" in slug
    assert slug.endswith("50turns")


def test_build_run_id_appends_short_suffix():
    run_id = build_run_id(_eliza_llm_config())
    slug = build_run_slug(_eliza_llm_config())
    assert run_id.startswith(slug + "_")
    assert len(run_id.split("_")[-1]) == 8


def test_format_run_timestamp_is_human_readable():
    when = datetime(2026, 7, 13, 17, 46)
    assert format_run_timestamp(when) == "Monday, 13 July 2026, 17:46"


def test_enrich_run_meta_adds_iso_and_human_timestamps():
    when = datetime(2026, 7, 13, 17, 46)
    meta = enrich_run_meta({"run_id": "run-1"}, timestamp=when)
    assert meta["timestamp"] == when.isoformat()
    assert meta["timestamp_human"] == "Monday, 13 July 2026, 17:46"


def test_timestamp_from_meta_falls_back_to_iso_timestamp():
    meta = {"timestamp": "2026-07-13T17:46:00"}
    assert timestamp_from_meta(meta) == "Monday, 13 July 2026, 17:46"


def test_build_run_slug_supports_legacy_agent_configs():
    config = ExperimentConfig(
        fetcher_config=FetcherConfig(
            fetcher=FetcherType.RANDOM,
            category="conversational",
        ),
        analyzer_config=AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY,
            analyze_window=2,
        ),
        agent_configs=[
            AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_0125_PREVIEW,
            )
        ],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        max_iterations=10,
    )
    slug = build_run_slug(config)
    assert "llm-openai-gpt-4-0125-preview" in slug
    assert "random" in slug
