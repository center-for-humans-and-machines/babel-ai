"""Tests for analysis turn selection."""

from datetime import datetime

import pandas as pd

from api.enums import OpenAIModels, Provider
from analysis_scope import (
    AnalysisScope,
    analysis_contents_for_metrics,
    is_llm_turn_value,
    resolve_analysis_scope,
    select_analysis_turns,
    trajectory_plot_frame,
)
from models.configs import AgentConfig
from models.metrics import AgentMetric


def _turns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn_index": [3, 0, 2, 1],
            "speaker": ["agent_0", "seed", "eliza", "agent_0"],
            "content": ["later", "seed", "prompt", "earlier"],
            "agent_config": ['{"model": "test"}', None, None, "{}"],
        }
    )


def test_is_llm_turn_value_accepts_config_markers():
    assert is_llm_turn_value('{"model": "gpt-4"}')
    assert is_llm_turn_value({})
    assert not is_llm_turn_value(None)
    assert not is_llm_turn_value("")


def test_select_analysis_turns_defaults_to_llm_only():
    selected = select_analysis_turns(_turns_frame())
    assert selected["turn_index"].tolist() == [1, 3]
    assert selected["trajectory_index"].tolist() == [0, 1]


def test_select_analysis_turns_all_turns_keeps_every_row():
    selected = select_analysis_turns(
        _turns_frame(),
        AnalysisScope.ALL_TURNS,
    )
    assert selected["turn_index"].tolist() == [0, 1, 2, 3]
    assert selected["trajectory_index"].tolist() == [0, 1, 2, 3]


def test_resolve_analysis_scope_defaults_to_llm_only():
    assert resolve_analysis_scope({}) is AnalysisScope.LLM_ONLY
    assert (
        resolve_analysis_scope(
            {"config": {"analyzer_config": {"analysis_scope": "all_turns"}}}
        )
        is AnalysisScope.ALL_TURNS
    )


def test_analysis_contents_for_metrics_skips_partner_turns():
    metrics = [
        AgentMetric(
            iteration=0,
            timestamp=datetime.now(),
            role="seed",
            content="seed",
            agent_id="seed",
            agent_config=None,
        ),
        AgentMetric(
            iteration=1,
            timestamp=datetime.now(),
            role="eliza",
            content="partner",
            agent_id="eliza",
            agent_config=None,
        ),
        AgentMetric(
            iteration=2,
            timestamp=datetime.now(),
            role="agent_0",
            content="llm one",
            agent_id="llm",
            agent_config=AgentConfig(
                provider=Provider.OPENAI,
                model=OpenAIModels.GPT4_0125_PREVIEW,
            ),
        ),
    ]
    assert analysis_contents_for_metrics(
        metrics,
        scope=AnalysisScope.LLM_ONLY,
        through_index=1,
    ) is None
    assert analysis_contents_for_metrics(
        metrics,
        scope=AnalysisScope.LLM_ONLY,
        through_index=2,
    ) == ["llm one"]
    assert analysis_contents_for_metrics(
        metrics,
        scope=AnalysisScope.ALL_TURNS,
        through_index=2,
    ) == ["seed", "partner", "llm one"]


def test_trajectory_plot_frame_assigns_plot_index():
    frame = trajectory_plot_frame(_turns_frame())
    assert frame["plot_index"].tolist() == [0, 1]
