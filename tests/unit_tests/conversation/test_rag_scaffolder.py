"""Tests for the retrieval-augmented (RAG) scaffolder."""

import json
from datetime import datetime

import numpy as np
import pytest

from conversation.agent_config import LLMAgentConfig, RagScaffolderAgentConfig
from conversation.agents import (
    AgentTurn,
    RagScaffolderConversationAgent,
    _scaffolder_turn_to_agent_turn,
)
from conversation.checkpoint import (
    CheckpointWriter,
    ConversationState,
)
from conversation.factory import (
    _legacy_agent_config,
    build_conversation_agents,
)
from conversation.messages import (
    ContextStack,
    ConversationMessage,
    MessageSource,
)
from conversation.rag.nudge import (
    RagMode,
    RagNudge,
    RagNudgeProvider,
    RagUnavailable,
)
from conversation.rag.scaffolder import RagScaffolder
from conversation.rag.search import (
    DuckDuckGoSearchClient,
    FallbackSearchClient,
    SearchResult,
    WikipediaSearchClient,
    build_search_client,
    fetch_page_text,
)
from conversation.rag.word_sampler import RandomWordSampler
from conversation.scaffolder import (
    DecisionScores,
    NoveltyNudgeKind,
    ScaffolderAction,
    ScaffolderTurn,
)
from enums import AgentSelectionMethod, AnalyzerType, FetcherType
from experiment import Experiment
from models.configs import (
    AnalyzerConfig,
    ExperimentConfig,
    FetcherConfig,
)
from models.metrics import AgentMetric
from persistence.run_store import load_run, save_run


def _message(index: int, content: str, speaker: str = "llm"):
    return ConversationMessage(
        turn_index=index,
        role="assistant",
        speaker=speaker,
        content=content,
        source=MessageSource.AGENT,
    )


def _legacy_llm_config():
    return _legacy_agent_config(
        LLMAgentConfig(provider="openai", model="gpt-4-0125-preview")
    )


class _FakeHTTPResponse:
    def __init__(
        self, *, text: str = "", status_code: int = 200, payload=None
    ):
        self.text = text
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload configured")
        return self._payload


class _FakeSession:
    def __init__(self, *, post_response=None, get_response=None) -> None:
        self._post_response = post_response or _FakeHTTPResponse()
        self._get_response = get_response or _FakeHTTPResponse()
        self.post_calls: list = []
        self.get_calls: list = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._post_response

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._get_response


class _FakeVectors:
    """Three orthogonal unit vectors standing in for a pretrained model."""

    def __init__(self) -> None:
        self.vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.index_to_key = ["north", "east", "up"]


class _StubSampler:
    def __init__(self, words) -> None:
        self._words = list(words)

    def sample(self, count: int) -> list[str]:
        return self._words[:count]

    def export_state(self) -> dict:
        return {"stub": True}

    def import_state(self, data: dict) -> None:
        self._imported = data


class _StubSearch:
    def __init__(self, results) -> None:
        self._results = list(results)
        self.queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.queries.append(query)
        return self._results[:top_k]


class _NarrowingSearch:
    """Return hits only for single-word queries."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.queries.append(query)
        if " " in query:
            return []
        return [SearchResult("Hit", "https://x.test/hit", "snippet")]


class _StubProvider:
    def __init__(
        self, *, content: str = "Grounded nudge.", error=None
    ) -> None:
        self._content = content
        self._error = error
        self.modes: list[RagMode] = []

    def produce(self, mode, messages, *, speaker):
        self.modes.append(mode)
        if self._error is not None:
            raise self._error
        return RagNudge(
            content=self._content,
            mode=mode,
            words=("alpha", "beta"),
            query="alpha beta",
            source_url="https://source.test/page",
            source_title="Source title",
        )

    def export_state(self) -> dict:
        return {"stub": True}

    def import_state(self, data: dict) -> None:
        self._imported = data


# --- word sampler ---------------------------------------------------------


def test_random_word_sampler_returns_vocabulary_words():
    sampler = RandomWordSampler(vectors=_FakeVectors(), seed=5)
    words = sampler.sample(4)
    assert len(words) == 4
    assert set(words) <= {"north", "east", "up"}


def test_random_word_sampler_is_seed_deterministic():
    left = RandomWordSampler(vectors=_FakeVectors(), seed=11).sample(6)
    right = RandomWordSampler(vectors=_FakeVectors(), seed=11).sample(6)
    assert left == right


def test_random_word_sampler_rejects_zero_count():
    with pytest.raises(ValueError):
        RandomWordSampler(vectors=_FakeVectors()).sample(0)


def test_random_word_sampler_state_round_trip():
    sampler = RandomWordSampler(vectors=_FakeVectors(), seed=3)
    sampler.sample(2)
    state = sampler.export_state()
    expected = sampler.sample(3)
    restored = RandomWordSampler(vectors=_FakeVectors(), seed=99)
    restored.import_state(state)
    assert restored.sample(3) == expected


# --- search ---------------------------------------------------------------

_DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Example A</a>
  <div class="result__snippet">Snippet A</div>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/b">Example B</a>
  <div class="result__snippet">Snippet B</div>
</div>
</body></html>
"""


def test_duckduckgo_search_parses_and_unwraps_urls():
    session = _FakeSession(
        post_response=_FakeHTTPResponse(text=_DDG_HTML, status_code=200)
    )
    client = DuckDuckGoSearchClient(session=session)
    results = client.search("query", 5)
    assert [result.title for result in results] == ["Example A", "Example B"]
    assert results[0].url == "https://example.com/a"
    assert results[1].url == "https://example.org/b"
    assert results[0].snippet == "Snippet A"


def test_duckduckgo_search_respects_top_k():
    session = _FakeSession(
        post_response=_FakeHTTPResponse(text=_DDG_HTML, status_code=200)
    )
    client = DuckDuckGoSearchClient(session=session)
    assert len(client.search("q", 1)) == 1


def test_duckduckgo_search_returns_empty_on_challenge():
    session = _FakeSession(
        post_response=_FakeHTTPResponse(text="challenge", status_code=202)
    )
    client = DuckDuckGoSearchClient(session=session)
    assert client.search("q", 5) == []


def test_wikipedia_search_parses_results():
    payload = {
        "query": {
            "search": [
                {
                    "title": "Ocean current",
                    "snippet": '<span class="searchmatch">Ocean</span> flow',
                }
            ]
        }
    }
    session = _FakeSession(
        get_response=_FakeHTTPResponse(payload=payload, status_code=200)
    )
    client = WikipediaSearchClient(session=session)
    results = client.search("ocean", 3)
    assert results[0].title == "Ocean current"
    assert results[0].url == "https://en.wikipedia.org/wiki/Ocean_current"
    assert results[0].snippet == "Ocean flow"


def test_fallback_search_uses_next_backend_when_first_is_empty():
    empty = _StubSearch([])
    results = [
        SearchResult("T", "https://x.test/r", "snippet"),
    ]
    fallback = FallbackSearchClient([empty, _StubSearch(results)])
    assert fallback.search("q", 3)[0].url == "https://x.test/r"


def test_build_search_client_selects_backend():
    assert isinstance(build_search_client("wikipedia"), WikipediaSearchClient)
    assert isinstance(
        build_search_client("duckduckgo"), DuckDuckGoSearchClient
    )
    assert isinstance(build_search_client("auto"), FallbackSearchClient)


def test_fetch_page_text_strips_scripts_and_styles():
    html = (
        "<html><head><style>.x{}</style></head><body>"
        "<script>bad()</script><p>Hello world</p></body></html>"
    )
    session = _FakeSession(
        get_response=_FakeHTTPResponse(text=html, status_code=200)
    )
    text = fetch_page_text("https://x.test", session=session, max_chars=100)
    assert "Hello world" in text
    assert "bad()" not in text


def test_fetch_page_text_raises_on_non_200():
    session = _FakeSession(
        get_response=_FakeHTTPResponse(text="nope", status_code=404)
    )
    with pytest.raises(OSError):
        fetch_page_text("https://x.test", session=session)


# --- nudge provider -------------------------------------------------------


def test_rag_nudge_provider_grounds_on_fetched_page():
    captured = {}

    def llm_fn(**kwargs):
        captured.update(kwargs)
        return "Ground the discussion in reservoir computing."

    provider = RagNudgeProvider(
        sampler=_StubSampler(["quantum", "tides"]),
        search_client=_StubSearch(
            [SearchResult("Reservoirs", "https://x.test/r", "snippet")]
        ),
        llm_config=_legacy_llm_config(),
        num_words=2,
        top_k=3,
        page_fetcher=lambda url, **kw: "Full page body about reservoirs",
        llm_fn=llm_fn,
        seed=1,
    )
    nudge = provider.produce(
        RagMode.NOVELTY,
        [_message(0, "We were discussing water.")],
        speaker="rag_scaffolder",
    )
    assert nudge.words == ("quantum", "tides")
    assert nudge.query == "quantum tides"
    assert nudge.source_url == "https://x.test/r"
    user_prompt = captured["messages"][1]["content"]
    assert "Full page body about reservoirs" in user_prompt
    assert "We were discussing water." in user_prompt


def test_rag_nudge_provider_falls_back_to_snippet_when_fetch_fails():
    captured = {}

    def llm_fn(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "ok"

    def boom(url, **kwargs):
        raise OSError("no network")

    provider = RagNudgeProvider(
        sampler=_StubSampler(["word"]),
        search_client=_StubSearch(
            [SearchResult("T", "https://x.test/r", "snippet body")]
        ),
        llm_config=_legacy_llm_config(),
        num_words=1,
        page_fetcher=boom,
        llm_fn=llm_fn,
    )
    provider.produce(
        RagMode.TOPIC, [_message(0, "hi")], speaker="rag_scaffolder"
    )
    assert "snippet body" in captured["messages"][1]["content"]


def test_rag_nudge_provider_narrows_query_until_results():
    search = _NarrowingSearch()
    provider = RagNudgeProvider(
        sampler=_StubSampler(["alpha", "beta", "gamma"]),
        search_client=search,
        llm_config=_legacy_llm_config(),
        num_words=3,
        fetch_page=False,
        llm_fn=lambda **kwargs: "grounded",
    )
    nudge = provider.produce(
        RagMode.NOVELTY, [_message(0, "hi")], speaker="rag_scaffolder"
    )
    assert nudge.query == "alpha"
    assert nudge.source_url == "https://x.test/hit"
    assert search.queries[:3] == ["alpha beta gamma", "alpha beta", "alpha"]


def test_rag_nudge_provider_raises_without_results():
    provider = RagNudgeProvider(
        sampler=_StubSampler(["word"]),
        search_client=_StubSearch([]),
        llm_config=_legacy_llm_config(),
        num_words=1,
        llm_fn=lambda **kwargs: "unused",
    )
    with pytest.raises(RagUnavailable):
        provider.produce(
            RagMode.NOVELTY, [_message(0, "hi")], speaker="rag_scaffolder"
        )


def test_rag_nudge_provider_raises_on_empty_llm_output():
    provider = RagNudgeProvider(
        sampler=_StubSampler(["word"]),
        search_client=_StubSearch(
            [SearchResult("T", "https://x.test/r", "snippet")]
        ),
        llm_config=_legacy_llm_config(),
        num_words=1,
        fetch_page=False,
        llm_fn=lambda **kwargs: "   ",
    )
    with pytest.raises(RagUnavailable):
        provider.produce(
            RagMode.NOVELTY, [_message(0, "hi")], speaker="rag_scaffolder"
        )


# --- RAG scaffolder -------------------------------------------------------


def _rag_policy(provider: _StubProvider) -> RagScaffolder:
    return RagScaffolder(
        ["ocean currents", "public libraries"],
        provider=provider,
        min_content_tokens=3,
        novelty_threshold=0.20,
        continuity_threshold=0.05,
        stuck_turns=1,
        memory_cooldown=0,
        novelty_nudge_rate=1.0,
        random_seed=7,
    )


def test_rag_scaffolder_attaches_novelty_provenance():
    policy = _rag_policy(_StubProvider(content="Explore tidal energy."))
    turn = policy.respond(
        [_message(0, "Ocean currents transport heat across the planet.")],
        speaker="rag_scaffolder",
    )
    assert turn.action is ScaffolderAction.THRIVE_PROTECTION
    assert turn.scores.informative
    assert turn.rag_used is True
    assert turn.rag_mode == "novelty"
    assert turn.novelty_nudge_kind is NoveltyNudgeKind.RAG_SEARCH
    assert turn.rag_words == ("alpha", "beta")
    assert turn.rag_source_url == "https://source.test/page"


def test_rag_scaffolder_falls_back_on_failure():
    provider = _StubProvider(error=RagUnavailable("network down"))
    policy = _rag_policy(provider)
    turn = policy.respond(
        [_message(0, "Ocean currents transport heat across the planet.")],
        speaker="rag_scaffolder",
    )
    assert turn.rag_used is False
    assert turn.rag_fallback_reason == "network down"
    assert turn.content
    assert turn.novelty_nudge_kind is not None


def test_rag_scaffolder_uses_rag_for_topic_injection():
    policy = _rag_policy(_StubProvider(content="Consider tidal power."))
    turn = policy.respond([_message(0, "Okay.")], speaker="rag_scaffolder")
    assert turn.action is ScaffolderAction.TOPIC_INJECTION
    assert turn.rag_used is True
    assert turn.rag_mode == "topic"
    assert turn.novelty_nudge_kind is None


def test_rag_scaffolder_state_round_trip_includes_provider():
    policy = _rag_policy(_StubProvider())
    policy.respond(
        [_message(0, "Ocean currents transport heat across the planet.")],
        speaker="rag_scaffolder",
    )
    state = policy.export_state()
    assert state["rag"] == {"stub": True}
    clone = _rag_policy(_StubProvider())
    clone.import_state(state)
    assert clone.export_state()["rag"] == {"stub": True}


# --- agent adapter / factory ---------------------------------------------


def test_scaffolder_turn_maps_rag_trace():
    turn = ScaffolderTurn(
        content="x",
        action=ScaffolderAction.TOPIC_INJECTION,
        scores=DecisionScores(
            informative=False,
            novelty=0.1,
            continuity=0.0,
            content_tokens=1,
            meta_detected=False,
        ),
        memory_size=0,
        rag_used=True,
        rag_mode="topic",
        rag_words=("a", "b"),
        rag_query="a b",
        rag_source_url="https://u",
        rag_source_title="t",
    )
    mapped = _scaffolder_turn_to_agent_turn(turn)
    assert mapped.rag_used is True
    assert mapped.rag_words == ["a", "b"]
    assert mapped.rag_query == "a b"
    assert mapped.rag_source_url == "https://u"


def test_factory_builds_rag_scaffolder_with_experiment_llm():
    configs = [
        LLMAgentConfig(provider="openai", model="gpt-4-0125-preview"),
        RagScaffolderAgentConfig(),
    ]
    agents = build_conversation_agents(configs)
    assert isinstance(agents[1], RagScaffolderConversationAgent)
    assert agents[1].speaker == "rag_scaffolder"


def test_factory_requires_llm_for_rag_scaffolder():
    from conversation.factory import build_agent

    with pytest.raises(ValueError):
        build_agent(
            RagScaffolderAgentConfig(),
            speaker="rag_scaffolder",
            llm_config=None,
        )


# --- trace forwarding / persistence --------------------------------------


def test_experiment_metric_forwards_rag_trace(tmp_path, monkeypatch):
    class _StubFetcher:
        def get_conversation(self):
            return [{"role": "user", "content": "hello"}]

    from prompt_fetcher import BasePromptFetcher

    monkeypatch.setattr(
        BasePromptFetcher,
        "create_fetcher",
        classmethod(lambda cls, *args, **kwargs: _StubFetcher()),
    )

    config = ExperimentConfig(
        fetcher_config=FetcherConfig(
            fetcher=FetcherType.SHAREGPT,
            data_path="data/x.json",
            min_messages=1,
            max_messages=2,
        ),
        analyzer_config=AnalyzerConfig(
            analyzer=AnalyzerType.SIMILARITY, analyze_window=3
        ),
        agents=[RagScaffolderAgentConfig()],
        agent_selection_method=AgentSelectionMethod.ROUND_ROBIN,
        output_dir=str(tmp_path),
    )
    experiment = Experiment(config)

    class _Agent:
        speaker = "rag_scaffolder"
        agent_id = "rag_scaffolder"
        _agent = None

    turn = AgentTurn(
        content="x",
        rag_used=True,
        rag_mode="novelty",
        rag_words=["a", "b"],
        rag_query="a b",
        rag_source_url="https://u",
        rag_source_title="t",
        scaffolder_action="thrive_protection",
        scaffolder_informative=True,
    )
    metric = experiment._build_agent_metric(_Agent(), 0, turn)
    assert metric.rag_used is True
    assert metric.rag_words == ["a", "b"]
    assert metric.rag_query == "a b"
    assert metric.rag_source_url == "https://u"
    assert metric.scaffolder_action == "thrive_protection"


def test_checkpoint_restores_rag_trace(tmp_path):
    metric = AgentMetric(
        iteration=0,
        timestamp=datetime(2026, 7, 13),
        role="rag_scaffolder",
        content="x",
        agent_id="rag_scaffolder",
        rag_used=True,
        rag_mode="novelty",
        rag_words=["a", "b"],
        rag_query="a b",
        rag_source_url="https://u",
        rag_source_title="t",
    )
    writer = CheckpointWriter(tmp_path)
    writer.save(
        state=ConversationState(run_id="run-1"),
        stack=ContextStack(),
        metrics=[metric],
        turn_taking_data={},
        settings={"max_iterations": 10},
        agents=[],
    )
    data = CheckpointWriter.load(tmp_path / "checkpoint.json")
    restored = CheckpointWriter.restore_metrics(data)[0]
    assert restored.rag_used is True
    assert restored.rag_mode == "novelty"
    assert restored.rag_words == ["a", "b"]
    assert restored.rag_source_url == "https://u"


def test_rag_trace_is_persisted(tmp_path):
    metric = AgentMetric(
        iteration=0,
        timestamp=datetime(2026, 7, 13),
        role="rag_scaffolder",
        content="x",
        agent_id="rag_scaffolder",
        rag_used=True,
        rag_mode="novelty",
        rag_words=["a", "b"],
        rag_query="a b",
        rag_source_url="https://u",
    )
    run_dir = tmp_path / "rag-run"
    save_run(run_dir, [metric], {"run_id": "rag-run"})
    row = load_run(run_dir).turns.iloc[0]
    assert bool(row["rag_used"]) is True
    assert row["rag_mode"] == "novelty"
    assert row["rag_source_url"] == "https://u"
    assert json.loads(row["rag_words"]) == ["a", "b"]
