"""Tests for traced ELIZA response generation."""

from eliza.response_trace import ElizaTrace
from eliza.session import PartnerSession


def test_keyword_branch_records_match_metadata():
    session = PartnerSession()
    turn = session.respond([{"role": "user", "content": "Men are all alike."}])
    assert turn.text == "In what way?"
    assert turn.eliza_branch == "keyword:like"
    assert turn.eliza_keyword == "like"
    assert turn.eliza_reassembly


def test_generic_branch_records_fallback():
    turn = PartnerSession().respond([{"role": "user", "content": "xyzzy"}])
    assert turn.eliza_branch == "generic:$"
    assert turn.used_generic_fallback


def test_memory_branch_records_pop():
    session = PartnerSession()
    session.respond([{"role": "user", "content": "My mother is kind."}])
    turn = session.respond([{"role": "user", "content": "xyzzy"}])
    assert turn.eliza_branch == "memory:pop"


def test_eliza_trace_label_formats_memory_push():
    trace = ElizaTrace(
        branch="keyword",
        keyword="mother",
        reassembly="TELL ME MORE",
        memory_pushed=True,
    )
    assert trace.label() == "keyword:mother+memory"
