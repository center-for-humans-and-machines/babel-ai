"""Traced ELIZA response generation for observability."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .vendor.utils.rank import rank
from .vendor.utils.response import (
    generate_generic_response,
    generate_memory_response,
)
from .vendor.utils.rules import decompose, reassemble


@dataclass(frozen=True)
class ElizaTrace:
    """Decision path taken by one ELIZA reply."""

    branch: str
    keyword: str | None = None
    reassembly: str | None = None
    memory_pushed: bool = False

    def label(self) -> str:
        """Return a compact human-readable branch label."""
        if self.branch == "keyword" and self.keyword:
            label = f"keyword:{self.keyword}"
            if self.memory_pushed:
                label += "+memory"
            return label
        if self.branch == "memory":
            return "memory:pop"
        if self.branch == "generic":
            return "generic:$"
        return self.branch


def generate_traced_response(
    in_str: str,
    script: list[dict],
    substitutions: dict,
    memory_stack: list[str],
    memory_inputs: list[str],
    intervention=None,
    *,
    session: bool = False,
) -> tuple[str, ElizaTrace]:
    """Mirror vendor response logic while recording the rule branch."""
    sentences = re.split(r"[.,!?](?!$)", in_str)
    sentence, sorted_keywords = rank(sentences, script, substitutions)

    for keyword in sorted_keywords:
        comps, reassembly_rule = decompose(keyword, sentence, script)
        if comps:
            memory_pushed = keyword in memory_inputs
            if memory_pushed:
                generate_memory_response(sentence, script, memory_stack)
            response = reassemble(comps, reassembly_rule)
            trace = ElizaTrace(
                branch="keyword",
                keyword=keyword,
                reassembly=reassembly_rule.strip(),
                memory_pushed=memory_pushed,
            )
            return _finalize(response, session), trace

    if memory_stack:
        response = memory_stack.pop()
        return _finalize(response, session), ElizaTrace(branch="memory")

    response = generate_generic_response(script)
    if intervention is not None:
        response = intervention(response)
    return _finalize(response, session), ElizaTrace(branch="generic")


def _finalize(response: str, session: bool) -> str:
    """Apply the same post-processing as the vendor helper."""
    if session:
        return response
    from .vendor.utils.response import prepare_response

    return prepare_response(response)
