"""Stateful partner session backed by vendored ELIZA rules."""

import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .interventions import GenericContext, GenericIntervention
from .response_trace import generate_traced_response
from .vendor.utils.startup import setup

_SCRIPTS_DIR = Path(__file__).parent / "vendor" / "scripts"
_TOPIC_REDIRECT_MARKER = "LAST_TOPIC_REDIRECT"
_PEER_ROLES = frozenset({"user", "human"})
_MAX_TOPIC_LEN = 80


@lru_cache
def _load_scripts(
    script_name: str,
) -> tuple[dict[str, Any], list[dict], list[str]]:
    """Load and preprocess the immutable script templates."""
    if Path(script_name).name != script_name:
        raise ValueError("script_name must not contain a path")

    general_path = _SCRIPTS_DIR / "general.json"
    script_path = _SCRIPTS_DIR / f"{script_name}.json"
    general, script, memory_inputs, _ = setup(
        str(general_path), str(script_path)
    )
    return general, script, memory_inputs


@dataclass
class PartnerTurn:
    """One rule-based partner reply."""

    text: str
    used_generic_fallback: bool = False
    eliza_branch: str | None = None
    eliza_keyword: str | None = None
    eliza_reassembly: str | None = None


class PartnerSession:
    """Maintain ELIZA rule rotation and memory for one conversation."""

    def __init__(
        self,
        intervention: GenericIntervention | None = None,
        script_name: str = "doctor",
    ) -> None:
        general, script, memory_inputs = deepcopy(_load_scripts(script_name))
        self._general_script = general
        self._script = script
        self._memory_inputs = memory_inputs
        self._intervention = intervention
        self._memory_stack: list[str] = []

    def respond(self, messages: list[dict[str, Any]]) -> PartnerTurn:
        """Produce the next partner turn from conversation history."""
        user_turn = self._last_user_turn(messages)
        used_generic_fallback = False
        turn_index = (
            sum(message.get("role") == "user" for message in messages) - 1
        )

        def intervene(default_response: str) -> str:
            nonlocal used_generic_fallback
            used_generic_fallback = True
            if self._intervention is None:
                return default_response
            context = GenericContext(
                user_turn=user_turn,
                default_response=self.strip_prefixes(default_response),
                messages=messages,
                turn_index=turn_index,
            )
            result = self._intervention.on_generic(context)
            return result.partner_text or default_response

        response, trace = generate_traced_response(
            user_turn,
            self._script,
            self._general_script["substitutions"],
            self._memory_stack,
            self._memory_inputs,
            intervention=intervene,
            session=True,
        )
        text = self.strip_prefixes(response)
        text, trace_reassembly = self._apply_topic_redirect(
            messages,
            text,
            trace.reassembly,
        )
        text, trace_reassembly = self._apply_self_reference_redirect(
            messages,
            user_turn,
            text,
            trace_reassembly,
        )
        return PartnerTurn(
            text=text,
            used_generic_fallback=used_generic_fallback,
            eliza_branch=trace.label(),
            eliza_keyword=trace.keyword,
            eliza_reassembly=trace_reassembly,
        )

    @staticmethod
    def _last_user_turn(messages: list[dict[str, Any]]) -> str:
        """Return the newest non-empty peer message."""
        for message in reversed(messages):
            role = str(message.get("role", "")).lower()
            if role not in _PEER_ROLES:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
        raise ValueError("messages must include a non-empty user message")

    @classmethod
    def _apply_self_reference_redirect(
        cls,
        messages: list[dict[str, Any]],
        user_turn: str,
        text: str,
        reassembly: str | None,
    ) -> tuple[str, str | None]:
        """Redirect self-focused peer turns back to the prior subject."""
        if not cls._is_self_referential(user_turn):
            return text, reassembly
        topic = cls._prior_peer_topic(messages)
        if not topic:
            return text, reassembly
        return (
            f"We are talking about {topic} - not me.",
            "self_reference_redirect",
        )

    @staticmethod
    def _is_self_referential(user_turn: str) -> bool:
        """Return True when the peer turn uses first-person phrasing."""
        return bool(re.search(r"\bI\b", user_turn))

    @classmethod
    def _apply_topic_redirect(
        cls,
        messages: list[dict[str, Any]],
        text: str,
        reassembly: str | None,
    ) -> tuple[str, str | None]:
        """Replace topic redirect markers with the prior peer subject."""
        if text != _TOPIC_REDIRECT_MARKER:
            return text, reassembly
        topic = cls._prior_peer_topic(messages)
        if topic:
            return f"We are talking about {topic} - not me.", reassembly
        return "Let's stay with the earlier subject - not me.", reassembly

    @staticmethod
    def _prior_peer_topic(messages: list[dict[str, Any]]) -> str | None:
        """Return the peer turn immediately before the latest one."""
        peers = PartnerSession._peer_contents(messages)
        if len(peers) < 2:
            return None
        topic = peers[-2]
        if len(topic) > _MAX_TOPIC_LEN:
            topic = topic[: _MAX_TOPIC_LEN - 3] + "..."
        return topic

    @staticmethod
    def _peer_contents(messages: list[dict[str, Any]]) -> list[str]:
        """Collect peer/user message contents in conversation order."""
        contents: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).lower()
            if role not in _PEER_ROLES:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                contents.append(content.strip())
        return contents

    @staticmethod
    def strip_prefixes(text: str) -> str:
        """Remove interactive ELIZA framing from a response."""
        cleaned = text.strip()
        if cleaned.startswith("Eliza:"):
            cleaned = cleaned.removeprefix("Eliza:").lstrip()
        if "\nYou:" in cleaned:
            cleaned = cleaned.split("\nYou:", maxsplit=1)[0].strip()
        return cleaned

    def export_state(self) -> dict[str, Any]:
        """Serialize memory stack and script counters for checkpoints."""
        return {
            "memory_stack": list(self._memory_stack),
            "script": deepcopy(self._script),
        }

    def import_state(self, data: dict[str, Any]) -> None:
        """Restore memory stack and script counters from a checkpoint."""
        self._memory_stack = list(data.get("memory_stack", []))
        if "script" in data:
            self._script = deepcopy(data["script"])
