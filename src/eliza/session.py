"""Stateful partner session backed by vendored ELIZA rules."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .interventions import GenericContext, GenericIntervention
from .vendor.utils.response import generate_response
from .vendor.utils.startup import setup

_SCRIPTS_DIR = Path(__file__).parent / "vendor" / "scripts"


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

        response = generate_response(
            user_turn,
            self._script,
            self._general_script["substitutions"],
            self._memory_stack,
            self._memory_inputs,
            intervention=intervene,
            session=True,
        )
        return PartnerTurn(
            text=self.strip_prefixes(response),
            used_generic_fallback=used_generic_fallback,
        )

    @staticmethod
    def _last_user_turn(messages: list[dict[str, Any]]) -> str:
        """Return the newest non-empty user message."""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
        raise ValueError("messages must include a non-empty user message")

    @staticmethod
    def strip_prefixes(text: str) -> str:
        """Remove interactive ELIZA framing from a response."""
        cleaned = text.strip()
        if cleaned.startswith("Eliza:"):
            cleaned = cleaned.removeprefix("Eliza:").lstrip()
        if "\nYou:" in cleaned:
            cleaned = cleaned.split("\nYou:", maxsplit=1)[0].strip()
        return cleaned
