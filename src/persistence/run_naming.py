"""Human-readable run directory names from experiment config."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import uuid4

from models.configs import ExperimentConfig


def format_run_timestamp(when: datetime) -> str:
    """Return a readable local timestamp for saved run metadata."""
    return when.strftime("%A, %d %B %Y, %H:%M")


def enrich_run_meta(
    meta: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    """Attach ISO and human-readable timestamps to run metadata."""
    when = timestamp or datetime.now()
    enriched = dict(meta)
    enriched["timestamp"] = when.isoformat()
    enriched["timestamp_human"] = format_run_timestamp(when)
    return enriched


def timestamp_from_meta(meta: dict) -> str:
    """Return a human-readable timestamp from saved run metadata."""
    human = meta.get("timestamp_human")
    if isinstance(human, str) and human.strip():
        return human
    raw = meta.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        return format_run_timestamp(datetime.fromisoformat(raw))
    except ValueError:
        return raw


def slugify_token(value: str, *, max_len: int = 24) -> str:
    """Normalize one config token for filesystem-safe run names."""
    cleaned = value.lower().replace(":", "-").replace(" ", "-")
    cleaned = re.sub(r"[^\w\-]+", "", cleaned)
    return cleaned[:max_len] or "unknown"


def build_run_slug(config: ExperimentConfig) -> str:
    """Build a descriptive slug from the experiment configuration."""
    parts: list[str] = []

    if config.agents:
        agent_bits: list[str] = []
        for agent in config.agents:
            agent_type = getattr(agent, "type", None)
            type_value = (
                agent_type.value if agent_type is not None else "agent"
            )
            if type_value == "llm":
                provider = slugify_token(agent.provider, max_len=12)
                model = slugify_token(agent.model, max_len=20)
                agent_bits.append(f"llm-{provider}-{model}")
            elif type_value == "rule_based":
                partner = slugify_token(getattr(agent, "partner", "eliza"))
                agent_bits.append(partner)
            elif type_value == "mirror":
                agent_bits.append("mirror")
            else:
                agent_bits.append(slugify_token(type_value))
        parts.append("-".join(agent_bits))
    elif config.agent_configs:
        first = config.agent_configs[0]
        provider = slugify_token(first.provider.value, max_len=12)
        model = slugify_token(first.model.value, max_len=20)
        parts.append(f"llm-{provider}-{model}")

    fetcher = config.fetcher_config.fetcher
    fetcher_value = (
        fetcher.value if hasattr(fetcher, "value") else str(fetcher)
    )
    parts.append(slugify_token(fetcher_value, max_len=16))
    parts.append(f"{config.max_iterations}turns")

    return "__".join(parts)


def build_run_id(config: ExperimentConfig) -> str:
    """Return ``{slug}_{short_uuid}`` for a completed run directory."""
    return f"{build_run_slug(config)}_{uuid4().hex[:8]}"
