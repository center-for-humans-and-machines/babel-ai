"""Run a short ELIZA + mirror conversation without external LLM calls."""

from __future__ import annotations

import logging
from pathlib import Path

from conversation.factory import build_conversation_agents
from conversation.manager import ConversationManager
from models.configs import ExperimentConfig
from models.metrics import AnalysisResult
from persistence.run_naming import enrich_run_meta
from persistence.run_store import RunManifest, save_run
from utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class _LightAnalyzer:
    """Cheap analyzer for smoke tests without embedding models."""

    def analyze(self, contents: list[str]) -> AnalysisResult:
        text = contents[-1] if contents else ""
        words = text.split()
        unique = len(set(words))
        total = max(len(words), 1)
        return AnalysisResult(
            word_count=len(words),
            unique_word_count=unique,
            coherence_score=unique / total,
        )


def main() -> None:
    """Execute a six-turn ELIZA scaffolding smoke conversation."""
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "brief_eliza_demo.yaml"
    )
    config = load_yaml_config(ExperimentConfig, str(config_path))
    settings = config.resolved_conversation_settings()
    settings.checkpoint_enabled = False

    agents = build_conversation_agents(config.agents)
    output_dir = Path(config.output_dir or "results")
    manager = ConversationManager(
        agents=agents,
        settings=settings,
        analyzer=_LightAnalyzer(),
        run_dir=output_dir,
        fetcher_config=config.fetcher_config,
    )

    seed = [{"role": "user", "content": "Men are all alike."}]
    metrics = manager.run(seed)

    print("\n--- Transcript ---")
    for message in manager.stack.messages:
        print(f"[{message.speaker}] {message.content}")

    meta = enrich_run_meta(
        {
            "run_id": manager.run_id,
            "config": config.model_dump(),
            "turn_count": len(metrics),
        }
    )
    run_dir = save_run(
        output_dir / manager.run_id,
        metrics,
        meta,
        manifest=RunManifest(),
    )
    print(f"\nSaved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
