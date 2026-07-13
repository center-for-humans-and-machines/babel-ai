"""Resume a conversation from ``results/{run_id}/checkpoint.json``."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from conversation.factory import build_conversation_agents
from conversation.manager import ConversationManager
from models.configs import ExperimentConfig
from models.metrics import AnalysisResult
from persistence.run_store import RunManifest, save_run
from utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class _LightAnalyzer:
    """Cheap analyzer for resumed smoke runs without embedding models."""

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
    """Resume from checkpoint and continue until stop limits."""
    parser = argparse.ArgumentParser(description="Resume a conversation run")
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint.json under results/{run_id}/",
    )
    parser.add_argument(
        "--config",
        help="Experiment YAML with the same agents as the original run",
        default="configs/brief_eliza_demo.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(ExperimentConfig, str(config_path))
    checkpoint_path = Path(args.checkpoint)
    agents = build_conversation_agents(config.agents)

    manager = ConversationManager.resume_from(
        checkpoint_path,
        agents=agents,
        analyzer=_LightAnalyzer(),
        fetcher_config=config.fetcher_config,
    )
    metrics = manager.continue_run()

    print("\n--- Transcript ---")
    for message in manager.stack.messages:
        print(f"[{message.speaker}] {message.content}")

    output_dir = Path(config.output_dir or "results")
    meta = {
        "run_id": manager.run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config.model_dump(),
        "turn_count": len(metrics),
        "resumed_from": str(checkpoint_path),
    }
    run_dir = save_run(
        output_dir / manager.run_id,
        metrics,
        meta,
        manifest=RunManifest(),
    )
    print(f"\nSaved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
