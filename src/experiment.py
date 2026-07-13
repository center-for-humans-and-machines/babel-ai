"""LLM Drift Experiment.

Runs multi-agent conversations via ConversationManager and persists flat
run artifacts under ``results/{run_id}/``.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from agent import Agent
from analyzer import Analyzer
from conversation.agents import ConversationAgent, LLMConversationAgent
from conversation.factory import build_conversation_agents
from conversation.manager import ConversationManager
from conversation.settings import AnalysisPolicy, TurnTakingMethod
from enums import AgentSelectionMethod
from models import AgentMetric, ExperimentConfig, ExperimentMetadata, Metric
from persistence.run_store import RunManifest, save_run
from prompt_fetcher import BasePromptFetcher

logger = logging.getLogger(__name__)


class Experiment:
    """Main class for running LLM drift experiments."""

    def __init__(self, config: ExperimentConfig):
        self.uuid = uuid4()
        logger.info(
            f"Initializing Experiment {self.uuid} with config: {config}"
        )

        self.config = config
        self.conversation_settings = config.resolved_conversation_settings()
        self.max_iterations = config.max_iterations
        self.max_total_characters = config.max_total_characters

        if config.output_dir is None:
            self.output_dir = Path.cwd() / "results"
            logger.warning(
                f"Experiment {self.uuid} "
                f"No output directory specified, using {self.output_dir}"
            )
        else:
            self.output_dir = Path(config.output_dir)
            logger.info(
                f"Experiment {self.uuid} "
                f"Using output directory {self.output_dir}"
            )

        self.metadata = ExperimentMetadata(
            timestamp=datetime.now(),
            config=config,
        )

        self.analyzer = Analyzer.create_analyzer(
            analyzer_type=self.config.analyzer_config.analyzer,
            analyze_window=self.config.analyzer_config.analyze_window,
        )

        fetcher_kwargs = {
            key: value
            for key, value in self.config.fetcher_config.model_dump(
                exclude={"fetcher"}
            ).items()
            if value is not None
        }
        self.prompt_fetcher = BasePromptFetcher.create_fetcher(
            fetcher_type=self.config.fetcher_config.fetcher,
            **fetcher_kwargs,
        )

        self.legacy_agents = [
            Agent(agent_config) for agent_config in self.config.agent_configs
        ]
        self.agent_selection_method = AgentSelectionMethod(
            self.config.agent_selection_method
        )
        self.result_metrics: List[Metric] = []
        self.messages: List[
            Dict[str, str]
        ] = self.prompt_fetcher.get_conversation()
        self.metadata.num_fetcher_messages = len(self.messages)
        self._manager: Optional[ConversationManager] = None

    def run(
        self,
        output_dir: Optional[Path] = None,
    ) -> List[Metric]:
        """Run the drift experiment."""
        self.run_interaction_loop()
        if self.conversation_settings.analysis_policy == AnalysisPolicy.AT_END:
            self._analyze_response(self.result_metrics)
        self._save_results(
            metrics=self.result_metrics,
            metadata=self.metadata,
            output_dir=output_dir,
        )
        logger.info(
            f"Experiment {self.uuid} completed with "
            f"{len(self.result_metrics)} metrics"
        )
        return self.result_metrics

    def run_interaction_loop(self) -> List[Metric]:
        """Run the conversation via ConversationManager."""
        conv_agents = self._build_conversation_agents()
        settings = self.conversation_settings.model_copy()
        settings.max_iterations = self.max_iterations
        settings.max_total_characters = self.max_total_characters
        if self.agent_selection_method == AgentSelectionMethod.ROUND_ROBIN:
            settings.turn_taking_method = TurnTakingMethod.ROUND_ROBIN

        manager = ConversationManager(
            agents=conv_agents,
            settings=settings,
            analyzer=self.analyzer,
            run_dir=self.output_dir,
            fetcher_config=self.config.fetcher_config,
            metric_factory=self._build_agent_metric,
        )
        self._manager = manager
        self.result_metrics = manager.run(self.messages)
        self.messages = [
            {"role": message.role, "content": message.content}
            for message in manager.stack.messages
        ]
        self.total_characters = manager.stack.total_characters()
        self.metadata.total_characters = self.total_characters
        self.metadata.num_iterations_total = len(self.result_metrics)
        return self.result_metrics

    def _build_conversation_agents(self) -> List[ConversationAgent]:
        """Build canonical or legacy conversation agents."""
        if self.config.uses_canonical_agents():
            return build_conversation_agents(self.config.agents)
        return [
            LLMConversationAgent(agent=agent, speaker=f"agent_{index}")
            for index, agent in enumerate(self.legacy_agents)
        ]

    def _build_agent_metric(
        self,
        agent: ConversationAgent,
        iteration: int,
        turn,
    ) -> AgentMetric:
        llm_agent = getattr(agent, "_agent", None)
        config = llm_agent.config if llm_agent is not None else None
        return AgentMetric(
            iteration=iteration,
            timestamp=datetime.now(),
            role=agent.speaker,
            content=turn.content,
            agent_id=agent.agent_id,
            agent_config=config,
            speaker=agent.speaker,
            used_generic_fallback=turn.used_generic_fallback or None,
        )

    def _analyze_response(self, metrics: List[Metric]) -> List[Metric]:
        """Analyze all metrics (at_end policy)."""
        logger.info(
            f"Experiment {self.uuid} "
            f"Analyzing response for {len(metrics)} metrics"
        )
        content = [metric.content for metric in metrics]
        for index, metric in enumerate(metrics):
            logger.info(
                f"Experiment {self.uuid} "
                f"Analyzing response for {index} of {len(metrics)} metrics"
            )
            metric.analysis = self.analyzer.analyze(content[: index + 1])
        return metrics

    def _save_results(
        self,
        metrics: List[Metric],
        metadata: ExperimentMetadata,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Persist canonical run artifacts under ``results/{run_id}/``."""
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = self._manager.run_id if self._manager else str(self.uuid)
        run_dir = output_dir / run_id
        meta = {
            "run_id": run_id,
            "experiment_uuid": str(self.uuid),
            "timestamp": metadata.timestamp.isoformat(),
            "config": metadata.config.model_dump(),
            "num_iterations_total": metadata.num_iterations_total,
            "num_fetcher_messages": metadata.num_fetcher_messages,
            "total_characters": metadata.total_characters,
        }
        save_run(run_dir, metrics, meta, manifest=RunManifest())
        logger.info(f"Saved canonical run artifacts to {run_dir}")
