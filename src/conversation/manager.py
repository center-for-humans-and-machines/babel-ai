"""ConversationManager — multi-agent loop orchestration."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional
from uuid import uuid4

from analyzer import Analyzer
from conversation.agents import ConversationAgent
from conversation.checkpoint import CheckpointWriter, ConversationState
from conversation.messages import (
    ContextStack,
    ConversationMessage,
    MessageSource,
)
from conversation.settings import AnalysisPolicy, ConversationSettings
from conversation.status import ConversationStatus
from conversation.turn_taking import TurnTakingAlgorithm, build_turn_taking
from models import AgentMetric, FetcherConfig, FetcherMetric, Metric

logger = logging.getLogger(__name__)


class ConversationManager:
    """Orchestrates agents, context, analysis, and checkpoints."""

    def __init__(
        self,
        agents: List[ConversationAgent],
        settings: ConversationSettings,
        analyzer: Analyzer,
        run_dir: Path,
        turn_taking: Optional[TurnTakingAlgorithm] = None,
        fetcher_config: Optional[FetcherConfig] = None,
        metric_factory: Optional[
            Callable[[ConversationAgent, int, Any], AgentMetric]
        ] = None,
    ):
        if not agents:
            raise ValueError("At least one agent is required")
        self.agents = agents
        self.settings = settings
        self.analyzer = analyzer
        self.run_dir = run_dir
        self.fetcher_config = fetcher_config
        self._turn_taking = turn_taking or build_turn_taking(
            settings.turn_taking_method.value,
            settings.fixed_order,
        )
        self._metric_factory = metric_factory or self._default_agent_metric
        self.run_id = str(uuid4())
        self.state = ConversationState(run_id=self.run_id)
        self.stack = ContextStack()
        self.metrics: List[Metric] = []
        self._checkpoint = (
            CheckpointWriter(run_dir / self.run_id)
            if settings.checkpoint_enabled
            else None
        )
        self._last_checkpoint_time = time.monotonic()

    def run(self, seed_messages: List[dict]) -> List[Metric]:
        """Run conversation from seed messages until stop condition."""
        self.state.status = ConversationStatus.RUNNING
        self._ingest_seed(seed_messages)

        while self._should_continue():
            agent = self._select_agent()
            turn = agent.generate(self.stack)
            self._append_agent_turn(agent, turn)
            self._maybe_analyze(AnalysisPolicy.PER_TURN)
            self._maybe_checkpoint()

        self._finalize_analysis()
        self.state.status = ConversationStatus.COMPLETED
        if self._checkpoint:
            self._write_checkpoint()
        return self.metrics

    @classmethod
    def resume_from(
        cls,
        checkpoint_path: Path,
        agents: List[ConversationAgent],
        analyzer: Analyzer,
    ) -> "ConversationManager":
        """Restore manager state from ``checkpoint.json`` (E4)."""
        raise NotImplementedError("resume_from not implemented (E4)")

    def _ingest_seed(self, seed_messages: List[dict]) -> None:
        """Load starting conversation into stack and metrics."""
        for i, msg in enumerate(seed_messages):
            entry = ConversationMessage(
                turn_index=i,
                role=msg["role"],
                speaker=msg.get("speaker", "seed"),
                content=msg["content"],
                source=MessageSource.SEED,
            )
            self.stack.append(entry)
            if self.fetcher_config is not None:
                self.metrics.append(
                    FetcherMetric(
                        iteration=i,
                        timestamp=datetime.now(),
                        role=msg["role"],
                        content=msg["content"],
                        fetcher_config=self.fetcher_config,
                    )
                )

    def _select_agent(self) -> ConversationAgent:
        index = self._turn_taking.next_agent_index(
            len(self.agents),
            self.state.turn_taking_state,
        )
        return self.agents[index]

    def _append_agent_turn(self, agent: ConversationAgent, turn) -> None:
        iteration = len(self.metrics)
        role = "assistant" if self.state.agent_turn_count % 2 else "user"
        self.state.agent_turn_count += 1
        entry = ConversationMessage(
            turn_index=iteration,
            role=role,
            speaker=agent.speaker,
            content=turn.content,
            source=MessageSource.AGENT,
        )
        self.stack.append(entry)
        metric = self._metric_factory(agent, iteration, turn)
        self.metrics.append(metric)
        if turn.llm_nudge:
            self.state.pending_llm_nudge = turn.llm_nudge

    def _default_agent_metric(
        self,
        agent: ConversationAgent,
        iteration: int,
        turn: Any,
    ) -> AgentMetric:
        llm_agent = getattr(agent, "_agent", None)
        config = llm_agent.config if llm_agent is not None else None
        content = turn.content if hasattr(turn, "content") else str(turn)
        fallback = getattr(turn, "used_generic_fallback", None)
        return AgentMetric(
            iteration=iteration,
            timestamp=datetime.now(),
            role=agent.speaker,
            content=content,
            agent_id=agent.agent_id,
            agent_config=config,
            speaker=agent.speaker,
            used_generic_fallback=fallback or None,
        )

    def _should_continue(self) -> bool:
        if len(self.stack.messages) >= self.settings.max_iterations:
            return False
        if self.stack.total_characters() >= self.settings.max_total_characters:
            return False
        return True

    def _maybe_analyze(self, policy: AnalysisPolicy) -> None:
        if self.settings.analysis_policy != policy:
            return
        self._analyze_latest()

    def _finalize_analysis(self) -> None:
        if self.settings.analysis_policy == AnalysisPolicy.AT_END:
            self._analyze_all()
        elif self.settings.analysis_policy == AnalysisPolicy.ON_CHECKPOINT:
            self._analyze_all()

    def _analyze_latest(self) -> None:
        if not self.metrics:
            return
        contents = self.stack.content_prefix()
        self.metrics[-1].analysis = self.analyzer.analyze(contents)

    def _analyze_all(self) -> None:
        contents = self.stack.content_prefix()
        for i, metric in enumerate(self.metrics):
            prefix = contents[: i + 1]
            metric.analysis = self.analyzer.analyze(prefix)

    def _maybe_checkpoint(self) -> None:
        if not self._checkpoint:
            return
        elapsed = time.monotonic() - self._last_checkpoint_time
        if elapsed < self.settings.checkpoint_interval_seconds:
            return
        self._maybe_analyze(AnalysisPolicy.ON_CHECKPOINT)
        self._write_checkpoint()
        self._last_checkpoint_time = time.monotonic()

    def _write_checkpoint(self) -> None:
        if not self._checkpoint:
            return
        self._checkpoint.save(
            state=self.state,
            stack=self.stack,
            metrics=self.metrics,
            turn_taking_data=self._turn_taking.snapshot(
                self.state.turn_taking_state
            ),
            settings=self.settings.model_dump(),
        )
