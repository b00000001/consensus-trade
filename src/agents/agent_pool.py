"""
Phase 4 — agents/agent_pool.py
Parallel fanout to all strategy agents + research agent, collects AgentSignals.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from agents.base_agent import AgentSignal
from agents.strategy_agent import StrategyAgent
from agents.research_agent import ResearchAgent, ResearchReport


log = logging.getLogger(__name__)

MAX_WORKERS = 8


@dataclass
class PoolResult:
    signals: list[AgentSignal]
    research: Optional[ResearchReport]
    errors: list[str]


class AgentPool:
    """
    Manages the pool of Tier 1 agents.
    Supports parallel fanout for signal generation + research.
    """

    def __init__(self, agents: list[StrategyAgent], research_agent: Optional[ResearchAgent] = None):
        self.agents = agents
        self.research_agent = research_agent or ResearchAgent()

    def fanout_signals(
        self,
        symbol: str,
        timeframe: str = "1h",
        sentiment: Optional[str] = None,
    ) -> list[AgentSignal]:
        """
        Parallel fanout to all strategy agents.
        Returns collected AgentSignals.
        """
        signals = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(agent.generate, symbol, timeframe, sentiment): agent.agent_id
                for agent in self.agents
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    signal = future.result()
                    signals.append(signal)
                except Exception as e:
                    log.error("[%s] Signal generation error: %s", agent_id, e)
                    signals.append(AgentSignal(
                        agent_id=agent_id,
                        role="signal",
                        model="unknown",
                        signal="HOLD",
                        confidence=0.0,
                        reasoning=f"error:{str(e)}",
                    ))
        return signals

    def run_full_cycle(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> PoolResult:
        """
        Run research + signal generation in parallel.
        Returns PoolResult with all signals + research report.
        """
        signals = []
        research = None
        errors = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            # Submit research task
            research_future = ex.submit(self.research_agent.run_pipeline)

            # Submit signal tasks
            signal_futures = {
                ex.submit(agent.generate, symbol, timeframe): agent.agent_id
                for agent in self.agents
            }

            # Collect research
            try:
                research = research_future.result()
            except Exception as e:
                log.error("Research pipeline error: %s", e)
                errors.append(f"research:{str(e)}")

            # Collect signals
            for future in as_completed(signal_futures):
                agent_id = signal_futures[future]
                try:
                    signals.append(future.result())
                except Exception as e:
                    log.error("[%s] Signal error: %s", agent_id, e)
                    errors.append(f"signal:{agent_id}:{str(e)}")

        return PoolResult(signals=signals, research=research, errors=errors)
