"""
Phase 2 — risk/budget_guard.py
Tier 0: per-strategy daily token cap, per-agent max calls/hour.
No LLM calls.
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class BudgetProfile:
    daily_token_cap: int = 1_000_000   # per strategy per day
    max_calls_per_hour: int = 100       # per agent per hour


@dataclass
class AgentBudget:
    calls_this_hour: int = 0
    tokens_today: int = 0
    hour_start: float = field(default_factory=time.time)
    day_start: float = field(default_factory=time.time)


class BudgetGuard:
    """
    Tracks token usage and call counts per agent/strategy.
    Trips when a budget is exhausted.
    """

    def __init__(self, profiles: Optional[dict[str, BudgetProfile]] = None):
        self._profiles = profiles or {}
        self._agent_budgets: dict[str, AgentBudget] = defaultdict(AgentBudget)
        self._strategy_tokens: dict[str, int] = defaultdict(int)
        self._strategy_day_start: dict[str, float] = {}
        self._lock = Lock()

    def get_profile(self, strategy: str) -> BudgetProfile:
        return self._profiles.get(strategy, BudgetProfile())

    def cap_reached(self, strategy: str) -> bool:
        """Return True if strategy has hit its daily token cap."""
        with self._lock:
            self._maybe_roll_strategy_day(strategy)
            profile = self.get_profile(strategy)
            return self._strategy_tokens.get(strategy, 0) >= profile.daily_token_cap

    def agent_cap_reached(self, agent_id: str) -> bool:
        """Return True if agent has hit its hourly call cap."""
        with self._lock:
            budget = self._agent_budgets[agent_id]
            self._maybe_roll_agent_hour(budget)
            profile = self.get_profile("")  # use default
            return budget.calls_this_hour >= profile.max_calls_per_hour

    def record_call(self, agent_id: str, input_tokens: int, output_tokens: int, strategy: str):
        """Record an API call and its token usage."""
        total_tokens = input_tokens + output_tokens
        with self._lock:
            budget = self._agent_budgets[agent_id]
            self._maybe_roll_agent_hour(budget)
            budget.calls_this_hour += 1
            budget.tokens_today += total_tokens

            self._maybe_roll_strategy_day(strategy)
            self._strategy_tokens[strategy] += total_tokens

    def status(self, strategy: str, agent_id: str) -> dict:
        """Return budget status for a strategy + agent combo."""
        with self._lock:
            self._maybe_roll_strategy_day(strategy)
            agent_budget = self._agent_budgets[agent_id]
            self._maybe_roll_agent_hour(agent_budget)

            profile = self.get_profile(strategy)
            return {
                "strategy_tokens_today": self._strategy_tokens.get(strategy, 0),
                "strategy_daily_cap": profile.daily_token_cap,
                "agent_calls_this_hour": agent_budget.calls_this_hour,
                "agent_hourly_cap": profile.max_calls_per_hour,
                "agent_tokens_today": agent_budget.tokens_today,
            }

    def _maybe_roll_strategy_day(self, strategy: str):
        now = time.time()
        if strategy not in self._strategy_day_start or now - self._strategy_day_start.get(strategy, 0) >= 86400:
            self._strategy_tokens[strategy] = 0
            self._strategy_day_start[strategy] = now

    @staticmethod
    def _maybe_roll_agent_hour(budget: AgentBudget):
        now = time.time()
        if now - budget.hour_start >= 3600:
            budget.calls_this_hour = 0
            budget.tokens_today = 0
            budget.hour_start = now
