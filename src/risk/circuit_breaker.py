"""
Phase 2 — risk/circuit_breaker.py
Tier 0: circuit breaker — max daily loss %, max trades/hour, max consecutive losses.
No LLM calls.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class CircuitBreakerState:
    consecutive_losses: int = 0
    total_trades_today: int = 0
    total_loss_today: float = 0.0    # absolute USD loss
    trades_this_hour: int = 0
    hour_start: float = field(default_factory=time.time)
    day_start: float = field(default_factory=time.time)

    # Rolling window of trade outcomes (True=win, False=loss)
    trade_outcomes: deque = field(default_factory=lambda: deque(maxlen=100))


class CircuitBreaker:
    """
    Tracks trading activity and trips if limits are exceeded.
    All state protected by a lock for thread safety.
    """

    def __init__(
        self,
        max_daily_loss_usd: float = 500.0,
        max_trades_per_hour: int = 20,
        max_consecutive_losses: int = 3,
    ):
        self._max_daily_loss_usd = max_daily_loss_usd
        self._max_trades_per_hour = max_trades_per_hour
        self._max_consecutive_losses = max_consecutive_losses
        self._lock = Lock()
        self._state = CircuitBreakerState()

    # ---- Queries ----

    def is_tripped(self) -> bool:
        """Return True if ANY circuit is open."""
        with self._lock:
            self._maybe_roll_hour()
            self._maybe_roll_day()

        return (
            self._is_daily_loss_tripped()
            or self._is_trades_per_hour_tripped()
            or self._is_consecutive_losses_tripped()
        )

    def status(self) -> dict:
        """Return current breaker status."""
        with self._lock:
            self._maybe_roll_hour()
            self._maybe_roll_day()
            return {
                "tripped": self.is_tripped(),
                "consecutive_losses": self._state.consecutive_losses,
                "total_trades_today": self._state.total_trades_today,
                "total_loss_today": self._state.total_loss_today,
                "trades_this_hour": self._state.trades_this_hour,
                "max_daily_loss_usd": self._max_daily_loss_usd,
                "max_trades_per_hour": self._max_trades_per_hour,
                "max_consecutive_losses": self._max_consecutive_losses,
            }

    # ---- Mutations (called by execution layer) ----

    def record_trade(self, pnl: float):
        """
        Record trade outcome: positive pnl = win, negative = loss.
        """
        with self._lock:
            self._maybe_roll_hour()
            self._maybe_roll_day()

            self._state.total_trades_today += 1
            self._state.trades_this_hour += 1
            self._state.trade_outcomes.append(pnl >= 0)

            if pnl < 0:
                self._state.consecutive_losses += 1
                self._state.total_loss_today += abs(pnl)
            else:
                self._state.consecutive_losses = 0

    def reset_day(self):
        """Manually reset daily counters (e.g., at session start)."""
        with self._lock:
            now = time.time()
            self._state.day_start = now
            self._state.total_trades_today = 0
            self._state.total_loss_today = 0.0
            self._state.consecutive_losses = 0
            self._state.trade_outcomes.clear()

    # ---- Internal checks ----

    def _is_daily_loss_tripped(self) -> bool:
        return self._state.total_loss_today >= self._max_daily_loss_usd

    def _is_trades_per_hour_tripped(self) -> bool:
        return self._state.trades_this_hour >= self._max_trades_per_hour

    def _is_consecutive_losses_tripped(self) -> bool:
        return self._state.consecutive_losses >= self._max_consecutive_losses

    def _maybe_roll_hour(self):
        now = time.time()
        if now - self._state.hour_start >= 3600:
            self._state.trades_this_hour = 0
            self._state.hour_start = now

    def _maybe_roll_day(self):
        now = time.time()
        if now - self._state.day_start >= 86400:
            self._state.total_trades_today = 0
            self._state.total_loss_today = 0.0
            self._state.consecutive_losses = 0
            self._state.day_start = now
            self._state.trade_outcomes.clear()
