"""
adapters/gemini_adapter.py
Gemini CLI adapter — calls `gemini -p "prompt"` as subprocess.
Requires: `gemini auth login` has been run.
Free tier: 60 req/min, 1000 req/day — adapter enforces per-day budget.
"""
import subprocess
import time
import os
from datetime import datetime, timedelta
from typing import Optional

from .base_adapter import BaseAdapter, AdapterResponse


class GeminiAdapter(BaseAdapter):
    """
    Research/macro-veto backend using Gemini CLI subprocess.
    Invokes: gemini -p "prompt" [--model gemini-2.0-flash]
    Free tier: 1000 req/day — budget enforcement required.
    """

    CLI_COMMAND = "gemini"
    DEFAULT_MODEL = "gemini-2.0-flash"
    DAILY_BUDGET = 900  # leave 100 req/day headroom for other uses
    CYCLE_GATE = 8      # only fire on every Nth cycle to conserve budget

    _daily_count: int = 0
    _daily_reset: Optional[datetime] = None

    def __init__(
        self,
        agent_id: str = "gemini-cli",
        model: Optional[str] = None,
        timeout: int = 60,
        daily_budget: int = DAILY_BUDGET,
        cycle_gate: int = CYCLE_GATE,
    ):
        super().__init__(agent_id, model or self.DEFAULT_MODEL)
        self.timeout = timeout
        self.daily_budget = daily_budget
        self.cycle_gate = cycle_gate

    def model_name(self) -> str:
        return self.model

    def _check_budget(self) -> bool:
        """Reset daily counter if day has rolled over."""
        now = datetime.utcnow()
        if self._daily_reset is None or now >= self._daily_reset:
            GeminiAdapter._daily_count = 0
            GeminiAdapter._daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        return GeminiAdapter._daily_count < self.daily_budget

    def should_fire(self, cycle_number: int) -> bool:
        """
        Only fire on every Nth cycle to conserve daily budget.
        Call this before invoking Gemini.
        """
        return cycle_number % self.cycle_gate == 0 and self._check_budget()

    def health_check(self) -> bool:
        try:
            result = subprocess.run(
                [self.CLI_COMMAND, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def ping(self) -> Optional[float]:
        """Ping Gemini CLI — returns latency ms or None."""
        start = time.monotonic()
        try:
            result = subprocess.run(
                [self.CLI_COMMAND, "-p", "hi"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return (time.monotonic() - start) * 1000
            return None
        except Exception:
            return None

    def call(self, prompt: str, **kwargs) -> AdapterResponse:
        if not self._check_budget():
            return AdapterResponse(
                text="",
                latency_ms=0,
                success=False,
                error="daily_budget_exhausted",
            )

        start = time.monotonic()
        try:
            cmd = [self.CLI_COMMAND, "-p", prompt]
            model_arg = kwargs.get("model")
            if model_arg and model_arg != self.DEFAULT_MODEL:
                cmd.insert(2, "--model")
                cmd.insert(3, model_arg)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", self.timeout),
            )
            latency = (time.monotonic() - start) * 1000

            if result.returncode != 0:
                GeminiAdapter._daily_count += 1
                return AdapterResponse(
                    text="",
                    latency_ms=latency,
                    success=False,
                    error=result.stderr or f"exit {result.returncode}",
                )

            text = result.stdout.strip()
            GeminiAdapter._daily_count += 1
            return AdapterResponse(text=text, latency_ms=latency, success=True)

        except subprocess.TimeoutExpired:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse(text="", latency_ms=latency, success=False, error="timeout")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse(text="", latency_ms=latency, success=False, error=str(e))