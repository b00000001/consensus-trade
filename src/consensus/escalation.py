"""
Phase 5 — consensus/escalation.py
Confidence gate + cooldown with hash-based cache key.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class EscalationConfig:
    confidence_threshold: float = 0.7
    cooldown_seconds: int = 300  # 5 minutes
    max_escalations_per_hour: int = 3


class EscalationTracker:
    """
    Tracks escalation state per asset to prevent rapid re-escalation.
    Thread-safe.
    """

    def __init__(self, config: EscalationConfig = None):
        self.config = config or EscalationConfig()
        self._lock = Lock()
        self._last_escalation: dict[str, float] = {}  # asset → timestamp
        self._escalation_count: dict[str, int] = {}   # asset → count this hour
        self._hour_start: dict[str, float] = {}        # asset → hour roll

    def can_escalate(self, asset: str) -> bool:
        """
        Returns True if escalation is allowed for this asset.
        Checks cooldown and hourly rate limit.
        """
        with self._lock:
            self._maybe_roll_hour(asset)
            now = time.time()

            # Cooldown check
            last = self._last_escalation.get(asset, 0)
            if now - last < self.config.cooldown_seconds:
                return False

            # Hourly rate check
            if self._escalation_count.get(asset, 0) >= self.config.max_escalations_per_hour:
                return False

            return True

    def record_escalation(self, asset: str):
        """Record that an escalation occurred."""
        with self._lock:
            now = time.time()
            self._last_escalation[asset] = now
            self._escalation_count[asset] = self._escalation_count.get(asset, 0) + 1

    def make_escalation_key(self, asset: str, signals: list) -> str:
        """
        Deterministic cache key from asset + sorted signal content.
        """
        signal_content = json.dumps(sorted(
            {k: str(v) for k, v in s.__dict__.items()} for s in signals
        ), sort_keys=True)
        raw = f"{asset}:{signal_content}"
        return f"escalation:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

    def _maybe_roll_hour(self, asset: str):
        now = time.time()
        if asset not in self._hour_start or now - self._hour_start[asset] >= 3600:
            self._escalation_count[asset] = 0
            self._hour_start[asset] = now

    def status(self, asset: str) -> dict:
        with self._lock:
            self._maybe_roll_hour(asset)
            now = time.time()
            last = self._last_escalation.get(asset, 0)
            return {
                "can_escalate": self.can_escalate(asset),
                "time_since_last": now - last if last else None,
                "escalations_this_hour": self._escalation_count.get(asset, 0),
                "config": self.config.__dict__,
            }
