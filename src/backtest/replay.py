"""
Phase 8 — backtest/replay.py
Stores prompt + snapshot per call for deterministic replay.
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class ReplayEntry:
    timestamp: str
    prompt: str
    market_snapshot: dict
    signal: str
    confidence: float
    decision: str


class ReplayStore:
    """
    Stores prompt + market snapshot for each agent call.
    Enables deterministic replay of backtest runs.
    """

    def __init__(self, output_dir: str = "/opt/consensus-trade/data/backtests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id: Optional[str] = None
        self._entries: list[dict] = []

    def start_run(self, run_id: str):
        self._run_id = run_id
        self._entries = []

    def append(
        self,
        agent_id: str,
        prompt: str,
        market_snapshot: dict,
        signal: str,
        confidence: float,
        decision: str,
    ):
        self._entries.append({
            "agent_id": agent_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt": prompt,
            "market_snapshot": market_snapshot,
            "signal": signal,
            "confidence": confidence,
            "decision": decision,
        })

    def flush(self, run_tag: str = ""):
        if not self._run_id or not self._entries:
            return
        tag = f"{self._run_id}_{run_tag}" if run_tag else self._run_id
        path = self.output_dir / f"replay_{tag}.jsonl"
        with open(path, "a") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, default=str) + "\n")
        self._entries.clear()

    def replay(self, run_id: str) -> list[ReplayEntry]:
        """Load and return all replay entries for a given run_id."""
        path = self.output_dir / f"replay_{run_id}.jsonl"
        if not path.exists():
            return []
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(ReplayEntry(**json.loads(line)))
        return entries
