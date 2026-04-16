"""
Phase 4 — agents/claude_judge.py
Tier 2 — only called when confidence < 0.7, uses judge synthesis cache.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import requests

from agents.base_agent import AgentSignal
from src.redis_setup import make_judge_synthesis_key, cache_get, cache_set, TTL_JUDGE_SYNTHESIS


log = logging.getLogger(__name__)

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_BASE_URL = "https://api.anthropic.com/v1"
CLAUDE_MODEL = os.environ.get("CLAUDE_JUDGE_MODEL", "claude-sonnet-4-6")
CLAUDE_TIMEOUT = 120


@dataclass
class JudgeDecision:
    agent_id: str
    signal: str          # BUY | SELL | HOLD
    confidence: float
    rationale: str
    stop_loss: float
    metadata: dict = field(default_factory=dict)


class ClaudeJudge:
    """
    Tier 2 arbiter. Called once per cycle ONLY when local agents disagree
    below confidence threshold (0.7). Uses judge synthesis cache to skip
    redundant calls.
    """

    def __init__(self, agent_id: str = "claude-sonnet"):
        self.agent_id = agent_id
        self.model = CLAUDE_MODEL

    def arbitrate(
        self,
        asset: str,
        signals: list[AgentSignal],
        research: Optional[dict] = None,
    ) -> JudgeDecision:
        """
        Run Claude judge on combined Tier 1 signals + research.
        Checks judge synthesis cache first.
        """
        # Build cache key from signal hashes
        cache_key = make_judge_synthesis_key([s.__dict__ for s in signals])
        cached = cache_get(cache_key)
        if cached:
            log.info("[%s] Judge cache hit, skipping API call", self.agent_id)
            return JudgeDecision(**{**cached, "agent_id": self.agent_id})

        # Build prompt
        signal_summary = self._summarize_signals(signals)
        research_text = json.dumps(research, default=str) if research else "No research available."

        system = (
            "You are a senior trading arbiter. You must respond with ONLY valid JSON:\n"
            '{"signal": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, '
            '"rationale": "...", "stop_loss": 0.0}'
        )
        prompt = (
            f"Analyze the following trading signals for {asset} and produce a final decision.\n\n"
            f"Signal Summary:\n{self._fence(signal_summary)}\n\n"
            f"Research:\n{self._fence(research_text)}\n\n"
            f"Output ONLY JSON."
        )

        result = self._call_claude(prompt=prompt, system=system)
        if result is None:
            return JudgeDecision(
                agent_id=self.agent_id,
                signal="HOLD",
                confidence=0.0,
                rationale="claude_unavailable",
                stop_loss=0.0,
            )

        stop_loss = float(result.get("stop_loss", 0.0))
        decision = JudgeDecision(
            agent_id=self.agent_id,
            signal=str(result.get("signal", "HOLD")).upper(),
            confidence=float(result.get("confidence", 0.0)),
            rationale=str(result.get("rationale", "")),
            stop_loss=stop_loss,
        )

        # Cache result
        cache_set(cache_key, decision.__dict__, TTL_JUDGE_SYNTHESIS)

        return decision

    @staticmethod
    def _summarize_signals(signals: list[AgentSignal]) -> str:
        lines = []
        for s in signals:
            lines.append(
                f"- Agent: {s.agent_id} | Model: {s.model} | "
                f"Signal: {s.signal} | Confidence: {s.confidence:.2f} | "
                f"Reasoning: {s.reasoning}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fence(data: str) -> str:
        return f"<market_data>\n{data}\n</market_data>"

    def _call_claude(self, prompt: str, system: str) -> Optional[dict]:
        import time

        if not CLAUDE_API_KEY:
            log.error("[%s] ANTHROPIC_API_KEY not set", self.agent_id)
            return None

        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages = [
            {"role": "user", "content": f"System: {system}\n\n{prompt}"}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,
        }

        try:
            resp = requests.post(
                f"{CLAUDE_BASE_URL}/messages",
                headers=headers, json=body, timeout=CLAUDE_TIMEOUT,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                log.warning("[%s] Claude 429 — sleeping %ds", self.agent_id, retry_after)
                time.sleep(retry_after)
                return self._call_claude(prompt, system)
            resp.raise_for_status()
            data = resp.json()
            raw = data["content"][0]["text"].strip()
            return json.loads(raw)
        except Exception as e:
            log.warning("[%s] Claude judge error: %s", self.agent_id, e)
            return None
