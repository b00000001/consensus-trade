"""
Phase 4 — agents/claude_judge.py
Tier 2 — only called when confidence < 0.65, uses Claude CLI subprocess adapter.
Falls back to HOLD if Claude CLI is unavailable or unauthenticated.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import AgentSignal
from src.redis_setup import TTL_JUDGE_SYNTHESIS, make_judge_synthesis_key, cache_get, cache_set
from agents.adapters import ClaudeAdapter


log = logging.getLogger(__name__)


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
    Tier 2 arbiter using Claude CLI subprocess.
    Called once per cycle ONLY when local agents disagree below
    confidence threshold (0.65). Uses judge synthesis cache to skip
    redundant calls.
    """

    def __init__(
        self,
        agent_id: str = "claude-judge",
        model: str = "claude-sonnet-4-6",
        timeout: int = 120,
    ):
        self.adapter = ClaudeAdapter(agent_id=agent_id, model=model, timeout=timeout)
        self.model = model

    def health_check(self) -> bool:
        return self.adapter.health_check()

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
        cache_key = make_judge_synthesis_key([s.__dict__ for s in signals])
        cached = cache_get(cache_key)
        if cached:
            log.info("[%s] Judge cache hit, skipping CLI call", self.adapter.agent_id)
            return JudgeDecision(**{**cached, "agent_id": self.adapter.agent_id})

        # Build prompt
        signal_summary = self._summarize_signals(signals)
        research_text = json.dumps(research, default=str) if research else "No research available."

        system = (
            "You are a senior trading arbiter. Output ONLY valid JSON:\n"
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
                agent_id=self.adapter.agent_id,
                signal="HOLD",
                confidence=0.0,
                rationale="claude_unavailable",
                stop_loss=0.0,
            )

        decision = JudgeDecision(
            agent_id=self.adapter.agent_id,
            signal=str(result.get("signal", "HOLD")).upper(),
            confidence=float(result.get("confidence", 0.0)),
            rationale=str(result.get("rationale", "")),
            stop_loss=float(result.get("stop_loss", 0.0)),
        )

        cache_set(cache_key, decision.__dict__, TTL_JUDGE_SYNTHESIS)
        return decision

    @staticmethod
    def _summarize_signals(signals: list[AgentSignal]) -> str:
        lines = []
        for s in signals:
            lines.append(
                f"- Agent: {s.agent_id} | Signal: {s.signal} | "
                f"Confidence: {s.confidence:.2f} | Reasoning: {s.reasoning}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fence(data: str) -> str:
        return f"<data>\n{data}\n</data>"

    def _call_claude(self, prompt: str, system: str) -> Optional[dict]:
        """Call Claude CLI subprocess. Returns parsed JSON dict or None on failure."""
        # The system prompt is prepended to the user prompt for claude -p
        full_prompt = f"System: {system}\n\n{prompt}"

        resp = self.adapter.call_json(full_prompt)
        if not resp.success:
            log.warning("[%s] Claude CLI call failed: %s", self.adapter.agent_id, resp.error)
            return None

        try:
            return json.loads(resp.text)
        except json.JSONDecodeError as e:
            log.warning(
                "[%s] Claude response not valid JSON: %s | raw: %s",
                self.adapter.agent_id, e, resp.text[:200]
            )
            return None