"""
Phase 5 — consensus/engine.py
Weighted consensus + escalation to Claude judge.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from agents.base_agent import AgentSignal
from agents.research_agent import ResearchReport
from agents.claude_judge import ClaudeJudge
from consensus.escalation import EscalationTracker, EscalationConfig
from src.config_loader import get_consensus_config


log = logging.getLogger(__name__)


# Default consensus thresholds per asset class
DEFAULT_QUORUM = 0.6
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
RESEARCH_WEIGHT = 1.5


@dataclass
class Decision:
    signal: str          # BUY | SELL | HOLD
    confidence: float
    reason: str          # e.g. "consensus", "judge_override", "validation_failed"
    escalated: bool = False
    stop_loss: float = 0.0
    agent_votes: dict = None  # agent_id -> signal

    def __post_init__(self):
        if self.agent_votes is None:
            self.agent_votes = {}


class ConsensusEngine:
    """
    Central consensus logic.
    Applies Tier 0 gates, quorum check, research veto,
    weighted Tier 1 consensus, then Tier 2 escalation if needed.
    """

    def __init__(
        self,
        judge: Optional[ClaudeJudge] = None,
        escalation_tracker: Optional[EscalationTracker] = None,
    ):
        self.judge = judge or ClaudeJudge()
        self.escalation = escalation_tracker or EscalationTracker()

    def resolve(
        self,
        asset: str,
        asset_class: str,
        signals: list[AgentSignal],
        research: Optional[ResearchReport],
        # Tier 0 gates (passed in from orchestrator)
        trade_validator_ok: bool = True,
        circuit_breaker_tripped: bool = False,
        budget_cap_reached: bool = False,
    ) -> Decision:
        """
        Main entry point. Applies full decision pipeline.
        """
        # ---- Tier 0 gates ----
        if not trade_validator_ok:
            return Decision(signal="HOLD", confidence=0.0, reason="validation_failed")

        if circuit_breaker_tripped:
            return Decision(signal="HOLD", confidence=0.0, reason="circuit_breaker")

        if budget_cap_reached:
            return Decision(signal="HOLD", confidence=0.0, reason="budget_cap_reached")

        # ---- Load consensus config for asset class ----
        config = get_consensus_config(asset_class)
        quorum = config.get("quorum", DEFAULT_QUORUM)
        confidence_threshold = config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        research_weight = config.get("research_agent_weight", RESEARCH_WEIGHT)
        veto_on_macro_risk = config.get("veto_on_macro_risk", True)
        macro_risk_threshold = 0.8

        # ---- Quorum check ----
        if len(signals) == 0:
            return Decision(signal="HOLD", confidence=0.0, reason="no_signals")

        # ---- Research veto ----
        if veto_on_macro_risk and research and research.macro_risk > macro_risk_threshold:
            return Decision(signal="HOLD", confidence=research.macro_risk, reason="macro_veto")

        # ---- Weighted consensus (Tier 1) ----
        decision = self._weighted_consensus(signals, research, research_weight, confidence_threshold)

        # ---- Tier 2 escalation check ----
        if decision.confidence < confidence_threshold:
            cache_key = self.escalation.make_escalation_key(asset, signals)
            if self.escalation.can_escalate(asset):
                self.escalation.record_escalation(asset)
                judge_result = self.judge.arbitrate(asset, signals, research.__dict__ if research else None)
                return Decision(
                    signal=judge_result.signal,
                    confidence=judge_result.confidence,
                    reason="judge_override",
                    escalated=True,
                    stop_loss=judge_result.stop_loss,
                    agent_votes={s.agent_id: s.signal for s in signals},
                )
            else:
                decision.reason = "low_confidence_cooldown"
                return decision

        decision.agent_votes = {s.agent_id: s.signal for s in signals}
        return decision

    @staticmethod
    def _weighted_consensus(
        signals: list[AgentSignal],
        research: Optional[ResearchReport],
        research_weight: float,
        confidence_threshold: float,
    ) -> Decision:
        """
        Score each BUY/SELL signal by confidence, weight research if available.
        """
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for s in signals:
            weight = s.confidence
            total_weight += weight
            if s.signal == "BUY":
                buy_score += weight
            elif s.signal == "SELL":
                sell_score += weight

        # Add research weight
        if research:
            total_weight += research_weight
            sentiment = research.sentiment_score  # -1 to 1
            if sentiment > 0:
                buy_score += research_weight * sentiment
            else:
                sell_score += research_weight * abs(sentiment)

        if total_weight == 0:
            return Decision(signal="HOLD", confidence=0.0, reason="no_weighted_signals")

        net = buy_score - sell_score
        confidence = min(abs(net) / total_weight, 1.0)

        if buy_score > sell_score:
            signal = "BUY"
        elif sell_score > buy_score:
            signal = "SELL"
        else:
            signal = "HOLD"

        return Decision(
            signal=signal,
            confidence=confidence,
            reason="weighted_consensus",
        )
