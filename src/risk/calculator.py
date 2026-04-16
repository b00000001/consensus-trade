"""
Phase 2 — risk/calculator.py
Tier 0: pure arithmetic risk checks — concentration limit, daily loss limit, max leverage.
No LLM calls.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskProfile:
    max_position_pct: float = 0.20      # max 20% of portfolio in single asset
    max_leverage: float = 3.0           # max 3× leverage
    max_daily_loss_pct: float = 0.05   # max 5% daily portfolio loss
    max_concentration_pct: float = 0.30  # max 30% in correlated assets


@dataclass
class RiskCheckResult:
    ok: bool
    reason: str
    risk_score: float  # 0.0 (safe) – 1.0 (dangerous)


def check_concentration(
    position_value: float,
    portfolio_value: float,
    max_pct: float = 0.20,
) -> RiskCheckResult:
    """Check if a new position would exceed concentration limit."""
    if portfolio_value <= 0:
        return RiskCheckResult(False, "zero_portfolio", 1.0)

    ratio = position_value / portfolio_value
    if ratio > max_pct:
        return RiskCheckResult(False, f"concentration_exceeded {ratio:.2%}>{max_pct:.0%}", ratio)
    return RiskCheckResult(True, "ok", ratio)


def check_leverage(
    notional_value: float,
    equity: float,
    max_leverage: float = 3.0,
) -> RiskCheckResult:
    """Check if leverage exceeds limit."""
    if equity <= 0:
        return RiskCheckResult(False, "zero_equity", 1.0)

    leverage = notional_value / equity
    if leverage > max_leverage:
        return RiskCheckResult(False, f"leverage_exceeded {leverage:.1f}×>{max_leverage}×", min(leverage / max_leverage, 1.0))
    return RiskCheckResult(True, "ok", leverage / max_leverage)


def check_daily_loss(
    current_loss_pct: float,
    max_loss_pct: float = 0.05,
) -> RiskCheckResult:
    """Check if cumulative daily loss exceeds threshold."""
    if current_loss_pct > max_loss_pct:
        return RiskCheckResult(False, f"daily_loss_exceeded {current_loss_pct:.2%}>{max_loss_pct:.0%}", current_loss_pct / max_loss_pct)
    return RiskCheckResult(True, "ok", current_loss_pct / max_loss_pct)


def compute_risk_score(
    position_value: float,
    portfolio_value: float,
    notional_value: float,
    equity: float,
    daily_loss_pct: float,
    profile: Optional[RiskProfile] = None,
) -> tuple[RiskCheckResult, RiskCheckResult, RiskCheckResult]:
    """
    Run all three risk checks. Returns (concentration, leverage, daily_loss).
    """
    profile = profile or RiskProfile()
    conc = check_concentration(position_value, portfolio_value, profile.max_position_pct)
    lev = check_leverage(notional_value, equity, profile.max_leverage)
    dl = check_daily_loss(daily_loss_pct, profile.max_daily_loss_pct)
    return conc, lev, dl


def aggregate_risk_score(conc: RiskCheckResult, lev: RiskCheckResult, dl: RiskCheckResult) -> float:
    """Weighted average of individual risk scores."""
    return 0.3 * conc.risk_score + 0.4 * lev.risk_score + 0.3 * dl.risk_score
