"""
Phase 8 — backtest/report.py
Metrics (Sharpe, Sortino, Calmar, drawdown) + on-demand Claude narrative.
"""
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

from src.redis_setup import cache_get, cache_set, TTL_MARKET_ANALYSIS


log = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trade_count: int
    consensus_override_rate: float  # how often Claude overrode Tier 1


def compute_metrics(
    equity_curve: list[float],
    trade_pnls: list[float],
    consensus_override_count: int,
) -> PerformanceMetrics:
    """
    Compute all performance metrics from equity curve and trade PnLs.
    """
    if not equity_curve or len(equity_curve) < 2:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Returns
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] != 0:
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] != 0 else 0.0

    # Sharpe / Sortino
    if returns:
        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 1e-9

        downside_returns = [r for r in returns if r < 0]
        downside_std = math.sqrt(sum(r ** 2 for r in downside_returns) / len(downside_returns)) if downside_returns else 1e-9

        sharpe = (avg_ret / std_dev) * math.sqrt(252) if std_dev != 0 else 0.0
        sortino = (avg_ret / downside_std) * math.sqrt(252) if downside_std != 0 else 0.0
    else:
        sharpe = sortino = 0.0

    # Calmar
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    calmar = (total_return / max_dd) if max_dd != 0 else 0.0

    # Win rate
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

    # Consensus override rate
    override_rate = consensus_override_count / len(trade_pnls) if trade_pnls else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        trade_count=len(trade_pnls),
        consensus_override_rate=override_rate,
    )


def generate_narrative(metrics: PerformanceMetrics, strategy_name: str) -> str:
    """
    On-demand Claude narrative for a backtest result.
    Cached in Redis to avoid repeat CLI calls.
    """
    cache_key = f"backtest:narrative:{strategy_name}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    prompt = (
        f"Generate a brief performance narrative for strategy '{strategy_name}'.\n"
        f"Total Return: {metrics.total_return:.2%}\n"
        f"Sharpe: {metrics.sharpe_ratio:.2f}, Sortino: {metrics.sortino_ratio:.2f}, Calmar: {metrics.calmar_ratio:.2f}\n"
        f"Max Drawdown: {metrics.max_drawdown:.2%}, Win Rate: {metrics.win_rate:.1%}\n"
        f"Trades: {metrics.trade_count}, Avg Win: ${metrics.avg_win:.2f}, Avg Loss: ${metrics.avg_loss:.2f}\n"
        f"Consensus Override Rate: {metrics.consensus_override_rate:.1%}\n\n"
        f"Write a 2-3 sentence summary."
    )

    try:
        from agents.adapters import ClaudeAdapter
        adapter = ClaudeAdapter(agent_id="report-claude", model="claude-sonnet-4-6")
        resp = adapter.call_json(prompt)
        if resp.success:
            cache_set(cache_key, resp.text, 3600)
            return resp.text
        else:
            log.warning("Claude narrative failed: %s", resp.error)
            return f"Performance summary: {strategy_name} returned {metrics.total_return:.2%} with Sharpe {metrics.sharpe_ratio:.2f}."
    except Exception as e:
        log.warning("Narrative generation failed: %s", e)
        return f"Performance summary: {strategy_name} returned {metrics.total_return:.2%} with Sharpe {metrics.sharpe_ratio:.2f}."