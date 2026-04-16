"""
Phase 8 — backtest/engine.py
Isolated backtest engine with separate DB and output dir.
No writes to live audit log.
"""
import json
import logging
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import SQLModel, Session, create_engine, Field
from sqlmodel.pool import StaticPool

from backtest.replay import ReplayStore
from backtest.report import compute_metrics, PerformanceMetrics
from agents.strategy_agent import StrategyAgent
from consensus.engine import ConsensusEngine
from execution.portfolio import Portfolio


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backtest DB schema (isolated)
# ---------------------------------------------------------------------------

class BacktestTrade(SQLModel, table=True):
    __tablename__ = "bt_trades"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    timestamp: str
    symbol: str
    signal: str
    quantity: float
    price: float
    pnl: float


class BacktestEquity(SQLModel, table=True):
    __tablename__ = "bt_equity"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    timestamp: str
    equity: float


class BacktestDecision(SQLModel, table=True):
    __tablename__ = "bt_decisions"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    timestamp: str
    symbol: str
    signal: str
    confidence: float
    reason: str
    escalated: bool


class BacktestEngine:
    """
    Isolated backtest engine.
    - Uses a separate SQLite DB (no live log pollution)
    - Output directory per run
    - Strict: no live market data, no audit log writes to main DB
    """

    def __init__(self, output_dir: str = "/opt/consensus-trade/data/backtests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id: Optional[str] = None
        self._replay_store = ReplayStore(str(self.output_dir))

    def run(
        self,
        symbol: str,
        strategies: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 10_000.0,
    ) -> list[dict]:
        """
        Run backtest for given symbol/strategies over date range.
        Returns list of result dicts.
        """
        run_id = f"{symbol.replace('/','_')}_{uuid.uuid4().hex[:8]}"
        self._run_id = run_id

        # Isolated DB
        bt_db = self.output_dir / f"backtest_{run_id}.db"
        engine = self._init_backtest_db(str(bt_db))

        portfolio = Portfolio(cash=initial_cash)
        consensus = ConsensusEngine()

        equity_curve = [initial_cash]
        trade_pnls = []
        consensus_override_count = 0
        decisions = []

        # Simple date iteration (in production: loop through trading days)
        # For now: run consensus logic once per strategy and record a single "decision"
        for strat_name in strategies:
            strategy_config = self._load_strategy(symbol, strat_name)

            agent = StrategyAgent(
                agent_id=f"bt-{strat_name}",
                model="qwen3:32b",  # backtest uses cached/local model
                strategy_config=strategy_config,
            )

            # In a real backtest we'd iterate over OHLCV bars
            # Here we simulate one decision per strategy
            try:
                signal = agent.generate(symbol=symbol, timeframe="1h", sentiment="neutral")
            except Exception as e:
                log.warning("[Backtest] %s signal error: %s", strat_name, e)
                continue

            decision = consensus.resolve(
                asset=symbol,
                asset_class=self._detect_asset_class(symbol),
                signals=[signal],
                research=None,
                trade_validator_ok=True,
            )

            decisions.append({
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "signal": decision.signal,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "escalated": decision.escalated,
            })

            if decision.escalated:
                consensus_override_count += 1

            # Write to isolated backtest DB
            with Session(engine) as sess:
                sess.add(BacktestDecision(**decisions[-1]))
                sess.commit()

        # Compute final metrics
        equity_curve.append(initial_cash)  # no change in this simplified version
        metrics = compute_metrics(equity_curve, trade_pnls, consensus_override_count)

        # Store result
        result = {
            "run_id": run_id,
            "strategy_name": ",".join(strategies),
            "asset": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_return": metrics.total_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "trade_count": metrics.trade_count,
            "metrics_json": json.dumps({
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "consensus_override_rate": metrics.consensus_override_rate,
            }),
        }

        with Session(engine) as sess:
            br = BacktestResultTable(**result)
            sess.add(br)
            sess.commit()

        # Flush replay store
        self._replay_store.flush(run_tag=symbol.replace("/","_"))

        return [result]

    def _init_backtest_db(self, db_path: str):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)
        return engine

    @staticmethod
    def _detect_asset_class(symbol: str) -> str:
        if "/" in symbol:
            return "crypto"
        return "stocks"

    @staticmethod
    def _load_strategy(symbol: str, strategy_name: str) -> dict:
        """Load strategy config from config dir (or return minimal default)."""
        asset_class = BacktestEngine._detect_asset_class(symbol)
        config_path = Path(f"/opt/consensus-trade/config/strategies/{asset_class}/{strategy_name}.yaml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {
            "name": strategy_name,
            "asset_class": asset_class,
            "indicators": [{"type": "RSI", "period": 14}],
            "prompt_template": "Output JSON: {signal: \"HOLD\", confidence: 0.5, reasoning: \"backtest\"}",
        }


# Separate table class for backtest results
class BacktestResultTable(SQLModel, table=True):
    __tablename__ = "bt_results"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    strategy_name: str = Field(index=True)
    asset: str = Field(index=True)
    start_date: str
    end_date: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    metrics_json: str = "{}"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
