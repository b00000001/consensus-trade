"""
Phase 1a — orchestrator.py
APScheduler loop, ties all layers together.
"""
import logging
import os
import signal
import sys
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger; from apscheduler.triggers.cron import CronTrigger

from src.config_loader import get_assets, get_backends, load_strategy, hot_reload_all
from src.market_data_fetcher import MarketDataFetcher
from src.db_models import init_db

from agents.strategy_agent import StrategyAgent
from agents.research_agent import ResearchAgent
from agents.agent_pool import AgentPool
from agents.claude_judge import ClaudeJudge

from consensus.engine import ConsensusEngine
from consensus.escalation import EscalationTracker

from risk.calculator import RiskProfile
from risk.trade_validator import validate_full_trade
from risk.circuit_breaker import CircuitBreaker
from risk.budget_guard import BudgetGuard

from execution.portfolio import Portfolio
from execution.paper_trader import PaperTrader

from dashboard.app import create_dashboard, update_execution_feed_event, update_consensus_log_event, update_agent_votes, update_risk_status


log = logging.getLogger(__name__)
from apscheduler.triggers.interval import IntervalTrigger; from apscheduler.triggers.cron import CronTrigger

from src.config_loader import get_assets, get_backends, load_strategy, hot_reload_all
from src.market_data_fetcher import MarketDataFetcher
from src.db_models import init_db

from agents.strategy_agent import StrategyAgent
from agents.research_agent import ResearchAgent
from agents.agent_pool import AgentPool
from agents.claude_judge import ClaudeJudge

from consensus.engine import ConsensusEngine
from consensus.escalation import EscalationTracker

from risk.calculator import RiskProfile
from risk.trade_validator import validate_full_trade
from risk.circuit_breaker import CircuitBreaker
from risk.budget_guard import BudgetGuard

from execution.portfolio import Portfolio
from execution.paper_trader import PaperTrader

from dashboard.app import create_dashboard, update_execution_feed_event, update_consensus_log_event, update_agent_votes, update_risk_status


log = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Main orchestrator: APScheduler-driven loop that ties all system layers.
    """

    def __init__(
        self,
        db_path: str = "db/live.db",
        poll_interval_seconds: int = 60,
        starting_cash: float = 10_000.0,
    ):
        self.poll_interval = poll_interval_seconds
        self._scheduler = BackgroundScheduler()
        self._running = False
        self._started = False

        # ---- Init all layers ----
        self.engine, self.db_writer = init_db(db_path)

        self.portfolio = Portfolio(cash=starting_cash)
        self.circuit_breaker = CircuitBreaker()
        self.budget_guard = BudgetGuard()

        self.market_fetcher = MarketDataFetcher()
        self.judge = ClaudeJudge()
        self.escalation_tracker = EscalationTracker()

        self.research_agent = ResearchAgent()

        # Build strategy agents from config
        self.strategy_agents = self._build_strategy_agents()
        self.agent_pool = AgentPool(agents=self.strategy_agents, research_agent=self.research_agent)
        self.consensus_engine = ConsensusEngine(
            judge=self.judge,
            escalation_tracker=self.escalation_tracker,
        )

        self.paper_trader = PaperTrader(
            portfolio=self.portfolio,
            db_writer=self.db_writer,
        )

        # ---- Dashboard ----
        self.dash_app = create_dashboard(
            paper_trader=self.paper_trader,
            portfolio=self.portfolio,
        )

    def _build_strategy_agents(self):
        """Load strategy agents from config."""
        agents = []
        assets = get_assets()
        backends = get_backends()

        # Map backend IDs to models
        backend_map = {b["id"]: b for b in backends.get("backends", [])}
        signal_backends = [b for b in backend_map.values() if b.get("role") == "signal" and b.get("includeInFanout")]

        for asset_class, asset_list in assets.items():
            strategies = asset_list.get("strategies", [])
            for strat_name in strategies:
                strat_cfg = load_strategy(asset_class, strat_name)
                if not strat_cfg:
                    continue
                # Pick a backend
                be = signal_backends[0] if signal_backends else {
                    "id": "signal-qwen", "model": "qwen3:32b", "adapter": "ollama"
                }
                agent = StrategyAgent(
                    agent_id=f"{asset_class}-{strat_name}",
                    model=be.get("model", "qwen3:32b"),
                    strategy_config=strat_cfg,
                    market_fetcher=self.market_fetcher,
                )
                agents.append(agent)
        return agents

    # ---- APScheduler jobs ----

    def _run_trading_cycle(self):
        """Main trading cycle — runs on schedule."""
        try:
            log.info("=== Trading cycle started ===")
            assets = get_assets()

            for asset_class, asset_data in assets.items():
                for symbol in asset_data.get("symbols", []):
                    self._process_asset(symbol, asset_class)

            # Update dashboard
            update_risk_status(self.circuit_breaker.status())
            update_consensus_log_event({
                "asset": "system",
                "signal": "cycle",
                "confidence": 1.0,
                "reason": "cycle_complete",
                "escalated": False,
            })
            log.info("=== Trading cycle complete ===")
        except Exception as e:
            log.error("Trading cycle error: %s", e, exc_info=True)

    def _process_asset(self, symbol: str, asset_class: str):
        """Process a single asset through the full pipeline."""
        try:
            # 1. Fetch market data
            snapshot = self.market_fetcher.fetch_price(symbol)
            if not snapshot:
                log.warning("No price for %s, skipping", symbol)
                return

            current_price = snapshot.price

            # 2. Run agent pool (research + signals)
            pool_result = self.agent_pool.run_full_cycle(symbol)

            if not pool_result.signals:
                log.info("No signals for %s", symbol)
                return

            # 3. Risk check (Tier 0)
            trade_val_ok = True
            for sig in pool_result.signals:
                result = validate_full_trade(
                    signal=sig.signal,
                    cash_available=self.portfolio.cash,
                    position_quantity=self.portfolio.get_position(symbol).quantity if self.portfolio.get_position(symbol) else 0.0,
                    price=current_price,
                    quantity=0.001,  # minimum trade unit
                )
                if not result.ok:
                    trade_val_ok = False
                    log.info("Trade validation failed for %s: %s", symbol, result.reason)

            # 4. Consensus
            decision = self.consensus_engine.resolve(
                asset=symbol,
                asset_class=asset_class,
                signals=pool_result.signals,
                research=pool_result.research,
                trade_validator_ok=trade_val_ok,
                circuit_breaker_tripped=self.circuit_breaker.is_tripped(),
                budget_cap_reached=self.budget_guard.cap_reached(symbol),
            )

            # Build agent -> model map for dashboard display
            agent_model_map = {s.agent_id: s.model for s in pool_result.signals}
            if decision.escalated:
                agent_model_map["claude-judge"] = "claude-sonnet-4-6"

            # 5. Dashboard updates
            update_consensus_log_event({
                "asset": symbol,
                "signal": decision.signal,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "escalated": decision.escalated,
                "agent_ids": [s.agent_id for s in pool_result.signals],
            }, agent_models=agent_model_map)
            update_agent_votes(
                symbol,
                {s.agent_id: s.signal for s in pool_result.signals},
                agent_models=agent_model_map,
            )

            # 6. Execute if not HOLD
            if decision.signal != "HOLD":
                trade = self.paper_trader.propose(
                    asset=symbol,
                    symbol=symbol,
                    signal=decision.signal,
                    price=current_price,
                    agent_id="consensus",
                    confidence=decision.confidence,
                    rationale=decision.reason,
                    price_fetcher=self.market_fetcher,
                )
                update_execution_feed_event({
                    "trade_id": trade.trade_id,
                    "asset": symbol,
                    "signal": decision.signal,
                    "quantity": trade.quantity,
                    "price": current_price,
                    "status": "PROPOSED",
                })

        except Exception as e:
            log.error("Error processing %s: %s", symbol, e)

    def _run_research_pipeline(self):
        """Background research pipeline — runs independently, cached."""
        try:
            log.info("Running research pipeline...")
            report = self.research_agent.run_pipeline(limit=10)
            log.info("Research: macro_risk=%.2f sentiment=%.2f", report.macro_risk, report.sentiment_score)
        except Exception as e:
            log.error("Research pipeline error: %s", e)

    def _run_heartbeat(self):
        """
        Ping all configured agents, log latency to DB.
        Alert via Telegram if any agent is down for >3 consecutive checks.
        """
        from src.db_models import BotHeartbeat
        from src.config_loader import get_backends

        backends = get_backends()
        ollama_base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

        for be in backends.get("backends", []):
            agent_id = be.get("id", "")
            adapter_type = be.get("adapter", "")
            model = be.get("model", "")

            if adapter_type == "ollama":
                import requests
                start = time.monotonic()
                try:
                    resp = requests.get(f"{ollama_base}/api/tags", timeout=5)
                    resp.raise_for_status()
                    latency_ms = (time.monotonic() - start) * 1000
                    BotHeartbeat.record(agent_id, model, latency_ms, success=True)
                    log.debug("[%s] Ollama heartbeat OK: %.0fms", agent_id, latency_ms)
                except Exception as e:
                    BotHeartbeat.record(agent_id, model, 0.0, success=False, error=str(e))
                    log.warning("[%s] Ollama heartbeat failed: %s", agent_id, e)
                    if BotHeartbeat.consecutive_failures(agent_id, threshold=3):
                        log.error("[%s] DOWN — alerting", agent_id)
                        # TODO: trigger Telegram alert via OpenClaw cron
            elif adapter_type == "minimax":
                from agents.adapters import MiniMaxAdapter
                a = MiniMaxAdapter(agent_id=agent_id, model=model)
                latency = a.ping()
                if latency is not None:
                    BotHeartbeat.record(agent_id, model, latency, success=True)
                else:
                    BotHeartbeat.record(agent_id, model, 0.0, success=False, error="ping_failed")
                    if BotHeartbeat.consecutive_failures(agent_id, threshold=3):
                        log.error("[%s] MiniMax DOWN — alerting", agent_id)
            elif adapter_type == "claude-cli":
                from agents.adapters import ClaudeAdapter
                a = ClaudeAdapter(agent_id=agent_id, model=model)
                latency = a.ping()
                if latency is not None:
                    BotHeartbeat.record(agent_id, model, latency, success=True)
                else:
                    BotHeartbeat.record(agent_id, model, 0.0, success=False, error="ping_failed")
                    if BotHeartbeat.consecutive_failures(agent_id, threshold=3):
                        log.error("[%s] Claude DOWN — alerting", agent_id)
            elif adapter_type == "gemini-cli":
                from agents.adapters import GeminiAdapter
                a = GeminiAdapter(agent_id=agent_id, model=model)
                latency = a.ping()
                if latency is not None:
                    BotHeartbeat.record(agent_id, model, latency, success=True)
                else:
                    BotHeartbeat.record(agent_id, model, 0.0, success=False, error="ping_failed")
                    if BotHeartbeat.consecutive_failures(agent_id, threshold=3):
                        log.error("[%s] Gemini DOWN — alerting", agent_id)

    def _keep_ollama_warm(self):
        """Periodic ping to keep Ollama models warm."""
        import requests
        try:
            requests.get("http://localhost:11434/api/generate", timeout=5, json={
                "model": "qwen3:32b",
                "prompt": ".",
                "stream": False,
            })
        except Exception:
            pass

    # ---- Lifecycle ----

    def start(self):
        if self._started:
            return
        self._running = True
        self._started = True

        # Main trading cycle
        self._scheduler.add_job(
            self._run_trading_cycle,
            trigger=IntervalTrigger(seconds=self.poll_interval),
            id="trading_cycle",
            replace_existing=True,
        )

        # Background research (every 5 min)
        self._scheduler.add_job(
            self._run_research_pipeline,
            trigger=IntervalTrigger(seconds=300),
            id="research_pipeline",
            replace_existing=True,
        )

        # Keep Ollama warm (every 2 min)
        self._scheduler.add_job(
            self._keep_ollama_warm,
            trigger=IntervalTrigger(seconds=120),
            id="ollama_warm",
            replace_existing=True,
        )

        # Heartbeat — ping all agents every 2 min
        self._scheduler.add_job(
            self._run_heartbeat,
            trigger=IntervalTrigger(seconds=120),
            id="heartbeat",
            replace_existing=True,
        )

        self._scheduler.start()
        log.info("Orchestrator started — polling every %ds", self.poll_interval)

    def stop(self):
        self._running = False
        self._scheduler.shutdown(wait=False)
        self.paper_trader.stop()
        self.db_writer.stop()
        log.info("Orchestrator stopped")

    def run(self):
        """Run forever (blocking)."""
        self.start()
        try:
            while self._running:
                signal.pause()
        except KeyboardInterrupt:
            self.stop()
