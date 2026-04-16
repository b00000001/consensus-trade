"""
Phase 6 — execution/paper_trader.py
PaperTrader — 0.1% slippage model, TRADE_PROPOSED→5s TTL→TRADE_EXECUTED gate,
hash-chained audit log.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from queue import Queue, Empty

from execution.portfolio import Portfolio
from execution.audit_log import compute_record_hash, make_audit_payload


log = logging.getLogger(__name__)

SLIPPAGE = 0.001  # 0.1%
PROPOSED_TTL_SECONDS = 5.0
DEFAULT_QUANTITY = 0.001  # BTC/ETH minimum trade unit


@dataclass
class ProposedTrade:
    trade_id: int
    asset: str
    symbol: str
    signal: str
    quantity: float
    price: float
    slippage: float
    agent_id: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    proposed_at: float = field(default_factory=time.time)
    status: str = "PROPOSED"  # PROPOSED | EXECUTED | REJECTED | EXPIRED


class PaperTrader:
    """
    Paper trading engine with:
    - 0.1% slippage on all executions
    - TRADE_PROPOSED → 5s TTL → TRADE_EXECUTED gate
    - Hash-chained audit log written via db_writer
    - Thread-safe operation
    """

    def __init__(
        self,
        portfolio: Portfolio,
        db_writer,  # DbWriter instance
        slippage: float = SLIPPAGE,
        proposed_ttl: float = PROPOSED_TTL_SECONDS,
    ):
        self.portfolio = portfolio
        self.db_writer = db_writer
        self.slippage = slippage
        self.proposed_ttl = proposed_ttl

        self._proposed_queue: Queue[ProposedTrade] = Queue()
        self._trade_counter = 0
        self._last_audit_hash = ""
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._execution_worker, daemon=True)
        self._stop_event = threading.Event()
        self._worker_thread.start()

    # ---- Public API ----

    def propose(
        self,
        asset: str,
        symbol: str,
        signal: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        agent_id: Optional[str] = None,
        confidence: Optional[float] = None,
        rationale: Optional[str] = None,
        price_fetcher=None,  # MarketDataFetcher or similar
    ) -> ProposedTrade:
        """
        Propose a trade. Returns ProposedTrade with PROPOSED status.
        Immediately enqueues for execution (TTL gate handled by worker).
        """
        # Resolve quantity/price if not provided
        if quantity is None:
            quantity = DEFAULT_QUANTITY
        if price is None and price_fetcher:
            snap = price_fetcher.fetch_price(symbol)
            price = snap.price if snap else 0.0
        if price is None or price <= 0:
            price = 0.0

        with self._lock:
            self._trade_counter += 1
            trade = ProposedTrade(
                trade_id=self._trade_counter,
                asset=asset,
                symbol=symbol,
                signal=signal,
                quantity=quantity,
                price=price,
                slippage=self.slippage,
                agent_id=agent_id,
                confidence=confidence,
                rationale=rationale,
            )

        # Write PROPOSED audit record
        self._write_audit(trade, "TRADE_PROPOSED")

        self._proposed_queue.put(trade)
        log.info("[PaperTrader] PROPOSED trade %d: %s %s @ %s",
                 trade.trade_id, signal, symbol, price)
        return trade

    def status(self) -> dict:
        """Return current paper trader status."""
        return {
            "slippage": self.slippage,
            "proposed_ttl": self.proposed_ttl,
            "cash": self.portfolio.cash,
            "total_equity": self.portfolio.total_equity(),
            "positions": len(self.portfolio.positions),
            "total_pnl": self.portfolio.total_pnl(),
            "proposed_queue_size": self._proposed_queue.qsize(),
        }

    def stop(self):
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)

    # ---- Internal ----

    def _execution_worker(self):
        """
        Background worker: processes PROPOSED trades after TTL expires.
        Applies 0.1% slippage and executes against portfolio.
        """
        while not self._stop_event.is_set():
            try:
                trade = self._proposed_queue.get(timeout=0.5)
            except Empty:
                continue

            # Wait for TTL
            elapsed = time.time() - trade.proposed_at
            if elapsed < self.proposed_ttl:
                time.sleep(self.proposed_ttl - elapsed)

            if self._stop_event.is_set():
                break

            # Re-check portfolio sufficiency
            ok, reason = self._can_execute(trade)
            if not ok:
                trade.status = "REJECTED"
                self._write_audit(trade, "TRADE_REJECTED", reason=reason)
                log.warning("[PaperTrader] trade %d REJECTED: %s", trade.trade_id, reason)
                continue

            # Apply slippage
            exec_price = self._apply_slippage(trade.signal, trade.price)

            # Execute against portfolio
            success, msg = self.portfolio.apply_trade(
                symbol=trade.symbol,
                asset=trade.asset,
                signal=trade.signal,
                quantity=trade.quantity,
                price=exec_price,
                slippage=0.0,  # already slipped
            )

            if success:
                trade.status = "EXECUTED"
                trade.price = exec_price
                self._write_audit(trade, "TRADE_EXECUTED")
                log.info("[PaperTrader] trade %d EXECUTED: %s %s @ %s (slippage %.2f%%)",
                         trade.trade_id, trade.signal, trade.symbol, exec_price,
                         self.slippage * 100)
            else:
                trade.status = "REJECTED"
                self._write_audit(trade, "TRADE_REJECTED", reason=msg)
                log.warning("[PaperTrader] trade %d REJECTED: %s", trade.trade_id, msg)

    def _can_execute(self, trade: ProposedTrade) -> tuple[bool, str]:
        """Final validation before execution."""
        if trade.signal == "HOLD":
            return False, "hold_signal"
        if trade.price <= 0:
            return False, "zero_price"
        if trade.quantity <= 0:
            return False, "zero_quantity"

        pos = self.portfolio.get_position(trade.symbol)
        pos_qty = pos.quantity if pos else 0.0

        if trade.signal == "BUY":
            cost = trade.price * trade.quantity * (1 + self.slippage)
            if self.portfolio.cash < cost:
                return False, f"insufficient_cash"
        elif trade.signal == "SELL":
            if pos_qty < trade.quantity:
                return False, f"insufficient_position"
        return True, "ok"

    def _apply_slippage(self, signal: str, price: float) -> float:
        """Apply 0.1% slippage: BUY→higher price, SELL→lower price."""
        if signal == "BUY":
            return price * (1 + self.slippage)
        elif signal == "SELL":
            return price * (1 - self.slippage)
        return price

    def _write_audit(self, trade: ProposedTrade, action: str, reason: str = ""):
        """Write hash-chained audit log entry via db_writer."""
        with self._lock:
            payload = make_audit_payload(
                action=action,
                asset=trade.asset,
                symbol=trade.symbol,
                signal=trade.signal,
                quantity=trade.quantity,
                price=trade.price,
                status=trade.status,
                agent_id=trade.agent_id,
                confidence=trade.confidence,
                rationale=trade.rationale or reason,
                trade_id=trade.trade_id,
            )
            record_hash = compute_record_hash(self._last_audit_hash, payload)
            self.db_writer.enqueue("audit_log", {
                "prev_hash": self._last_audit_hash,
                "record_hash": record_hash,
                "action": action,
                "payload": json.dumps(payload, sort_keys=True),
            })
            self._last_audit_hash = record_hash
