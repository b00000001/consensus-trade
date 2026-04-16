"""
db_models.py — Consensus Trade
SQLModel entities + WAL mode + hash-chained audit log + DbWriter.
"""
import hashlib
import json
import queue
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Field, SQLModel, Session, create_engine
from sqlmodel.pool import StaticPool

DB_PATH = Path("/opt/consensus-trade/data/trades.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─── DbWriter — 100ms batched writes ─────────────────────────────────────────

class DbWriter:
    """Thread-safe batched DB writer. Accumulates writes and flushes every 100ms."""

    def __init__(self, engine):
        self._engine = engine
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        batch = []
        last = time.monotonic()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.05)
                batch.append(item)
            except queue.Empty:
                pass
            if batch and (time.monotonic() - last) >= 0.1:
                self._flush(batch)
                batch.clear()
                last = time.monotonic()
        while True:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush(batch)

    def _flush(self, batch):
        with Session(self._engine) as s:
            for row in batch:
                s.add(row)
            s.commit()

    def add(self, obj):
        self._q.put(obj)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)


def init_db(db_path: str):
    """Init SQLite WAL mode, return (engine, db_writer)."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.close()
    engine = create_engine(
        f"sqlite:///{p}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    SQLModel.metadata.create_all(engine)
    writer = DbWriter(engine)
    return engine, writer


# ─── Module-level engine (singleton) ────────────────────────────────────────────

_engine = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_engine.execute("PRAGMA journal_mode=WAL")
_engine.execute("PRAGMA synchronous=NORMAL")
_engine.execute("PRAGMA busy_timeout=5000")
_engine.execute("PRAGMA foreign_keys=ON")
_engine.close()

_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)
SQLModel.metadata.create_all(_engine)

_local = threading.local()


def get_session() -> Session:
    if not hasattr(_local, "session"):
        _local.session = Session(_engine)
    return _local.session


# ─── Hash-chain helpers ────────────────────────────────────────────────────────


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _chain_hash(prev: Optional[str], payload_hash: str) -> str:
    return _sha256((prev or "") + payload_hash)


# ─── TradeSignal ───────────────────────────────────────────────────────────────


class TradeSignal(SQLModel, table=True):
    __tablename__ = "trade_signals"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    direction: str  # "long" | "short" | "neutral"
    confidence: float  # 0.0 – 1.0
    agent: str
    model_used: str
    market_data_snapshot: str  # JSON blob

    def payload_digest(self) -> str:
        return _sha256(json.dumps({
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "agent": self.agent,
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True))


# ─── TradeExecution ───────────────────────────────────────────────────────────


class TradeExecution(SQLModel, table=True):
    __tablename__ = "trade_executions"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    direction: str  # "long" | "short"
    quantity: float
    price: float
    notional: float
    signal_id: int = Field(foreign_key="trade_signals.id")
    risk_check_passed: bool
    consensus_score: float
    execution_mode: str = "paper"  # "paper" | "live"


# ─── AuditLog ─────────────────────────────────────────────────────────────────


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str
    payload: str  # JSON blob
    payload_hash: str  # SHA-256 of payload
    prev_hash: str
    chain_hash: str  # SHA256(prev_hash + payload_hash)

    @classmethod
    def append(cls, event_type: str, payload: dict) -> "AuditLog":
        session = get_session()
        last: Optional[AuditLog] = session.query(cls).order_by(cls.id.desc()).first()
        prev = last.chain_hash if last else ""
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = _sha256(payload_json)
        chain_hash = _chain_hash(prev, payload_hash)
        entry = cls(
            event_type=event_type,
            payload=payload_json,
            payload_hash=payload_hash,
            prev_hash=prev,
            chain_hash=chain_hash,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry

    @classmethod
    def verify_chain(cls) -> tuple[bool, list[str]]:
        session = get_session()
        entries: list[AuditLog] = session.query(cls).order_by(cls.id).all()
        errors = []
        for i, entry in enumerate(entries):
            if _sha256(entry.payload) != entry.payload_hash:
                errors.append(f"Entry {i}: payload hash mismatch")
            prev = entries[i-1].chain_hash if i > 0 else ""
            if _chain_hash(prev, entry.payload_hash) != entry.chain_hash:
                errors.append(f"Entry {i}: chain broken")
        return len(errors) == 0, errors


# ─── CRUD helpers ─────────────────────────────────────────────────────────────


def save_signal(symbol: str, direction: str, confidence: float,
               agent: str, model_used: str, market_data: dict) -> TradeSignal:
    session = get_session()
    sig = TradeSignal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        agent=agent,
        model_used=model_used,
        market_data_snapshot=json.dumps(market_data),
    )
    session.add(sig)
    session.commit()
    session.refresh(sig)
    AuditLog.append("signal", {
        "signal_id": sig.id,
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "agent": agent,
    })
    return sig


def save_execution(symbol: str, direction: str, quantity: float,
                   price: float, signal_id: int, risk_passed: bool,
                   consensus_score: float, mode: str = "paper") -> TradeExecution:
    session = get_session()
    exec_ = TradeExecution(
        symbol=symbol,
        direction=direction,
        quantity=quantity,
        price=price,
        notional=quantity * price,
        signal_id=signal_id,
        risk_check_passed=risk_passed,
        consensus_score=consensus_score,
        execution_mode=mode,
    )
    session.add(exec_)
    session.commit()
    session.refresh(exec_)
    AuditLog.append("execution", {
        "execution_id": exec_.id,
        "symbol": symbol,
        "direction": direction,
        "notional": exec_.notional,
        "risk_passed": risk_passed,
        "mode": mode,
    })
    return exec_
