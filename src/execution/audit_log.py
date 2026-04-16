"""
Phase 6 — execution/audit_log.py
Hash-chain integrity functions for immutable audit log.
"""
import hashlib
import json
from typing import Optional


def compute_record_hash(prev_hash: str, payload: dict) -> str:
    """
    SHA-256 hash of previous hash concatenated with sorted JSON payload.
    """
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    raw = prev_hash + payload_json
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_chain(records: list[dict]) -> tuple[bool, Optional[str]]:
    """
    Verify hash chain integrity given a list of audit records.
    Each record must have 'prev_hash' and 'payload' (JSON string).
    Returns (is_valid, first_error_message).
    """
    prev_hash = ""
    for i, rec in enumerate(records):
        computed = compute_record_hash(prev_hash, json.loads(rec.get("payload", "{}")))
        if computed != rec.get("record_hash", ""):
            return False, f"Record {i} hash mismatch: expected {computed}, got {rec.get('record_hash')}"
        prev_hash = rec.get("record_hash", "")
    return True, None


def make_audit_payload(
    action: str,
    asset: str,
    symbol: str,
    signal: str,
    quantity: float,
    price: float,
    status: str,
    agent_id: Optional[str] = None,
    confidence: Optional[float] = None,
    rationale: Optional[str] = None,
    trade_id: Optional[int] = None,
) -> dict:
    """Build a canonical audit payload dict."""
    return {
        "action": action,
        "asset": asset,
        "symbol": symbol,
        "signal": signal,
        "quantity": quantity,
        "price": price,
        "status": status,
        "agent_id": agent_id,
        "confidence": confidence,
        "rationale": rationale,
        "trade_id": trade_id,
    }
