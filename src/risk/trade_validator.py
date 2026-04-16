"""
Phase 2 — risk/trade_validator.py
Tier 0: BUY/SELL/HOLD validation, cash/position sufficiency.
No LLM calls.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    ok: bool
    reason: str


def validate_signal(signal: str) -> ValidationResult:
    """Ensure signal is one of the three valid values."""
    valid = {"BUY", "SELL", "HOLD"}
    s = signal.strip().upper()
    if s in valid:
        return ValidationResult(True, "ok")
    return ValidationResult(False, f"invalid_signal:{signal}")


def validate_buy(
    cash_available: float,
    price: float,
    quantity: float,
    slippage: float = 0.001,
) -> ValidationResult:
    """Check cash is sufficient for a BUY order including slippage."""
    total_cost = price * quantity * (1 + slippage)
    if cash_available < total_cost:
        return ValidationResult(False, f"insufficient_cash need={total_cost:.2f} have={cash_available:.2f}")
    return ValidationResult(True, "ok")


def validate_sell(
    position_quantity: float,
    quantity: float,
) -> ValidationResult:
    """Check position is sufficient for a SELL order."""
    if position_quantity < quantity:
        return ValidationResult(False, f"insufficient_position have={position_quantity:.4f} need={quantity:.4f}")
    return ValidationResult(True, "ok")


def validate_full_trade(
    signal: str,
    cash_available: float,
    position_quantity: float,
    price: float,
    quantity: float,
    slippage: float = 0.001,
) -> ValidationResult:
    """
    Run all validation checks for a proposed trade.
    """
    sig_result = validate_signal(signal)
    if not sig_result.ok:
        return sig_result

    s = signal.strip().upper()
    if s == "BUY":
        return validate_buy(cash_available, price, quantity, slippage)
    elif s == "SELL":
        return validate_sell(position_quantity, quantity)
    else:
        # HOLD — always valid
        return ValidationResult(True, "ok")
