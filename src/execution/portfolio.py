"""
Phase 6 — execution/portfolio.py
Position tracking and P&L calculation.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Position:
    asset: str
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    starting_cash: float = 0.0

    def total_equity(self) -> float:
        pos_value = sum(p.quantity * p.current_price for p in self.positions.values())
        return self.cash + pos_value

    def total_pnl(self) -> float:
        return self.realized_pnl + sum(p.unrealized_pnl for p in self.positions.values())

    def pnl_pct(self) -> float:
        if self.starting_cash <= 0:
            return 0.0
        return self.total_pnl() / self.starting_cash

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def update_price(self, symbol: str, price: float):
        if symbol in self.positions:
            p = self.positions[symbol]
            p.current_price = price
            p.unrealized_pnl = (price - p.avg_entry_price) * p.quantity
            p.updated_at = datetime.now(timezone.utc)

    def apply_trade(
        self,
        symbol: str,
        asset: str,
        signal: str,
        quantity: float,
        price: float,
        slippage: float = 0.001,
    ) -> tuple[bool, str]:
        """
        Apply a BUY or SELL trade to the portfolio.
        Returns (success, message).
        """
        match signal:
            case "BUY":
                cost = price * quantity * (1 + slippage)
                if self.cash < cost:
                    return False, f"insufficient_cash need={cost:.2f} have={self.cash:.2f}"
                self.cash -= cost
                if symbol in self.positions:
                    p = self.positions[symbol]
                    new_qty = p.quantity + quantity
                    p.avg_entry_price = (p.avg_entry_price * p.quantity + price * quantity) / new_qty
                    p.quantity = new_qty
                else:
                    self.positions[symbol] = Position(
                        asset=asset,
                        symbol=symbol,
                        quantity=quantity,
                        avg_entry_price=price,
                        current_price=price,
                    )
                return True, "buy_executed"

            case "SELL":
                if symbol not in self.positions:
                    return False, "no_position"
                p = self.positions[symbol]
                if p.quantity < quantity:
                    return False, f"insufficient_position have={p.quantity:.4f} need={quantity:.4f}"
                proceeds = price * quantity * (1 - slippage)
                self.cash += proceeds
                realized = (price - p.avg_entry_price) * quantity
                p.quantity -= quantity
                self.realized_pnl += realized
                if p.quantity <= 1e-8:
                    del self.positions[symbol]
                return True, "sell_executed"

            case _:
                return True, "hold_no_op"
