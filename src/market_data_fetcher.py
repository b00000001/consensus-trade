"""
Phase 1e — Market Data Fetcher
ccxt for crypto (Binance, Coinbase), yfinance for stocks/forex, CoinGecko as crypto fallback.
All outputs wrapped in MarketDataFencedAdapter to add <market_data> fence.
"""
import textwrap
from dataclasses import dataclass
from typing import Optional, Any

import requests

import ccxt
import yfinance


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    volume: float
    timestamp: str
    raw: dict  # original exchange response (for reference)


@dataclass
class OrderBookEntry:
    price: float
    size: float


@dataclass
class OrderBook:
    symbol: str
    bids: list[OrderBookEntry]
    asks: list[OrderBookEntry]
    timestamp: str


@dataclass
class OHLCV:
    symbol: str
    timeframe: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Fenced adapter — wraps external data in prompt-injection-safe fence
# ---------------------------------------------------------------------------

@dataclass
class FencedMarketData:
    """
    Wraps external market data in <market_data> tags.
    Use this before any data enters a prompt context.
    """
    raw_data: Any

    def __str__(self) -> str:
        content = str(self.raw_data)
        return f"<market_data>\n{content}\n</market_data>"


def fence(data: Any) -> FencedMarketData:
    return FencedMarketData(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COINGECKO_ID_MAP = {
    "BTC/USDT": "bitcoin",
    "BTC/USD": "bitcoin",
    "ETH/USDT": "ethereum",
    "ETH/USD": "ethereum",
    "SOL/USDT": "solana",
    "SOL/USD": "solana",
    "DOGE/USDT": "dogecoin",
    "DOGE/USD": "dogecoin",
    "XRP/USDT": "ripple",
    "XRP/USD": "ripple",
    "ADA/USDT": "cardano",
    "ADA/USD": "cardano",
}


def _symbol_to_coingecko_id(symbol: str) -> Optional[str]:
    """Convert trading symbol to CoinGecko coin ID."""
    upper = symbol.upper()
    if upper in _COINGECKO_ID_MAP:
        return _COINGECKO_ID_MAP[upper]
    # Strip /USD or /USDT suffix and lowercase
    for sep in ("/USDT", "/USD", "/BTC"):
        if upper.endswith(sep):
            base = upper[: -len(sep)]
            return base.lower()
    return None


# ---------------------------------------------------------------------------
# MarketDataFetcher
# ---------------------------------------------------------------------------

class MarketDataFetcher:
    """
    Unified fetcher for crypto and traditional assets.
    Crypto: Binance → Coinbase → CoinGecko (free, no auth)
    Stocks/Forex: yfinance
    """

    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self._binance = ccxt.binance({"enableRateLimit": True})
        self._coinbase = ccxt.coinbase({})
        self._yf = yfinance

    # ---- Price ----

    def fetch_price(self, symbol: str) -> Optional[MarketSnapshot]:
        """
        Fetch current price. Tries Binance, then Coinbase, then CoinGecko, then yfinance.
        """
        market_type = self._detect_market_type(symbol)

        if market_type == "crypto":
            snap = self._fetch_crypto_price(symbol)
            if snap:
                return snap
            # CoinGecko fallback
            return self._fetch_coingecko_price(symbol)
        else:
            return self._fetch_yfinance_price(symbol)

    def _fetch_crypto_price(self, symbol: str) -> Optional[MarketSnapshot]:
        for exchange in (self._binance, self._coinbase):
            try:
                ticker = exchange.fetch_ticker(symbol)
                return MarketSnapshot(
                    symbol=symbol,
                    price=float(ticker["last"]),
                    volume=float(ticker.get("quoteVolume", 0)),
                    timestamp=str(ticker["timestamp"]),
                    raw=ticker,
                )
            except Exception:
                pass
        return None

    def _fetch_coingecko_price(self, symbol: str) -> Optional[MarketSnapshot]:
        """Free crypto price API — no auth needed, rate limit 10-50 req/min."""
        coin_id = _symbol_to_coingecko_id(symbol)
        if not coin_id:
            return None
        try:
            resp = requests.get(
                f"{self.COINGECKO_BASE}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd", "include_24hr_vol": "true"},
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if coin_id not in data:
                return None
            entry = data[coin_id]
            return MarketSnapshot(
                symbol=symbol,
                price=float(entry["usd"]),
                volume=float(entry.get("usd_24h_vol", 0)),
                timestamp="",
                raw=data,
            )
        except Exception:
            return None

    def _fetch_yfinance_price(self, symbol: str) -> Optional[MarketSnapshot]:
        try:
            ticker = self._yf.Ticker(symbol)
            info = ticker.fast_info
            price = info.last_price or info.previous_close
            volume = info.last_volume or 0
            return MarketSnapshot(
                symbol=symbol,
                price=float(price),
                volume=float(volume),
                timestamp=str(info.last_epoch),
                raw={},
            )
        except Exception:
            return None

    # ---- Order Book ----

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        market_type = self._detect_market_type(symbol)
        if market_type != "crypto":
            return None
        exchange = self._binance
        try:
            book = exchange.fetch_order_book(symbol, limit=limit)
            return OrderBook(
                symbol=symbol,
                bids=[OrderBookEntry(float(p), float(s)) for p, s in book.get("bids", [])],
                asks=[OrderBookEntry(float(p), float(s)) for p, s in book.get("asks", [])],
                timestamp=str(book.get("timestamp", "")),
            )
        except Exception:
            return None

    # ---- OHLCV ----

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> list[OHLCV]:
        market_type = self._detect_market_type(symbol)
        if market_type == "crypto":
            return self._fetch_ccxt_ohlcv(symbol, timeframe, limit)
        return self._fetch_yfinance_ohlcv(symbol, timeframe)

    def _fetch_ccxt_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
        for exchange in (self._binance, self._coinbase):
            try:
                raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return [
                    OHLCV(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=str(ohlcv[0]),
                        open=float(ohlcv[1]),
                        high=float(ohlcv[2]),
                        low=float(ohlcv[3]),
                        close=float(ohlcv[4]),
                        volume=float(ohlcv[5]),
                    )
                    for ohlcv in raw
                ]
            except Exception:
                pass
        return []

    def _fetch_yfinance_ohlcv(self, symbol: str, timeframe: str) -> list[OHLCV]:
        tf_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d",
        }
        yf_tf = tf_map.get(timeframe, "1d")
        try:
            ticker = self._yf.Ticker(symbol)
            df = ticker.history(period="60d", interval=yf_tf)
            return [
                OHLCV(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=str(idx.timestamp()),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
                for idx, row in df.iterrows()
            ]
        except Exception:
            return []

    # ---- Helpers ----

    @staticmethod
    def _detect_market_type(symbol: str) -> str:
        crypto_suffixes = ("/USDT", "/USD", "/BTC", "/ETH")
        if any(symbol.upper().endswith(s) for s in crypto_suffixes):
            return "crypto"
        if "/" in symbol:
            return "crypto"
        return "stock_forex"

    @staticmethod
    def symbol_to_display(symbol: str) -> str:
        return symbol.replace("/", "").upper()
