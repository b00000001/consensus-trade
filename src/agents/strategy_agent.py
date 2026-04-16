"""
Phase 3 — agents/strategy_agent.py
StrategyAgent — loads strategy YAML, computes indicators (RSI, MACD, BB),
calls Ollama, returns AgentSignal.
"""
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.redis_cache import RedisCache
from src.market_data_fetcher import MarketDataFetcher, fence


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical indicators (pure Python)
# ---------------------------------------------------------------------------

def compute_rsi(closes: list[float], period: int = 14) -> float:
    """Compute RSI over a list of close prices."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """Compute MACD line, signal line, and histogram."""
    def ema(data: list[float], n: int) -> float:
        if len(data) < n:
            return data[-1] if data else 0.0
        k = 2 / (n + 1)
        ema_val = sum(data[:n]) / n
        for price in data[n:]:
            ema_val = price * k + ema_val * (1 - k)
        return ema_val

    if len(closes) < slow:
        return 0.0, 0.0, 0.0

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow

    # Signal line: 9-day EMA of MACD (simplified — use MACD as proxy)
    macd_hist_series = [macd_line]  # would need historical MACD for real signal line
    signal_line = macd_line * 0.9  # placeholder
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    closes: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float, float, float]:
    """Returns upper_band, middle_band, lower_band."""
    if len(closes) < period:
        last = closes[-1] if closes else 0.0
        return last, last, last

    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((p - mean) ** 2 for p in window) / period
    std = math.sqrt(variance)
    return mean + num_std * std, mean, mean - num_std * std


# ---------------------------------------------------------------------------
# Indicator registry
# ---------------------------------------------------------------------------

INDICATOR_FUNCS = {
    "RSI": lambda closes, **kw: compute_rsi(closes, kw.get("period", 14)),
    "MACD": lambda closes, **kw: compute_macd(closes,
        kw.get("fast", 12), kw.get("slow", 26), kw.get("signal", 9)),
    "BB": lambda closes, **kw: compute_bollinger_bands(closes, kw.get("period", 20), kw.get("num_std", 2.0)),
}


# ---------------------------------------------------------------------------
# StrategyAgent
# ---------------------------------------------------------------------------

class StrategyAgent(BaseAgent):
    """
    Tier 1 signal agent. Loads strategy config, fetches market data,
    computes indicators, calls local Ollama model, returns AgentSignal.
    """

    def __init__(
        self,
        agent_id: str,
        model: str,
        strategy_config: dict,
        market_fetcher: Optional[MarketDataFetcher] = None,
    ):
        super().__init__(agent_id, role="signal", model=model, adapter="ollama")
        self.strategy_config = strategy_config
        self.market_fetcher = market_fetcher or MarketDataFetcher()

    def generate(
        self,
        symbol: str,
        timeframe: str = "1h",
        sentiment: Optional[str] = None,
    ) -> AgentSignal:
        """
        Full signal generation pipeline:
        1. Fetch OHLCV data
        2. Compute configured indicators
        3. Build fenced prompt
        4. Call Ollama
        5. Return AgentSignal
        """
        indicators_config = self.strategy_config.get("indicators", [])
        template = self.strategy_config.get("prompt_template", "")

        # Fetch market data
        ohlcv_list = self.market_fetcher.fetch_ohlcv(symbol, timeframe)
        if not ohlcv_list:
            return self._fallback("market_data_unavailable")

        closes = [float(o.close) for o in ohlcv_list]
        current_price = closes[-1]

        # Compute indicators
        indicator_values = {}
        for ic in indicators_config:
            itype = ic.get("type", "").upper()
            params = ic.get("fast", ic.get("period", 14))
            if itype in INDICATOR_FUNCS:
                result = INDICATOR_FUNCS[itype](closes,
                    **{k: v for k, v in ic.items() if k != "type"})
                indicator_values[itype] = result

        # Build fenced market data block
        indicator_str = json.dumps(indicator_values, indent=2)
        sentiment_str = sentiment or "neutral"

        # Extract RSI period for template substitution
        rsi_period = 14
        for ic in indicators_config:
            if ic.get("type", "").upper() == "RSI":
                rsi_period = ic.get("period", 14)
                break

        # Substitute template vars
        market_block = (
            f"Symbol: {symbol}\n"
            f"Current price: {current_price}\n"
            f"Indicators: {indicator_str}\n"
            f"Recent news sentiment: {sentiment_str}\n"
        )

        prompt = template.format(
            symbol=symbol,
            price=current_price,
            period=rsi_period,
            rsi=indicator_values.get("RSI", 50.0),
            macd=indicator_values.get("MACD", (0, 0, 0))[0],
            signal_line=indicator_values.get("MACD", (0, 0, 0))[1],
            histogram=indicator_values.get("MACD", (0, 0, 0))[2],
            bb_upper=indicator_values.get("BB", (0, 0, 0))[0],
            bb_middle=indicator_values.get("BB", (0, 0, 0))[1],
            bb_lower=indicator_values.get("BB", (0, 0, 0))[2],
            sentiment=sentiment_str,
        )

        # Wrap in fence
        fenced_prompt = self._fence(market_block) + "\n\n" + prompt

        # Check cache
        cache_key = f"signal:{self.agent_id}:{symbol}:{timeframe}"
        cached = self._cache_check(cache_key)
        if cached:
            return AgentSignal(**{**cached, "agent_id": self.agent_id})

        # Call Ollama with timeout fallback
        system = "You are a quantitative trading signal agent. Output ONLY valid JSON."
        result = self._call_ollama(prompt=fenced_prompt, system=system, timeout=60)

        if result is None:
            return self._fallback("ollama_error")

        # Validate response shape
        signal = result.get("signal", "HOLD").strip().upper()
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"
        confidence = float(result.get("confidence", 0.5))
        reasoning = str(result.get("reasoning", ""))

        # Cache result
        self._cache_store(cache_key, {
            "role": self.role,
            "model": self.model,
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "metadata": {"symbol": symbol, "timeframe": timeframe},
        }, ttl=60)

        return AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            model=self.model,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"symbol": symbol, "timeframe": timeframe},
        )

    def _fallback(self, reason: str) -> AgentSignal:
        log.warning("[%s] Falling back to HOLD: %s", self.agent_id, reason)
        return AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            model=self.model,
            signal="HOLD",
            confidence=0.0,
            reasoning=f"fallback:{reason}",
            metadata={},
        )
