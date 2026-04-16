"""
Phase 3 — agents/redis_cache.py
Wraps Redis with TTL methods mirroring redis_setup.
Used by strategy agents.
"""
from typing import Optional, Any

from src.redis_setup import (
    cache_get, cache_set, cache_invalidate,
    TTL_NEWS_SENTIMENT, TTL_MARKET_ANALYSIS,
    TTL_PATTERN_SIGNAL, TTL_JUDGE_SYNTHESIS,
)


class RedisCache:
    """
    High-level cache wrapper for agent use.
    Provides typed TTL buckets and convenience methods.
    """

    @staticmethod
    def get_news_sentiment(key: str) -> Optional[Any]:
        return cache_get(f"news:sentiment:{key}")

    @staticmethod
    def set_news_sentiment(key: str, value: Any):
        cache_set(f"news:sentiment:{key}", value, TTL_NEWS_SENTIMENT)

    @staticmethod
    def get_market_analysis(symbol: str) -> Optional[Any]:
        return cache_get(f"market:analysis:{symbol}")

    @staticmethod
    def set_market_analysis(symbol: str, value: Any):
        cache_set(f"market:analysis:{symbol}", value, TTL_MARKET_ANALYSIS)

    @staticmethod
    def get_pattern(key: str) -> Optional[Any]:
        return cache_get(f"pattern:{key}")

    @staticmethod
    def set_pattern(key: str, value: Any):
        cache_set(f"pattern:{key}", value, TTL_PATTERN_SIGNAL)

    @staticmethod
    def get_judge_synthesis(key: str) -> Optional[Any]:
        return cache_get(f"judge:synthesis:{key}")

    @staticmethod
    def set_judge_synthesis(key: str, value: Any):
        cache_set(f"judge:synthesis:{key}", value, TTL_JUDGE_SYNTHESIS)

    @staticmethod
    def invalidate_patterns(symbol: str):
        cache_invalidate(f"pattern:{symbol}:*")
