"""
Phase 1c — Redis Setup (TTL-bucketed cache)
"""
import json
import hashlib
from typing import Optional, Any

import redis


# ---------------------------------------------------------------------------
# TTL Buckets
# ---------------------------------------------------------------------------
TTL_NEWS_SENTIMENT = 3600    # news headlines: immutable once published
TTL_MARKET_ANALYSIS = 300    # market condition analysis
TTL_PATTERN_SIGNAL = 60       # price pattern recognition
TTL_JUDGE_SYNTHESIS = 300    # Claude judge synthesis cache

import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "192.168.1.22")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))


# ---------------------------------------------------------------------------
# Connection pool (singleton)
# ---------------------------------------------------------------------------
_pool: Optional[redis.ConnectionPool] = None


def get_redis_pool() -> redis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10,
        )
    return _pool


def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=get_redis_pool())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_get(key: str) -> Optional[Any]:
    """Return decoded JSON value or None if missing/expired."""
    try:
        r = get_redis()
        val = r.get(key)
        if val is None:
            return None
        return json.loads(val)
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl: int) -> bool:
    """Store value as JSON with TTL in seconds."""
    try:
        r = get_redis()
        return r.setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        return False


def cache_invalidate(pattern: str) -> int:
    """Delete all keys matching glob pattern. Returns count deleted."""
    try:
        r = get_redis()
        keys = r.keys(pattern)
        if keys:
            return r.delete(*keys)
        return 0
    except Exception:
        return 0


def make_judge_synthesis_key(signals: list) -> str:
    """
    Deterministic cache key for judge synthesis:
    SHA-256 of sorted JSON-serialized signal list.
    """
    serialized = json.dumps([s if isinstance(s, dict) else s for s in signals], sort_keys=True, default=str)
    return f"judge:synthesis:{hashlib.sha256(serialized.encode()).hexdigest()[:32]}"


def make_pattern_cache_key(symbol: str, timeframe: str) -> str:
    return f"pattern:{symbol}:{timeframe}"


def make_market_cache_key(symbol: str, kind: str = "price") -> str:
    return f"market:{kind}:{symbol}"


def make_news_cache_key(headline: str) -> str:
    """Hash of headline for news sentiment cache."""
    return f"news:sentiment:{hashlib.sha256(headline.encode()).hexdigest()[:32]}"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def ping() -> bool:
    try:
        return get_redis().ping()
    except Exception:
        return False
