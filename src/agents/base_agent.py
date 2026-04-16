"""
Phase 3 — agents/base_agent.py
Base agent class: _call_ollama(), _call_claude(), _cache_check(), _cache_store(), _fence()
"""
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

import redis
import requests

from src.redis_setup import (
    cache_get, cache_set, get_redis,
    TTL_NEWS_SENTIMENT, TTL_MARKET_ANALYSIS, TTL_PATTERN_SIGNAL,
)


log = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_TIMEOUT = 60  # seconds


@dataclass
class AgentSignal:
    agent_id: str
    role: str  # signal | risk | research | judge
    model: str
    signal: str  # BUY | SELL | HOLD
    confidence: float
    reasoning: str
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base for all Tier 1 / Tier 2 agents.
    Provides Ollama/Claude calling with timeouts and fallbacks.
    """

    def __init__(self, agent_id: str, role: str, model: str, adapter: str = "ollama"):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.adapter = adapter  # "ollama" or "claude"
        self._redis_client: Optional[redis.Redis] = None

    # ---- Ollama ----

    def _call_ollama(
        self,
        prompt: str,
        system: Optional[str] = None,
        timeout: int = OLLAMA_TIMEOUT,
    ) -> Optional[dict]:
        """
        Call Ollama API. Returns parsed JSON dict or None on error.
        Timeout enforced; falls back to HOLD on any failure.
        """
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            }
            if system:
                payload["system"] = system

            resp = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            return json.loads(raw)
        except requests.exceptions.Timeout:
            log.warning("[%s] Ollama timeout after %ds", self.agent_id, timeout)
        except Exception as e:
            log.warning("[%s] Ollama error: %s", self.agent_id, e)
        return None

    # ---- Claude ----

    def _call_claude(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[dict]:
        """
        Call Anthropic Claude API. Handles 429 with Retry-After.
        Logs token usage. Returns parsed JSON dict or None on error.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.error("[%s] ANTHROPIC_API_KEY not set", self.agent_id)
            return None

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "user", "content": f"System: {system}\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
                timeout=120,
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                log.warning("[%s] Claude 429 — waiting %ds", self.agent_id, retry_after)
                time.sleep(retry_after)
                return self._call_claude(prompt, system, max_tokens)

            resp.raise_for_status()
            data = resp.json()

            # Log token usage
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            log.info(
                "[%s] Claude token usage — input:%d output:%d",
                self.agent_id, input_tokens, output_tokens,
            )
            # TODO: write ApiCostLog here via db_writer

            raw = data["content"][0]["text"].strip()
            return json.loads(raw)
        except requests.exceptions.Timeout:
            log.warning("[%s] Claude timeout", self.agent_id)
        except Exception as e:
            log.warning("[%s] Claude error: %s", self.agent_id, e)
        return None

    # ---- Cache ----

    def _cache_check(self, key: str) -> Optional[Any]:
        return cache_get(key)

    def _cache_store(self, key: str, value: Any, ttl: int):
        cache_set(key, value, ttl)

    # ---- Fence ----

    @staticmethod
    def _fence(data: Any) -> str:
        """Wrap external content in <market_data> fence tags."""
        content = str(data)
        return f"<market_data>\n{content}\n</market_data>"

    # ---- Abstract method ----

    @abstractmethod
    def generate(self, *args, **kwargs) -> AgentSignal:
        """Subclasses implement their signal generation logic."""
        ...
