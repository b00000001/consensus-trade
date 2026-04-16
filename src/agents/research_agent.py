"""
Phase 4 — agents/research_agent.py
Research agent supporting multiple backends:
  - ollama (default): local, free, unlimited
  - gemini-cli: free tier, rate-limited to 900 req/day, gated to every N cycles
  - minimax: API-based, paid
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import feedparser
import requests

from agents.base_agent import AgentSignal
from src.redis_setup import cache_set, TTL_NEWS_SENTIMENT, cache_get
from agents.adapters import MiniMaxAdapter, GeminiAdapter, AdapterResponse


log = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
NEWS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]


@dataclass
class Headline:
    title: str
    link: str
    published: str


@dataclass
class ResearchReport:
    agent_id: str
    macro_risk: float           # 0.0 (safe) – 1.0 (dangerous)
    sentiment_score: float     # -1.0 (bearish) – 1.0 (bullish)
    summary: str
    headlines: list[str]
    adapter_used: str = "unknown"
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class ResearchAgent:
    """
    Tier 1 research agent.
    Fetches RSS headlines, batches up to 10 into a single backend call,
    writes result to Redis cache.

    Supports multiple backends: ollama (default), gemini-cli, minimax.
    """

    def __init__(
        self,
        agent_id: str = "research-local",
        model: str = "qwen3:32b",
        adapter_name: str = "ollama",  # "ollama" | "gemini-cli" | "minimax"
        cycle_gate: int = 1,  # only fire every N cycles (gemini-cli: 8 recommended)
        timeout: int = 60,
    ):
        self.agent_id = agent_id
        self.model = model
        self.adapter_name = adapter_name
        self.cycle_gate = cycle_gate
        self.timeout = timeout

        self._ollama_base = OLLAMA_BASE
        self._minimax_adapter: Optional[MiniMaxAdapter] = None
        self._gemini_adapter: Optional[GeminiAdapter] = None

    def fetch_headlines(self, limit: int = 10) -> list[Headline]:
        """Fetch latest headlines from configured RSS feeds."""
        headlines = []
        for url in NEWS_FEEDS:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:limit]:
                    headlines.append(Headline(
                        title=entry.get("title", ""),
                        link=entry.get("link", ""),
                        published=entry.get("published", ""),
                    ))
                    if len(headlines) >= limit:
                        break
            except Exception as e:
                log.warning("[%s] Failed to fetch feed %s: %s", self.agent_id, url, e)
            if len(headlines) >= limit:
                break
        return headlines[:limit]

    def _make_cache_key(self, headlines: list[Headline]) -> str:
        import hashlib, json
        titles = json.dumps(sorted(h.title for h in headlines), sort_keys=True)
        return f"research:sentiment:{hashlib.sha256(titles.encode()).hexdigest()[:32]}"

    def _call_ollama(self, prompt: str, system: str) -> Optional[dict]:
        """Direct Ollama call (existing behavior)."""
        import json as _json
        try:
            resp = requests.post(
                f"{self._ollama_base}/api/generate",
                json={"model": self.model, "prompt": f"System: {system}\n\n{prompt}",
                      "system": system, "options": {"temperature": 0.3}, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Handle doubly-quoted JSON
            if raw.startswith('"') and raw.endswith('"'):
                try:
                    raw = _json.loads(raw)
                except _json.JSONDecodeError:
                    pass
            return _json.loads(raw)
        except Exception as e:
            log.warning("[%s] Ollama call failed: %s", self.agent_id, e)
            return None

    def _call_gemini(self, prompt: str, system: str) -> Optional[dict]:
        """Gemini CLI — fires only if should_fire(cycle) is True."""
        if self._gemini_adapter is None:
            self._gemini_adapter = GeminiAdapter(agent_id=self.agent_id, model="gemini-2.0-flash", timeout=self.timeout)
        resp = self._gemini_adapter.call(prompt=f"System: {system}\n\n{prompt}")
        if not resp.success:
            log.warning("[%s] Gemini CLI call failed: %s", self.agent_id, resp.error)
            return None
        try:
            import json as _json
            text = resp.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:]) if text.endswith("```") else text
            return _json.loads(text)
        except Exception as e:
            log.warning("[%s] Gemini response parse failed: %s", self.agent_id, e)
            return None

    def _call_minimax(self, prompt: str, system: str) -> Optional[dict]:
        """MiniMax API call."""
        if self._minimax_adapter is None:
            self._minimax_adapter = MiniMaxAdapter(agent_id=self.agent_id, model="MiniMax-M2.7")
        resp = self._minimax_adapter.call(prompt=f"System: {system}\n\n{prompt}")
        if not resp.success:
            log.warning("[%s] MiniMax call failed: %s", self.agent_id, resp.error)
            return None
        try:
            import json as _json
            return _json.loads(resp.text)
        except Exception as e:
            log.warning("[%s] MiniMax parse failed: %s", self.agent_id, e)
            return None

    def score_batch(self, headlines: list[Headline]) -> ResearchReport:
        """
        Batched sentiment: up to 10 headlines → 1 backend call.
        Returns ResearchReport with macro_risk and sentiment_score.
        """
        if not headlines:
            return ResearchReport(
                agent_id=self.agent_id,
                macro_risk=0.5,
                sentiment_score=0.0,
                summary="No headlines available.",
                headlines=[],
                adapter_used=self.adapter_name,
            )

        cache_key = self._make_cache_key(headlines)
        cached = cache_get(cache_key)
        if cached:
            return ResearchReport(**{**cached, "agent_id": self.agent_id})

        headline_text = "\n".join(f"- {h.title}" for h in headlines)
        system = (
            "You are a financial news analysis agent. "
            "Assess the given headlines and respond with ONLY valid JSON: "
            '{"macro_risk": 0.0-1.0, "sentiment_score": -1.0-1.0, "summary": "..."}'
        )
        prompt = f"Analyze these crypto/finance headlines:\n<data>\n{headline_text}\n</data>\n\nRespond with JSON only."

        if self.adapter_name == "gemini-cli":
            result = self._call_gemini(prompt, system)
        elif self.adapter_name == "minimax":
            result = self._call_minimax(prompt, system)
        else:
            result = self._call_ollama(prompt, system)

        if result is None:
            return ResearchReport(
                agent_id=self.agent_id,
                macro_risk=0.5,
                sentiment_score=0.0,
                summary=f"{self.adapter_name} unavailable — using fallback.",
                headlines=[h.title for h in headlines],
                adapter_used=self.adapter_name,
            )

        macro_risk = float(result.get("macro_risk", 0.5))
        sentiment_score = float(result.get("sentiment_score", 0.0))
        summary = str(result.get("summary", ""))

        report = {
            "macro_risk": macro_risk,
            "sentiment_score": sentiment_score,
            "summary": summary,
            "headlines": [h.title for h in headlines],
            "adapter_used": self.adapter_name,
            "metadata": {},
        }
        cache_set(cache_key, report, TTL_NEWS_SENTIMENT)

        return ResearchReport(agent_id=self.agent_id, **report)

    def should_fire(self, cycle_number: int) -> bool:
        """
        For gemini-cli: only fire on every Nth cycle.
        For ollama/minimax: always fire.
        """
        if self.adapter_name == "gemini-cli":
            return cycle_number % self.cycle_gate == 0
        return True

    def run_pipeline(self, limit: int = 10) -> ResearchReport:
        """Full pipeline: fetch → score → cache → return report."""
        headlines = self.fetch_headlines(limit=limit)
        return self.score_batch(headlines)