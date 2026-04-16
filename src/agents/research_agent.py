"""
Phase 4 — agents/research_agent.py
Batches up to 10 headlines → 1 Ollama call, writes sentiment to Redis cache.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import feedparser
import requests

from agents.base_agent import BaseAgent, AgentSignal
from src.redis_setup import cache_set, TTL_NEWS_SENTIMENT
from src.market_data_fetcher import fence


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
    macro_risk: float          # 0.0 (safe) – 1.0 (dangerous)
    sentiment_score: float    # -1.0 (bearish) – 1.0 (bullish)
    summary: str
    headlines: list[str]
    metadata: dict = field(default_factory=dict)


class ResearchAgent(BaseAgent):
    """
    Tier 1 research agent.
    Fetches RSS headlines, batches up to 10 into a single Ollama call,
    writes result to Redis cache.
    """

    def __init__(self, agent_id: str = "research-1", model: str = "qwen3:32b"):
        super().__init__(agent_id, role="research", model=model, adapter="ollama")
        self._headlines_cache: list[Headline] = []

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

    def score_batch(self, headlines: list[Headline]) -> ResearchReport:
        """
        Batched sentiment: up to 10 headlines → 1 Ollama call.
        Returns ResearchReport with macro_risk and sentiment_score.
        """
        if not headlines:
            return ResearchReport(
                agent_id=self.agent_id,
                macro_risk=0.5,
                sentiment_score=0.0,
                summary="No headlines available.",
                headlines=[],
            )

        # Check cache
        cache_key = self._make_cache_key(headlines)
        cached = self._cache_check(cache_key)
        if cached:
            return ResearchReport(**{**cached, "agent_id": self.agent_id})

        # Build prompt
        headline_text = "\n".join(f"- {h.title}" for h in headlines)
        system = (
            "You are a financial news analysis agent. "
            "Assess the given headlines and respond with ONLY valid JSON: "
            '{"macro_risk": 0.0-1.0, "sentiment_score": -1.0-1.0, "summary": "..."}'
        )
        prompt = f"Analyze these crypto/finance headlines:\n{self._fence(headline_text)}\n\nProvide JSON analysis."

        result = self._call_ollama(prompt=prompt, system=system, timeout=60)

        if result is None:
            return ResearchReport(
                agent_id=self.agent_id,
                macro_risk=0.5,
                sentiment_score=0.0,
                summary="Ollama unavailable.",
                headlines=[h.title for h in headlines],
            )

        macro_risk = float(result.get("macro_risk", 0.5))
        sentiment_score = float(result.get("sentiment_score", 0.0))
        summary = str(result.get("summary", ""))

        # Cache result
        report = {
            "macro_risk": macro_risk,
            "sentiment_score": sentiment_score,
            "summary": summary,
            "headlines": [h.title for h in headlines],
            "metadata": {},
        }
        self._cache_store(cache_key, report, TTL_NEWS_SENTIMENT)

        return ResearchReport(agent_id=self.agent_id, **report)

    @staticmethod
    def _make_cache_key(headlines: list[Headline]) -> str:
        import hashlib, json
        titles = json.dumps(sorted(h.title for h in headlines), sort_keys=True)
        return f"research:sentiment:{hashlib.sha256(titles.encode()).hexdigest()[:32]}"

    def generate(self, prompt: str, **kwargs) -> "AgentSignal":
        """Required by ABC — not used directly; see run_pipeline."""
        from agents.base_agent import AgentSignal
        report = self.run_pipeline(limit=5)
        return AgentSignal(
            agent_id=self.agent_id,
            signal="HOLD",
            confidence=0.5,
            rationale=f"Research: {report.summary[:200]}",
            model_used=self.model,
        )

    def run_pipeline(self, limit: int = 10) -> ResearchReport:
        """
        Full background pipeline: fetch → score → cache → return report.
        """
        headlines = self.fetch_headlines(limit=limit)
        return self.score_batch(headlines)
