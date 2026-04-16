"""
adapters/minimax_adapter.py
MiniMax API adapter — OpenAI-compatible completions endpoint.
"""
import time
import os
from typing import Optional

import requests

from .base_adapter import BaseAdapter, AdapterResponse


MINIMAX_BASE = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")


class MiniMaxAdapter(BaseAdapter):
    """
    Signal agent backend using MiniMax API.
    Used as fallback when Ollama is slow/unavailable or as configured secondary.
    """

    def __init__(
        self,
        agent_id: str = "signal-minimax",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(agent_id, model or MINIMAX_MODEL)
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.timeout = timeout
        self.base_url = MINIMAX_BASE.rstrip("/")

    def model_name(self) -> str:
        return self.model

    def health_check(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False

    def ping(self) -> Optional[float]:
        start = time.monotonic()
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            return (time.monotonic() - start) * 1000
        except Exception:
            return None

    def call(self, prompt: str, **kwargs) -> AdapterResponse:
        start = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.3),
                },
                timeout=kwargs.get("timeout", self.timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse(text=text, latency_ms=latency, success=True)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse(text="", latency_ms=latency, success=False, error=str(e))