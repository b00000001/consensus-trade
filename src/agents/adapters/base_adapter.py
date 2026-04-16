"""
adapters/base_adapter.py
Abstract base for all agent backends.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AdapterResponse:
    text: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BaseAdapter(ABC):
    """All backends must implement this interface."""

    def __init__(self, agent_id: str, model: str):
        self.agent_id = agent_id
        self.model = model

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> AdapterResponse:
        """Send a prompt and return the response."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the adapter is reachable and authed."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier used by this adapter."""
        pass