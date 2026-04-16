"""
adapters/__init__.py
Consensus Trade — backend adapters.
"""
from .base_adapter import BaseAdapter, AdapterResponse
from .minimax_adapter import MiniMaxAdapter
from .claude_adapter import ClaudeAdapter
from .gemini_adapter import GeminiAdapter

__all__ = [
    "BaseAdapter",
    "AdapterResponse",
    "MiniMaxAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
]