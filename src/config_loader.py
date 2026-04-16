"""
Phase 1d — Config Loader
Loads YAML configs from config/ subdirectories and .env via python-dotenv.
Supports hot-reload via importlib.reload.
"""
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


CONFIG_BASE = Path(os.environ.get("CONFIG_BASE", "/opt/consensus-trade/config"))
ENV_FILE = Path(os.environ.get("ENV_FILE", "/opt/consensus-trade/.env"))

# Load .env at import time
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ---------------------------------------------------------------------------
# Raw YAML / JSON loaders
# ---------------------------------------------------------------------------

def load_yaml(rel_path: str) -> dict:
    path = CONFIG_BASE / rel_path
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_json(rel_path: str) -> dict:
    path = CONFIG_BASE / rel_path
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Backends config
# ---------------------------------------------------------------------------

_backends_cache: Optional[dict] = None


def get_backends() -> dict:
    global _backends_cache
    if _backends_cache is None:
        _backends_cache = load_json("backends.json")
    return _backends_cache


def save_backends(backends_data: dict) -> bool:
    """Write backends data to backends.json and invalidate cache."""
    global _backends_cache
    path = CONFIG_BASE / "backends.json"
    try:
        with open(path, "w") as f:
            json.dump(backends_data, f, indent=2)
        _backends_cache = backends_data
        return True
    except Exception:
        return False


def get_available_models() -> list[dict]:
    """Fetch available Ollama models via API."""
    import requests
    try:
        resp = requests.get(f"{os.environ.get('OLLAMA_BASE', 'http://localhost:11434')}/api/tags",
                             timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Assets config
# ---------------------------------------------------------------------------

_assets_cache: Optional[dict] = None


def get_assets() -> dict:
    global _assets_cache
    if _assets_cache is None:
        _assets_cache = load_yaml("assets.yaml")
    return _assets_cache


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

_strategies_cache: dict[str, dict] = {}


def load_strategy(asset_class: str, strategy_name: str) -> dict:
    key = f"{asset_class}/{strategy_name}"
    if key not in _strategies_cache:
        path = CONFIG_BASE / "strategies" / asset_class / f"{strategy_name}.yaml"
        if path.exists():
            with open(path, "r") as f:
                _strategies_cache[key] = yaml.safe_load(f) or {}
        else:
            _strategies_cache[key] = {}
    return _strategies_cache[key]


def list_strategies(asset_class: str) -> list[str]:
    strategies_dir = CONFIG_BASE / "strategies" / asset_class
    if not strategies_dir.exists():
        return []
    return [p.stem for p in strategies_dir.glob("*.yaml")]


# ---------------------------------------------------------------------------
# Consensus configs
# ---------------------------------------------------------------------------

_consensus_cache: dict[str, dict] = {}


def get_consensus_config(asset_class: str) -> dict:
    if asset_class not in _consensus_cache:
        _consensus_cache[asset_class] = load_yaml(f"consensus/{asset_class}.yaml")
    return _consensus_cache[asset_class]


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------

def reload_strategies():
    """Clear strategy cache — call after editing strategy YAMLs."""
    global _strategies_cache
    _strategies_cache.clear()


def reload_assets():
    """Clear assets cache."""
    global _assets_cache
    _assets_cache = None


def reload_backends():
    """Clear backends cache."""
    global _backends_cache
    _backends_cache = None


def reload_consensus():
    """Clear consensus config cache."""
    global _consensus_cache
    _consensus_cache.clear()


def hot_reload_all():
    reload_strategies()
    reload_assets()
    reload_backends()
    reload_consensus()


# ---------------------------------------------------------------------------
# Env helper
# ---------------------------------------------------------------------------

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    val = os.environ.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def get_env_float(key: str, default: float = 0.0) -> float:
    val = os.environ.get(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------
# Agent backend lookup
# ---------------------------------------------------------------------------

def get_backend_by_id(backend_id: str) -> Optional[dict]:
    backends = get_backends()
    for be in backends.get("backends", []):
        if be["id"] == backend_id:
            return be
    return None


def get_judge_backend() -> Optional[dict]:
    backends = get_backends()
    synth = backends.get("synthesis", {})
    return get_backend_by_id(synth.get("judgeBackendId", "claude-sonnet"))
