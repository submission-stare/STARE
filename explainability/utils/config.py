"""
Centralised configuration loader for Alpha-Inspector.

Reads ``config/settings.yaml`` once per process (singleton), and exposes a
simple ``get(dotted.key, default)`` interface.  Environment variables always
override YAML values when an explicit mapping is defined.

Usage::

    from utils.config import cfg

    cap  = cfg("forum.max_hypotheses", 20)
    thr  = cfg("forum.dedup_threshold", 0.70)
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Resolve the YAML path relative to the project root ──────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"

# ── Env-var overrides: yaml dotted key → (ENV_VAR, cast_fn) ──────
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "data_scope.run_dir":       ("DATA_SCOPE_RUN_DIR", str),
    "data_scope.trading_regime_dir": ("DATA_SCOPE_TRADING_REGIME_DIR", str),
    "data_scope.aggregated_dir": ("DATA_SCOPE_AGGREGATED_DIR", str),
    "forum.dedup_threshold": ("FORUM_DEDUP_THRESHOLD", float),
    "forum.panelists":       ("FORUM_PANELISTS", None),     # special handling
    "code_forum.panelists":  ("CODE_FORUM_PANELISTS", None),
    "llm.default_provider":  ("LLM_PROVIDER", str),
    "llm.max_tokens_light":  ("LLM_MAX_TOKENS_LIGHT", int),
    "llm.max_tokens_heavy":  ("LLM_MAX_TOKENS_HEAVY", int),
}


@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    """Load and cache the YAML config file."""
    if not _CONFIG_PATH.exists():
        logger.warning(f"Config file not found at {_CONFIG_PATH}. Using defaults.")
        return {}
    with open(_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    logger.info(f"Config loaded from {_CONFIG_PATH}")
    return data


def _resolve(key: str, default: Any = None) -> Any:
    """
    Lookup *key* (dotted notation) in the YAML dict, then check env overrides.

    Env vars always win when set.
    """
    data = _load_yaml()

    # ── Walk the dotted path in the YAML dict ───────────────────
    parts = key.split(".")
    node: Any = data
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        else:
            node = None
            break

    yaml_val = node if node is not None else default

    # ── Check env override ──────────────────────────────────────
    override = _ENV_OVERRIDES.get(key)
    if override:
        env_name, cast_fn = override
        env_raw = os.getenv(env_name)
        if env_raw is not None:
            if cast_fn is None:
                # Special: FORUM_PANELISTS → returns list from comma-separated string
                return [s.strip() for s in env_raw.split(",") if s.strip()]
            try:
                return cast_fn(env_raw)
            except (ValueError, TypeError):
                logger.warning(f"Env var {env_name}='{env_raw}' could not be cast to {cast_fn}. Using YAML value.")

    return yaml_val


# ── Public API ──────────────────────────────────────────────────

def cfg(key: str, default: Any = None) -> Any:
    """
    Get a config value by dotted key.

    Examples::

        cfg("forum.max_hypotheses")       # → 20
        cfg("investigator.max_calls")     # → 3
        cfg("forum.panelists")            # → ["ollama:qwen2.5:14b", ...]
        cfg("graph.recursion_limit", 50)  # → 50
    """
    return _resolve(key, default)


def get_config() -> dict:
    """Return the full (raw) YAML config dict — useful for debugging."""
    return _load_yaml()
