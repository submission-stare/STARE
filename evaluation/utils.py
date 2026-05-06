import os
import yaml
from typing import Dict, Optional

from data.preprocessors.features import normalize_technical_indicator_selection

SUPPORTED_RL_MODES = {"portfolio", "trading"}
DEFAULT_INITIAL_CAPITAL = 1.0
RISK_FREE_RATE_ANNUAL = 0.03

def _resolve_rl_mode(config: Dict) -> str:
    mode = str(config.get("rl_mode", "portfolio")).strip().lower()
    if mode not in SUPPORTED_RL_MODES:
        raise ValueError(f"Unsupported rl_mode '{mode}'. Supported modes: {sorted(SUPPORTED_RL_MODES)}")
    return mode


def _resolve_technical_indicator_columns(config: Dict) -> list[str]:
    return normalize_technical_indicator_selection(config.get("technical_indicator_columns"))

def resolve_env_class(config: Dict):
    from envs.gym_wrappers.portfolio_env import PortfolioEnv
    from envs.gym_wrappers.trading_env import TradingEnv
    
    rl_mode = _resolve_rl_mode(config)
    return PortfolioEnv if rl_mode == "portfolio" else TradingEnv

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _resolve_save_dir(save_models_dir: Optional[str], base_output_dir: str) -> Optional[str]:
    if not save_models_dir:
        return None
    save_dir = save_models_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(base_output_dir, save_dir)
    return _ensure_dir(save_dir)

