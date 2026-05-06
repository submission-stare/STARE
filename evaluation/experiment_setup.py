"""Experiment setup utilities — data loading, env construction, benchmark.

Generic helpers that any experiment run.py can import. All functions accept
a ``config`` dict (parsed from YAML) and an optional ``base_dir`` for
resolving relative data paths.

Example::

    import yaml
    from evaluation.experiment_setup import build_envs, compute_benchmark_series

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_env, test_env = build_envs(config, base_dir=".")
    benchmark = compute_benchmark_series(config, base_dir=".")
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from stable_baselines3.common.noise import NormalActionNoise

from data.fetchers.experiment_data_loader import load_experiment_price_data
from data.preprocessors.features import calculate_technical_features
from data.ticker_presets import resolve_benchmark_tickers, resolve_tickers
from envs.gym_wrappers.trading_env import TradingEnv
from evaluation.utils import _resolve_technical_indicator_columns


# ------------------------------------------------------------------
# Seed control
# ------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------
# Internal: source-config builder
# ------------------------------------------------------------------
def _build_source_config(
    config: dict,
    source_key: str,
    synthetic_path_key: str,
    start_date_key: str,
    end_date_key: str,
    cache_key: str,
    default_cache: str,
) -> dict:
    source_cfg = dict(config)
    source_cfg["data_source"] = config.get(source_key, config.get("data_source", "yahoo"))
    synthetic_path = config.get(synthetic_path_key)
    if synthetic_path:
        source_cfg["synthetic_data_path"] = synthetic_path
    if config.get(cache_key):
        source_cfg["data_cache_file"] = config[cache_key]
    elif "data_cache_file" not in source_cfg:
        source_cfg["data_cache_file"] = default_cache
    source_cfg["start_date"] = config[start_date_key]
    source_cfg["test_end"] = config[end_date_key]
    return source_cfg


# ------------------------------------------------------------------
# Environment builder
# ------------------------------------------------------------------
def build_envs(config: dict, base_dir: str = ".") -> Tuple[TradingEnv, TradingEnv]:
    """Build train + test TradingEnv from a config dict.

    Args:
        config: Parsed YAML config dictionary.
        base_dir: Directory for resolving relative paths (data caches, synthetic CSVs).

    Returns:
        (train_env, test_env)
    """
    tickers = resolve_tickers(config)
    config["tickers"] = tickers
    allow_missing = bool(config.get("allow_missing_tickers", True))

    train_src = _build_source_config(
        config, "train_data_source", "train_synthetic_data_path",
        "start_date", "train_end", "train_data_cache_file", "data_cache_train.csv",
    )
    test_src = _build_source_config(
        config, "test_data_source", "test_synthetic_data_path",
        "test_start", "test_end", "test_data_cache_file", "data_cache_test.csv",
    )

    train_raw = load_experiment_price_data(
        config=train_src, tickers=tickers, base_dir=base_dir,
        default_cache_filename="data_cache_train.csv", allow_missing_tickers=allow_missing,
    )
    test_raw = load_experiment_price_data(
        config=test_src, tickers=tickers, base_dir=base_dir,
        default_cache_filename="data_cache_test.csv", allow_missing_tickers=allow_missing,
    )

    # Keep only tickers present in both splits
    train_avail = set(train_raw["Ticker"].astype(str).unique())
    test_avail = set(test_raw["Ticker"].astype(str).unique())
    common = [t for t in tickers if t in train_avail and t in test_avail]
    removed = [t for t in tickers if t not in common]
    if removed:
        print(f"Warning: tickers excluded (missing in one split): {', '.join(removed)}")
    if not common:
        raise ValueError("No common tickers across train/test sources.")
    tickers = common
    config["tickers"] = tickers
    train_raw = train_raw[train_raw["Ticker"].isin(tickers)]
    test_raw = test_raw[test_raw["Ticker"].isin(tickers)]

    train_df = calculate_technical_features(train_raw, tickers)
    test_df = calculate_technical_features(test_raw, tickers)
    train_df = train_df.loc[config["train_start"]:config["train_end"]]
    test_df = test_df.loc[config["test_start"]:config["test_end"]]
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty after date filtering.")

    ti_cols = _resolve_technical_indicator_columns(config)
    turb = config.get("turbulence_threshold", 1e9)

    train_env = TradingEnv(
        train_df, tickers, turbulence_threshold=turb,
        initial_capital=1_000_000.0, technical_indicator_columns=ti_cols,
    )
    test_env = TradingEnv(
        test_df, tickers, turbulence_threshold=turb,
        initial_capital=1_000_000.0, technical_indicator_columns=ti_cols,
    )
    return train_env, test_env


# ------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------
def compute_benchmark_series(config: dict, base_dir: str = ".") -> pd.Series:
    """Compute benchmark account-value series for the test period.

    Supports ``benchmark_source: equal_weight`` (default) or ``djia_index``.
    """
    initial_capital = 1_000_000.0
    benchmark_source = str(config.get("benchmark_source", "equal_weight")).strip().lower()

    if benchmark_source == "djia_index":
        benchmark_symbol = str(config.get("benchmark_index_symbol", "^DJI")).strip()
        bm_config = dict(config)
        bm_config["data_source"] = config.get("benchmark_index_data_source", "yahoo")
        bm_config["data_cache_file"] = config.get("benchmark_index_cache_file", "benchmark_index_cache.csv")
        bm_config["start_date"] = config["test_start"]
        bm_config["test_end"] = config["test_end"]
        bm_df = load_experiment_price_data(
            config=bm_config, tickers=[benchmark_symbol], base_dir=base_dir,
            default_cache_filename="benchmark_index_cache.csv", allow_missing_tickers=False,
        )
        test_start = pd.Timestamp(config["test_start"])
        test_end = pd.Timestamp(config["test_end"])
        bm_df = bm_df[(bm_df.index >= test_start) & (bm_df.index <= test_end)]
        close = (
            bm_df.loc[bm_df["Ticker"].astype(str) == benchmark_symbol, "Close"]
            .astype(float).sort_index()
        )
        close = close[~close.index.duplicated(keep="last")].dropna()
        if close.empty:
            raise ValueError(f"No benchmark prices for '{benchmark_symbol}'.")
        return pd.Series(
            (initial_capital * (close / float(close.iloc[0]))).values,
            index=pd.to_datetime(close.index), name="benchmark",
        )

    # --- equal_weight (default) ---
    benchmark_tickers = resolve_benchmark_tickers(config)

    bm_config = dict(config)
    bm_config["data_source"] = config.get(
        "benchmark_data_source",
        config.get("test_data_source", config.get("data_source", "yahoo")),
    )
    bm_synth = config.get("benchmark_synthetic_data_path") or config.get("test_synthetic_data_path")
    if bm_synth:
        bm_config["synthetic_data_path"] = bm_synth
    bm_config["data_cache_file"] = config.get("benchmark_data_cache_file", "benchmark_data_cache.csv")
    bm_config["start_date"] = config["test_start"]
    bm_config["test_end"] = config["test_end"]
    bm_df = load_experiment_price_data(
        config=bm_config, tickers=benchmark_tickers, base_dir=base_dir,
        default_cache_filename="benchmark_data_cache.csv", allow_missing_tickers=True,
    )
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])
    bm_df = bm_df[(bm_df.index >= test_start) & (bm_df.index <= test_end)]

    available = set(bm_df["Ticker"].astype(str).unique())
    benchmark_tickers = [t for t in benchmark_tickers if t in available]
    if not benchmark_tickers:
        raise ValueError("Benchmark ticker universe is empty after filtering.")

    close_table = bm_df.pivot(columns="Ticker", values="Close").reindex(columns=benchmark_tickers).ffill().bfill()
    daily_ret = close_table.pct_change().fillna(0.0).mean(axis=1)
    values = initial_capital * (1.0 + daily_ret).cumprod()
    return pd.Series(values.values, index=pd.to_datetime(close_table.index), name="benchmark")


# ------------------------------------------------------------------
# DRL kwarg materialization
# ------------------------------------------------------------------
def materialize_drl_kwargs(agent_key: str, raw_kwargs: dict, n_assets: int) -> dict:
    """Translate Optuna-style hyperparameters into SB3 constructor kwargs.

    ``noise_std`` → ``NormalActionNoise(sigma=noise_std * I)`` for DDPG/TD3.
    """
    kw = dict(raw_kwargs or {})
    if agent_key in ("ddpg", "td3") and "noise_std" in kw:
        noise_std = float(kw.pop("noise_std"))
        kw["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_assets), sigma=noise_std * np.ones(n_assets),
        )
    return kw


# ------------------------------------------------------------------
# Unified agent resolver (DRL + LLM)
# ------------------------------------------------------------------

# Lazy import to avoid hard dependency on SB3 at module level for
# code paths that only need build_envs / compute_benchmark_series.
_DRL_CLASSES = None


def _get_drl_classes():
    global _DRL_CLASSES
    if _DRL_CLASSES is None:
        from stable_baselines3 import A2C, DDPG, PPO, TD3
        _DRL_CLASSES = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG, "td3": TD3}
    return _DRL_CLASSES


def resolve_agent(agent_key: str, config: dict, n_assets: int):
    """Resolve an agent key to (model_class, model_kwargs, timesteps).

    DRL keys (a2c, ppo, ddpg, td3) are resolved from the SB3 registry.
    LLM aliases (e.g. LLM_GPT) are resolved from ``config["llm_openrouter_models"]``
    using ``build_openrouter_strategist_class``.

    Returns:
        (model_class, model_kwargs, timesteps)
    """
    drl = _get_drl_classes()
    lower_key = agent_key.lower()

    if lower_key in drl:
        cls = drl[lower_key]
        kwargs = materialize_drl_kwargs(lower_key, config.get(f"{lower_key}_params", {}), n_assets)
        train_seed = config.get("train_seed")
        if train_seed is not None:
            kwargs["seed"] = int(train_seed)
        return cls, kwargs, config["timesteps_per_model"]

    llm_models = config.get("llm_openrouter_models", {})
    if agent_key in llm_models:
        from agents.llms.llm_strategist import build_openrouter_strategist_class
        cls = build_openrouter_strategist_class(llm_models[agent_key])
        kwargs = dict(config.get("llm_agent_params", {}))
        timesteps = int(config.get("llm_timesteps", 1))
        return cls, kwargs, timesteps

    raise ValueError(
        f"Unknown agent '{agent_key}'. "
        f"Valid DRL keys: {sorted(drl.keys())}. "
        f"Valid LLM aliases: {sorted(llm_models.keys())}."
    )
