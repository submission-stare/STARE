"""Hyperparameter tuning with Optuna for DRL agents (NeurIPS 2026 STARE).

Tunes A2C, PPO, DDPG, TD3 by maximizing Sharpe Ratio on a held-out
validation period (2016-01-01 to 2018-12-31), training on 2009-2015
real Yahoo data with the 28-ticker DJIA universe inherited from
experiments/neurips2026/config.yaml.

Usage:
    python tuning/experiments/neurips2026/optuna_tune.py \\
        --agent ppo --n-trials 50 --timesteps 20000 \\
        --output-dir tuning/experiments/neurips2026/results/optuna
"""

import argparse
import csv
import json
import os
import pickle
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import A2C, DDPG, PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise

from evaluation.experiment_setup import (
    build_envs,
    set_global_seed,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_AGENTS = ("a2c", "ppo", "ddpg", "td3")

# Validation window — train on 2009-2015, validate on 2016-2018
_VAL_TRAIN_START = "2009-01-01"
_VAL_TRAIN_END = "2015-12-31"
_VAL_TEST_START = "2016-01-01"
_VAL_TEST_END = "2018-12-31"


# ---------------------------------------------------------------------------
# Search-space samplers
# ---------------------------------------------------------------------------
def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096]),
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.999]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
    }


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096]),
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.999]),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 500000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 100, 256, 512]),
        "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01]),
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.999]),
        "noise_std": trial.suggest_categorical("noise_std", [0.05, 0.1, 0.2]),
    }


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 500000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 100, 256, 512]),
        "tau": trial.suggest_categorical("tau", [0.001, 0.005, 0.01]),
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.999]),
        "policy_delay": trial.suggest_categorical("policy_delay", [1, 2, 3]),
    }


_SAMPLERS: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {
    "a2c": sample_a2c_params,
    "ppo": sample_ppo_params,
    "ddpg": sample_ddpg_params,
    "td3": sample_td3_params,
}

_AGENT_CLASSES = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG, "td3": TD3}


# ---------------------------------------------------------------------------
# Validation config
# ---------------------------------------------------------------------------
def _validation_config(agent_key: str | None = None) -> dict:
    """Load base config and adjust dates for the validation window.

    Uses dedicated cache filenames so the validation window (2016-2018)
    does NOT collide with the production caches that the multi-seed runs
    populated for the 2019-2020 test window. When ``agent_key`` is given,
    cache filenames are also per-agent so independent processes (one per
    agent) never write to the same CSV concurrently.
    """
    import yaml
    _turb_config = os.path.join(
        _PROJECT_ROOT, "experiments", "neurips2026_turb", "config.yaml")
    with open(_turb_config) as f:
        cfg = yaml.safe_load(f)
    cfg["train_start"] = _VAL_TRAIN_START
    cfg["train_end"] = _VAL_TRAIN_END
    cfg["test_start"] = _VAL_TEST_START
    cfg["test_end"] = _VAL_TEST_END
    # Force start_date so the train downloader covers the validation window.
    cfg["start_date"] = _VAL_TRAIN_START
    # Per-agent caches — prevent concurrent-write corruption when several
    # processes (one per agent) launch in parallel and each sees a cache miss.
    suffix = f"_{agent_key}" if agent_key else ""
    cfg["train_data_cache_file"] = f"data_cache_train_optuna_val{suffix}.csv"
    cfg["test_data_cache_file"] = f"data_cache_test_optuna_val{suffix}.csv"
    return cfg


# ---------------------------------------------------------------------------
# Sharpe evaluation
# ---------------------------------------------------------------------------
def _evaluate_sharpe(model, test_env, risk_free_rate: float = 0.0) -> float:
    """Run a deterministic backtest and return annualized Sharpe."""
    import pandas as pd

    reset_res = test_env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
    vals = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step = test_env.step(action)
        if len(step) == 5:
            obs, _, term, trunc, _ = step
            done = term or trunc
        else:
            obs, _, done, _ = step
        vals.append(test_env.portfolio_value)
    if len(vals) < 2:
        return float("nan")
    returns = pd.Series(vals).pct_change().dropna()
    if returns.std() < 1e-12:
        return 0.0
    sr_daily = (returns.mean() - risk_free_rate / 252) / (returns.std() + 1e-8)
    return float(sr_daily * np.sqrt(252))


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------
def _build_objective(agent_key: str, timesteps: int, base_seed: int = 0):
    sampler = _SAMPLERS[agent_key]
    model_cls = _AGENT_CLASSES[agent_key]

    def objective(trial: optuna.Trial) -> float:
        set_global_seed(base_seed + trial.number)
        params = sampler(trial)
        cfg = deepcopy(_validation_config(agent_key=agent_key))
        train_env, test_env = build_envs(cfg)

        # Build SB3 model with sampled params
        kwargs: Dict[str, Any] = {"policy": "MlpPolicy", "env": train_env, "verbose": 0}

        # Off-policy noise (DDPG only)
        if agent_key == "ddpg":
            noise_std = params.pop("noise_std")
            n_assets = train_env.action_space.shape[0]
            kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_assets), sigma=noise_std * np.ones(n_assets)
            )

        # Device selection (CUDA for off-policy when available, CPU otherwise).
        import torch
        if agent_key in ("ddpg", "td3") and torch.cuda.is_available():
            kwargs["device"] = "cuda"
        else:
            kwargs["device"] = "cpu"

        kwargs.update(params)

        try:
            model = model_cls(**kwargs)
            model.learn(total_timesteps=timesteps)
        except Exception as e:
            print(f"  Trial {trial.number}: training failed: {e}")
            raise optuna.TrialPruned()

        if hasattr(model, "set_env"):
            try:
                model.set_env(test_env)
            except Exception:
                model.env = test_env

        sr = _evaluate_sharpe(model, test_env, risk_free_rate=cfg.get("risk_free_rate", 0.0))
        if not np.isfinite(sr):
            raise optuna.TrialPruned()
        return sr

    return objective


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_study_results(study: optuna.Study, output_dir: str, agent_key: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Best params JSON
    best = {
        "agent": agent_key,
        "best_value": study.best_value if study.best_trial is not None else None,
        "best_params": study.best_params if study.best_trial is not None else {},
        "best_trial_number": study.best_trial.number if study.best_trial is not None else None,
        "n_trials": len(study.trials),
        "study_name": study.study_name,
    }
    with open(os.path.join(output_dir, f"{agent_key}_best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    # Study pickle
    with open(os.path.join(output_dir, f"{agent_key}_study.pkl"), "wb") as f:
        pickle.dump(study, f)

    # All trials CSV
    csv_path = os.path.join(output_dir, f"{agent_key}_all_trials.csv")
    fieldnames = ["number", "value", "state", "datetime_start", "datetime_complete", "duration_s"]
    param_keys: List[str] = []
    for t in study.trials:
        for k in t.params.keys():
            if k not in param_keys:
                param_keys.append(k)
    fieldnames += [f"param_{k}" for k in param_keys]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in study.trials:
            duration = (t.datetime_complete - t.datetime_start).total_seconds() \
                if t.datetime_complete and t.datetime_start else None
            row = {
                "number": t.number,
                "value": t.value,
                "state": str(t.state.name),
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else "",
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else "",
                "duration_s": duration,
            }
            for k in param_keys:
                row[f"param_{k}"] = t.params.get(k)
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for DRL agents")
    p.add_argument("--agent", required=True, choices=list(SUPPORTED_AGENTS) + ["all"])
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=20_000,
                   help="Training timesteps per trial (default: 20000)")
    p.add_argument("--output-dir", default=os.path.join(
        _PROJECT_ROOT, "tuning", "experiments", "neurips2026", "results", "optuna"))
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for reproducibility")
    p.add_argument("--n-startup-trials", type=int, default=10,
                   help="TPE startup trials before pruning kicks in")
    return p.parse_args(argv)


def run_study(agent_key: str, n_trials: int, timesteps: int,
              output_dir: str, seed: int, n_startup_trials: int) -> optuna.Study:
    sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    pruner = MedianPruner(n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{agent_key}_sharpe_tuning",
        sampler=sampler,
        pruner=pruner,
    )
    objective = _build_objective(agent_key, timesteps=timesteps, base_seed=seed)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    if study.best_trial is not None:
        print(f"\n[{agent_key}] {n_trials} trials in {elapsed:.1f}s. "
              f"Best SR = {study.best_value:.4f}  Best params = {study.best_params}")
    save_study_results(study, output_dir, agent_key)
    return study


def main(argv=None) -> None:
    args = parse_args(argv)
    agents = list(SUPPORTED_AGENTS) if args.agent == "all" else [args.agent]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Optuna tuning ===")
    print(f"Agents: {agents} | Trials: {args.n_trials} | Timesteps: {args.timesteps}")
    print(f"Validation window: {_VAL_TRAIN_START} to {_VAL_TRAIN_END} (train), "
          f"{_VAL_TEST_START} to {_VAL_TEST_END} (val)")
    print(f"Output: {args.output_dir}\n")

    for agent_key in agents:
        run_study(
            agent_key=agent_key,
            n_trials=args.n_trials,
            timesteps=args.timesteps,
            output_dir=args.output_dir,
            seed=args.seed,
            n_startup_trials=args.n_startup_trials,
        )


if __name__ == "__main__":
    main()
