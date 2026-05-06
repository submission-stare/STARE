"""Multi-seed DRL experiment runner.

Runs DRL agents across multiple random seeds, collects per-seed metrics,
and computes aggregated statistics (mean, std, bootstrap 95% CI).

Usage from any experiment::

    from evaluation.multiseed import run_multiseed
    run_multiseed(config, agents=["ppo","td3"], seeds=10, output_dir="results/ms")

Or as a CLI (requires --config)::

    python -m evaluation.multiseed --config experiments/my_exp/config.yaml \\
        --agent all --seeds 10 --output-dir results/multi_seed
"""

import argparse
import csv
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, DDPG, PPO, TD3

from evaluation.experiment_setup import (
    build_envs,
    materialize_drl_kwargs,
    set_global_seed,
)
from evaluation.runner import evaluate_agent

# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------
AGENT_CLASSES: Dict[str, Any] = {
    "a2c": A2C,
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
}

# Metrics to collect from evaluate_agent return dict
_SCALAR_METRICS = (
    "AR", "TotalReturn", "SR", "Sortino", "PSR", "DSR",
    "CI_Low", "CI_High", "training_time_s", "evaluation_time_s",
)


# ---------------------------------------------------------------------------
# MinTRL (Bailey & López de Prado, 2012)
# ---------------------------------------------------------------------------
def compute_mintrl(
    sr: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length required for SR to be significant.

    Returns float('inf') if SR ≈ 0.
    """
    from scipy.stats import norm

    if abs(sr) < 1e-8:
        return float("inf")
    z = norm.ppf(confidence)
    excess_kurt = kurtosis  # pandas returns excess kurtosis already
    mintrl = 1.0 + (1.0 - skewness * sr + (excess_kurt / 4.0) * sr ** 2) * (z / sr) ** 2
    return max(float(mintrl), 1.0)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap percentile confidence interval."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(0)
    means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - confidence) / 2.0
    return (float(np.percentile(means, 100 * alpha)),
            float(np.percentile(means, 100 * (1 - alpha))))


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------
_AGG_KEYS = ("SR", "Sortino", "PSR", "DSR", "AR", "TotalReturn", "MaxDrawdown", "MinTRL")


def compute_aggregate_stats(records: List[Dict]) -> Dict[str, float]:
    """Compute mean, std, and bootstrap 95% CI for each metric."""
    agg: Dict[str, float] = {}
    for key in _AGG_KEYS:
        vals = [r[key] for r in records if key in r and r[key] is not None]
        if not vals:
            continue
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        lo, hi = bootstrap_ci(vals)
        agg[f"{key}_ci_lo"] = lo
        agg[f"{key}_ci_hi"] = hi
    return agg


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_single_seed(
    config: dict,
    agent_key: str,
    seed: int,
    timesteps: int = 20_000,
    base_dir: str = ".",
    label: str = "",
) -> Dict[str, Any]:
    """Train and evaluate one agent with one seed. Returns a flat metrics dict."""
    set_global_seed(seed)

    cfg = deepcopy(config)
    cfg["timesteps_per_model"] = timesteps

    model_class = AGENT_CLASSES[agent_key]
    agent_params = cfg.get(f"{agent_key}_params", {}) or {}

    train_env, test_env = build_envs(cfg, base_dir=base_dir)

    n_assets = train_env.action_space.shape[0]
    sb3_kwargs = materialize_drl_kwargs(agent_key, agent_params, n_assets)
    sb3_kwargs["seed"] = seed

    result = evaluate_agent(
        model_class,
        agent_key.upper(),
        train_env,
        test_env,
        timesteps,
        model_kwargs=sb3_kwargs,
        trials=cfg.get("trials", 1),
    )

    # Derived metrics
    account = result.get("AccountValue")
    if account is not None and len(account) > 1:
        rolling_max = account.cummax()
        drawdown = ((account - rolling_max) / rolling_max).min()
        max_dd = float(drawdown)
    else:
        max_dd = 0.0

    returns = account.pct_change().dropna() if account is not None else pd.Series(dtype=float)
    sk = float(returns.skew()) if len(returns) > 2 else 0.0
    ku = float(returns.kurtosis()) if len(returns) > 3 else 3.0
    sr_daily = result.get("SR", 0.0) / np.sqrt(252)
    mintrl = compute_mintrl(sr_daily, skewness=sk, kurtosis=ku)

    record: Dict[str, Any] = {
        "agent": agent_key,
        "label": label,
        "seed": seed,
        "timesteps": timesteps,
    }
    for k in _SCALAR_METRICS:
        record[k] = result.get(k)
    record["MaxDrawdown"] = max_dd
    record["MinTRL"] = mintrl
    record["FinalValue"] = float(account.iloc[-1]) if account is not None and len(account) > 0 else None

    return record


# ---------------------------------------------------------------------------
# CSV / JSON persistence
# ---------------------------------------------------------------------------
_CSV_COLUMNS = [
    "agent", "label", "seed", "timesteps",
    "AR", "TotalReturn", "SR", "Sortino", "PSR", "DSR",
    "CI_Low", "CI_High", "MaxDrawdown", "MinTRL", "FinalValue",
    "training_time_s", "evaluation_time_s",
]


def _append_csv(path: str, record: dict) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _save_aggregated(output_dir: str, all_records: List[dict]) -> None:
    agg_rows = []
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in all_records:
        key = (r["agent"], r.get("label", ""))
        groups.setdefault(key, []).append(r)

    for (agent, label), records in sorted(groups.items()):
        stats = compute_aggregate_stats(records)
        stats["agent"] = agent
        stats["label"] = label
        stats["n_seeds"] = len(records)
        agg_rows.append(stats)

    agg_path = os.path.join(output_dir, "aggregated_stats.json")
    with open(agg_path, "w") as f:
        json.dump(agg_rows, f, indent=2)
    print(f"\nAggregated stats saved to {agg_path}")

    agg_csv_path = os.path.join(output_dir, "aggregated_stats.csv")
    if agg_rows:
        keys = list(agg_rows[0].keys())
        with open(agg_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(agg_rows)
    print(f"Aggregated CSV saved to {agg_csv_path}")


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------
def run_multiseed(
    config: dict,
    agents: List[str],
    seeds: int = 10,
    timesteps: int = 20_000,
    output_dir: str = "results/multi_seed",
    base_dir: str = ".",
    label: str = "",
    seed_start: int = 0,
) -> List[dict]:
    """Run all agent×seed combinations and save results.

    Args:
        config: Parsed YAML config dict.
        agents: Agent keys (e.g. ["a2c", "ppo"]).
        seeds: Number of random seeds.
        timesteps: Training timesteps per run.
        output_dir: Where to write CSVs.
        base_dir: For resolving relative data paths.
        label: Optional label (e.g. scenario name).
        seed_start: First seed index (for resuming).

    Returns:
        List of per-seed result dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    seed_list = list(range(seed_start, seed_start + seeds))
    total = len(agents) * len(seed_list)
    csv_path = os.path.join(output_dir, "all_runs.csv")
    all_records: List[dict] = []

    print(f"=== Multi-seed experiment ===")
    print(f"Agents: {agents} | Seeds: {seed_list} | Label: {label or '(none)'}")
    print(f"Timesteps: {timesteps} | Total runs: {total}")
    print(f"Output: {output_dir}\n")

    t0 = time.perf_counter()
    done = 0

    for agent_key in agents:
        for seed in seed_list:
            done += 1
            print(f"\n[{done}/{total}] agent={agent_key} seed={seed}")
            try:
                record = run_single_seed(
                    config, agent_key, seed,
                    timesteps=timesteps,
                    base_dir=base_dir,
                    label=label,
                )
                all_records.append(record)
                _append_csv(csv_path, record)
                print(f"  -> SR={record['SR']:.3f}  PSR={record['PSR']:.4f}  "
                      f"MaxDD={record['MaxDrawdown']:.4f}  MinTRL={record['MinTRL']:.1f}")
            except Exception as e:
                print(f"  !! FAILED: {e}")
                all_records.append({
                    "agent": agent_key, "label": label,
                    "seed": seed, "timesteps": timesteps, "error": str(e),
                })

    elapsed = time.perf_counter() - t0
    print(f"\n=== Completed {done} runs in {elapsed:.1f}s ===")

    json_path = os.path.join(output_dir, "all_runs.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)

    valid = [r for r in all_records if "error" not in r]
    if valid:
        _save_aggregated(output_dir, valid)

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import yaml

    p = argparse.ArgumentParser(description="Multi-seed DRL experiments")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--agent", default="all", help="a2c, ppo, ddpg, td3, or 'all'")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--output-dir", default="results/multi_seed")
    p.add_argument("--label", default="", help="Label for this run (e.g. scenario name)")
    p.add_argument("--seed-start", type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    agents = list(AGENT_CLASSES.keys()) if args.agent == "all" else [args.agent]
    base_dir = os.path.dirname(os.path.abspath(args.config))

    run_multiseed(
        config, agents,
        seeds=args.seeds,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        base_dir=base_dir,
        label=args.label,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    main()
