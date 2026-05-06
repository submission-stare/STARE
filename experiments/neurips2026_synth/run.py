"""NeurIPS 2026 — Synthetic data scenario.

Train on real Yahoo Finance data, evaluate on causal synthetic data.
Zero look-ahead bias — the test set is entirely generated.

Usage:
    python experiments/neurips2026_synth/run.py
"""

import os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))

from evaluation.experiment_setup import build_envs, compute_benchmark_series, resolve_agent
from evaluation.runner import evaluate_agent
from evaluation.reporting import save_experiment_results

from evaluation.experiment_setup import set_global_seed

with open(os.path.join(HERE, "config.yaml")) as f:
    config = yaml.safe_load(f)

set_global_seed(config.get("train_seed", 0))
train_env, test_env = build_envs(config, base_dir=HERE)
n_assets = train_env.action_space.shape[0]

results = {}
for agent_key in config["agents"]:
    cls, kwargs, timesteps = resolve_agent(agent_key, config, n_assets)
    results[agent_key.upper()] = evaluate_agent(
        cls, agent_key.upper(), train_env, test_env, timesteps,
        model_kwargs=kwargs,
        trials=config.get("trials", 1),
    )

benchmark = compute_benchmark_series(config, base_dir=HERE)
save_experiment_results(
    results,
    os.path.join(HERE, "results.txt"),
    "NeurIPS 2026 — Synthetic",
    benchmark_series=benchmark,
)
