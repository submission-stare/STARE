#!/usr/bin/env python
"""Generate a new experiment directory with template config and runner.

Usage:
    python scaffold_experiment.py my_experiment
    python scaffold_experiment.py experiments/icml2027

Creates:
    <name>/
        config.yaml   — YAML template with all knobs documented
        run.py        — Minimal runner that loads config and calls evaluate_agent
"""

import argparse
import os
import textwrap


CONFIG_TEMPLATE = textwrap.dedent("""\
    # ── Experiment config ──
    # Copy and adjust parameters for your setup.

    # --- Agents to train ---
    agents:
      - a2c
      - ppo
      - ddpg
      - td3

    # --- DRL hyperparameters (tune with Optuna or set manually) ---
    a2c_params:
      learning_rate: 0.0007
      n_steps: 5
      gamma: 0.99

    ppo_params:
      learning_rate: 0.0003
      n_steps: 2048
      gamma: 0.99
      clip_range: 0.2
      batch_size: 64

    ddpg_params:
      learning_rate: 0.001
      buffer_size: 50000
      batch_size: 64
      tau: 0.005
      gamma: 0.99
      noise_std: 0.1

    td3_params:
      learning_rate: 0.001
      buffer_size: 50000
      batch_size: 64
      tau: 0.005
      gamma: 0.99
      policy_delay: 2

    # --- Training ---
    timesteps_per_model: 100000
    train_seed: 0

    # --- Data ---
    data_source: "yahoo"            # or "synthetic_csv"
    ticker_preset: "djia_2019"      # or explicit: tickers: [AAPL, MSFT, ...]
    allow_missing_tickers: true

    # --- Date ranges ---
    start_date: '2008-01-01'
    train_start: '2009-01-01'
    train_end: '2015-12-31'
    test_start: '2019-01-01'
    test_end: '2020-12-31'

    # --- Environment ---
    turbulence_threshold: 200.0     # set to 1e9 to disable
    technical_indicator_columns:
      - MACD
      - RSI
      - SO
      - FR

    # --- Benchmark (buy-and-hold, used for PSR) ---
    benchmark_source: "equal_weight"
    benchmark_ticker_preset: "djia_2019"
""")


RUN_TEMPLATE = textwrap.dedent('''\
    """Experiment runner — {name}.

    Usage:
        python {path}/run.py
    """

    import os, sys, yaml

    HERE = os.path.dirname(os.path.abspath(__file__))

    from evaluation.experiment_setup import build_envs, materialize_drl_kwargs, compute_benchmark_series, set_global_seed
    from evaluation.multiseed import AGENT_CLASSES
    from evaluation.runner import evaluate_agent
    from evaluation.reporting import save_experiment_results

    with open(os.path.join(HERE, "config.yaml")) as f:
        config = yaml.safe_load(f)

    set_global_seed(config.get("train_seed", 0))
    train_env, test_env = build_envs(config, base_dir=HERE)
    n_assets = train_env.action_space.shape[0]

    results = {{}}
    for agent_key in config["agents"]:
        cls = AGENT_CLASSES[agent_key]
        kwargs = materialize_drl_kwargs(agent_key, config.get(f"{{agent_key}}_params", {{}}), n_assets)
        results[agent_key.upper()] = evaluate_agent(
            cls, agent_key.upper(), train_env, test_env, config["timesteps_per_model"],
            model_kwargs=kwargs,
            trials=config.get("trials", 1),
        )

    benchmark = compute_benchmark_series(config, base_dir=HERE)
    save_experiment_results(
        results,
        os.path.join(HERE, "results.txt"),
        "{name}",
        benchmark_series=benchmark,
    )
''')


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new STARE experiment.")
    parser.add_argument("name", help="Experiment directory name (e.g. 'my_experiment' or 'experiments/icml2027')")
    args = parser.parse_args()

    target = args.name
    if not target.startswith("experiments/"):
        target = os.path.join("experiments", target)

    os.makedirs(target, exist_ok=True)

    config_path = os.path.join(target, "config.yaml")
    run_path = os.path.join(target, "run.py")

    if os.path.exists(config_path):
        print(f"  ⚠  {config_path} already exists — skipping.")
    else:
        with open(config_path, "w") as f:
            f.write(CONFIG_TEMPLATE)
        print(f"  ✓ Created {config_path}")

    if os.path.exists(run_path):
        print(f"  ⚠  {run_path} already exists — skipping.")
    else:
        with open(run_path, "w") as f:
            f.write(RUN_TEMPLATE.format(name=os.path.basename(target), path=target))
        print(f"  ✓ Created {run_path}")

    print(f"\nDone. Run your experiment with:\n  python {run_path}")


if __name__ == "__main__":
    main()
