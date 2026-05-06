#!/usr/bin/env python
"""STARE — Unified experiment launcher.

Convenience wrapper that runs the scenario run.py scripts, multiseed,
or analysis from a single CLI. For understanding how things work,
read the simple run.py in each experiments/ directory instead.

    python run_experiment.py --scenario turb
    python run_experiment.py --scenario turb --mode multiseed --seeds 10
    python run_experiment.py --mode analyze
"""

import argparse
import os
import subprocess
import sys

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SCENARIO_DIRS = {
    "turb": os.path.join(_PROJECT_ROOT, "experiments", "neurips2026_turb"),
    "noturb": os.path.join(_PROJECT_ROOT, "experiments", "neurips2026_noturb"),
    "synth": os.path.join(_PROJECT_ROOT, "experiments", "neurips2026_synth"),
}


def _run_single(args):
    """Run the scenario's run.py script."""
    run_py = os.path.join(SCENARIO_DIRS[args.scenario], "run.py")
    subprocess.run([sys.executable, run_py], check=True)


def _run_multiseed(args):
    """Run multi-seed via evaluation.multiseed."""
    from evaluation.multiseed import run_multiseed

    config_path = os.path.join(SCENARIO_DIRS[args.scenario], "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    agents = ["a2c", "ppo", "ddpg", "td3"] if args.agents == "all" else args.agents.split(",")
    run_multiseed(
        config, agents,
        seeds=args.seeds,
        timesteps=args.timesteps,
        output_dir=os.path.join(args.output_dir, args.scenario),
        base_dir=SCENARIO_DIRS[args.scenario],
        label=args.scenario,
        seed_start=args.seed_start,
    )


def _run_analyze(args):
    """Run analysis on multi-seed results."""
    from evaluation.analyze_multiseed import main as analyze_main
    analyze_main()


def main():
    p = argparse.ArgumentParser(description="STARE — Unified experiment launcher.")
    p.add_argument("--scenario", choices=["turb", "noturb", "synth"], default="turb")
    p.add_argument("--mode", choices=["single", "multiseed", "analyze"], default="single")
    p.add_argument("--agents", default="all")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "results", "multi_seed"))
    args = p.parse_args()

    {"single": _run_single, "multiseed": _run_multiseed, "analyze": _run_analyze}[args.mode](args)


if __name__ == "__main__":
    main()
