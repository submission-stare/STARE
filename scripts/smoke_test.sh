#!/usr/bin/env bash
# ============================================================
# Quick smoke test — run a short training to verify setup.
# Uses only 5,000 timesteps (< 2 minutes on CPU).
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── Smoke test: turb scenario, PPO only, 5k steps ──"
python run_experiment.py \
    --scenario turb \
    --mode single \
    --agents ppo \
    --timesteps 5000

echo ""
echo "── Smoke test passed! ──"
