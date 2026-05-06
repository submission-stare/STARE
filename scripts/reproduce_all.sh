#!/usr/bin/env bash
# ============================================================
# STARE — Reproduce ALL paper results (3 scenarios × 4 agents)
# 
# Usage:
#   bash scripts/reproduce_all.sh            # full run
#   bash scripts/reproduce_all.sh --dry-run  # print commands only
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

run() {
    echo ">>> $*"
    $DRY_RUN || "$@"
}

echo "============================================"
echo "  STARE — Full Paper Reproduction Pipeline"
echo "============================================"
echo ""

# ── Step 1: Single-run evaluation (DRL only) ──
for scenario in turb noturb synth; do
    echo ""
    echo "── Scenario: ${scenario} (single run, DRL agents) ──"
    run python run_experiment.py --scenario "${scenario}" --mode single --agents all
done

# ── Step 2: Multi-seed robustness (10 seeds × 4 agents × 3 scenarios) ──
for scenario in turb noturb synth; do
    echo ""
    echo "── Multi-seed: ${scenario} (10 seeds) ──"
    run python run_experiment.py --scenario "${scenario}" --mode multiseed --seeds 10 \
        --timesteps 1000000 --output-dir results/multi_seed
done

# ── Step 3: Analyse multi-seed results ──
echo ""
echo "── Analysing multi-seed results ──"
run python -m evaluation.analyze_multiseed \
    --input-dir results/multi_seed

echo ""
echo "============================================"
echo "  Done. Results saved under results/"
echo "============================================"
