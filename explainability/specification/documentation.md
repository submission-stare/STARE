# Benchmark Pipeline Documentation

This document describes the results to be analyzed: benchmark of differente Reinforcement Learning (RL) and Large Language Model (LLM) trading agents on real historical market data and subsequently stress-test them on synthetic regime data (e.g., bull, bear, and lateral markets). 

## Overview

The **Benchmark Pipeline** is designed to train Reinforcement Learning (RL) and Large Language Model (LLM) trading agents on real historical market data and subsequently stress-test them on synthetic regime data (e.g., bull, bear, and lateral markets). 

**Key Objectives:**
- **Training Phase:** Agents learn trading policies using real historical market data (e.g., 1990–2025).
- **Testing Phase:** Agents are evaluated on synthetic financial data generated (typically via TimeGAN) to replicate specific market regimes.
- **Robustness:** Runs multiple trials with different random seeds in parallel to aggregate statistically significant results.

## Architecture

The pipeline leverages a `ProcessPoolExecutor` pattern to run multiple benchmarking rounds concurrently. 
1. **Data Pre-loading:** Real and synthetic regime data are pre-loaded and processed with financial indicators *once* in the main process to save memory and time.
2. **Dataset Stitching:** Instead of testing each regime independently in isolated blocks, the pipeline natively interleaves multiple regime datasets (e.g., `bull_1`, `bear_1`, `lateral_1`) into a single **continuous test dataset**. It re-scales prices at regime intersections to prevent unrealistic price jumps while maintaining continuous calendar business days.
3. **Parallel Execution:** A worker function (`_run_single_benchmark`) is launched for each run, inheriting the pre-loaded data.
4. **Agent Handling:** RL agents can be either trained per-run or loaded from a pre-trained "master" set (using `--train-once`). LLM agents (e.g., via OpenRouter) are always instantiated fresh as they do not maintain local weights.

## Data Flow

### 1. Training (Real Data)
- Downloads historical data for the configured `tickers` between `train.start_date` and `train.end_date`.
- Applies technical indicators.
- Agents are trained for `total_timesteps` in the `TradingEnv`.

### 2. Testing (Synthetic Regime Data)
- Loads configured regimes from their respective `data_dir` directories.
- Stitches the CSV files into a continuous interleaved dataset.
- Agents trade in the environment using this synthetic out-of-sample data.
- An equal-weight **benchmark** (passive buy & hold) is calculated for comparison.

## Outputs

The pipeline generates the following outputs under `results/pipeline_outputs/<output_dir>`:

1. **`run_*/`**: Individual run directories containing:
   - `agent_real_*`: Saved SB3 zipped agent models (if trained per run).
   - `interleaved/account_values.csv`: The daily portfolio value of each agent and the passive benchmark.
   - Financial metrics reports.
2. **`aggregated_interleaved/`**: Aggregation of returns and risk metrics (e.g., Sharpe Ratio) across all parallel runs. Provides confidence intervals and averages.
3. **`run_metadata.json`**: A dump of the configuration, runtime arguments, elapsed time, and which seeds were used.
4. **`data_verification/`**: Raw and processed CSV dumps of the exact training and testing datasets fed to the environment.

