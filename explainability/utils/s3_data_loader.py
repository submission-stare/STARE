"""
S3 Transaction Data Loader

Loads transaction logs and portfolio snapshots from the local S3 cache
for all available agents. Produces a unified DataFrame for each data type
and generates verification artifacts in the ``check_input_data/`` directory.

Data cleaning:
  - Snapshots: the last row per agent is a duplicate of step 0 (environment
    reset artefact).  It is dropped automatically on load.

Expected S3 path structure (already synced to .data_cache/):
  .data_cache/run_000/interleaved/agents_trading/trading_analysis/<agent>/transactions.csv
  .data_cache/run_000/interleaved/agents_trading/trading_analysis/<agent>/snapshots.csv

Agents may be flat names (a2c, ppo, sac, ddpg, td3) or nested paths
(openrouter/openai/gpt-5.2-chat, openrouter/google/gemini-3-flash-preview).
"""

import os
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from utils.data_scope import get_data_scope

logger = logging.getLogger(__name__)

# ── Default base inside the inspector repo ──────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_SCOPE = get_data_scope()
DEFAULT_CACHE_BASE = os.path.join(
    _PROJECT_ROOT,
    ".data_cache",
    _DATA_SCOPE["run_dir"],
    _DATA_SCOPE["trading_regime_dir"],
    "agents_trading",
    "trading_analysis",
)
CHECK_INPUT_DATA_DIR = os.path.join(_PROJECT_ROOT, "check_input_data")


def _snapshot_time_column(snapshots: pd.DataFrame) -> Optional[str]:
    for column in ("step", "date"):
        if column in snapshots.columns:
            return column
    return None


def _discover_agents(base_dir: str) -> Dict[str, str]:
    """
    Walk *base_dir* and find every folder that contains both
    ``transactions.csv`` and ``snapshots.csv``.

    Returns a dict  { agent_label : absolute_dir_path }.
    """
    agents: Dict[str, str] = {}
    for root, _dirs, files in os.walk(base_dir):
        if "transactions.csv" in files and "snapshots.csv" in files:
            # Build a human-readable label from the relative path
            rel = os.path.relpath(root, base_dir)
            label = rel.replace(os.sep, "/")  # e.g. "openrouter/openai/gpt-5.2-chat"
            agents[label] = root
    logger.info(f"Discovered {len(agents)} agents: {list(agents.keys())}")
    return agents


def load_transactions(base_dir: str = DEFAULT_CACHE_BASE) -> pd.DataFrame:
    """
    Load and concatenate ``transactions.csv`` from every agent folder.
    Adds an ``agent`` column for identification.
    """
    agents = _discover_agents(base_dir)
    frames = []
    for label, path in agents.items():
        fpath = os.path.join(path, "transactions.csv")
        try:
            df = pd.read_csv(fpath)
            df["agent"] = label
            frames.append(df)
            logger.info(f"Loaded transactions for {label}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load transactions for {label}: {e}")
    if not frames:
        logger.warning("No transaction data found.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total transactions loaded: {len(combined)} rows across {len(agents)} agents")
    return combined


def load_snapshots(base_dir: str = DEFAULT_CACHE_BASE) -> pd.DataFrame:
    """
    Load and concatenate ``snapshots.csv`` from every agent folder.
    Adds an ``agent`` column for identification.

    The last row of each agent's snapshot is an environment-reset artefact
    (step 0 duplicated at the end) and is **dropped** automatically.
    """
    agents = _discover_agents(base_dir)
    frames = []
    for label, path in agents.items():
        fpath = os.path.join(path, "snapshots.csv")
        try:
            df = pd.read_csv(fpath)
            # Drop the spurious last row (env reset: step goes back to 0)
            if len(df) > 1 and "step" in df.columns and df["step"].iloc[-1] == 0:
                dropped_row = df.iloc[-1]
                df = df.iloc[:-1]
                logger.info(
                    f"Dropped reset row for {label}: step={int(dropped_row['step'])}, "
                    f"date={dropped_row.get('date','?')}, total_asset={dropped_row.get('total_asset','?')}"
                )
            df["agent"] = label
            frames.append(df)
            logger.info(f"Loaded snapshots for {label}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load snapshots for {label}: {e}")
    if not frames:
        logger.warning("No snapshot data found.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total snapshots loaded: {len(combined)} rows across {len(agents)} agents")
    return combined


def load_all(
    base_dir: str = DEFAULT_CACHE_BASE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load both transactions and snapshots."""
    return load_transactions(base_dir), load_snapshots(base_dir)


# ── Verification: check_input_data ──────────────────────────────────

def generate_check_input_data(
    transactions: pd.DataFrame,
    snapshots: pd.DataFrame,
    output_dir: str = CHECK_INPUT_DATA_DIR,
) -> str:
    """
    Create the ``check_input_data/`` folder with:
      - ``transactions_all.csv``   : full combined transaction data
      - ``snapshots_all.csv``      : full combined snapshot data
      - ``check_verification.png`` : multi-panel chart for visual QA

    Returns the path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Save full CSVs for inspection ───────────────────────────────
    tx_path = os.path.join(output_dir, "transactions_all.csv")
    sn_path = os.path.join(output_dir, "snapshots_all.csv")
    transactions.to_csv(tx_path, index=False)
    snapshots.to_csv(sn_path, index=False)
    logger.info(f"Saved full transactions ({len(transactions)} rows) to {tx_path}")
    logger.info(f"Saved full snapshots ({len(snapshots)} rows) to {sn_path}")

    # ── Build verification figure ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Data Verification – check_input_data", fontsize=14, fontweight="bold")

    # Panel 1: Transaction counts per agent
    if not transactions.empty and "agent" in transactions.columns:
        tx_counts = transactions.groupby("agent").size()
        tx_counts.plot.bar(ax=axes[0, 0], color="steelblue", edgecolor="black")
        axes[0, 0].set_title("Transaction Count by Agent")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=30)
    else:
        axes[0, 0].text(0.5, 0.5, "No transaction data", ha="center", va="center")

    # Panel 2: Action type distribution per agent
    if not transactions.empty and "action_type" in transactions.columns:
        action_pivot = transactions.groupby(["agent", "action_type"]).size().unstack(fill_value=0)
        action_pivot.plot.bar(ax=axes[0, 1], edgecolor="black")
        axes[0, 1].set_title("Action Types by Agent")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis="x", rotation=30)
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "No action_type data", ha="center", va="center")

    snapshot_time_col = _snapshot_time_column(snapshots)

    # Panel 3: Portfolio total_asset evolution per agent (from snapshots)
    if not snapshots.empty and "total_asset" in snapshots.columns and snapshot_time_col:
        for agent_name, grp in snapshots.groupby("agent"):
            axes[1, 0].plot(grp[snapshot_time_col], grp["total_asset"], label=agent_name, alpha=0.8)
        axes[1, 0].set_title("Portfolio Value Over Time")
        axes[1, 0].set_xlabel(snapshot_time_col.title())
        axes[1, 0].set_ylabel("Total Asset")
        axes[1, 0].tick_params(axis="x", rotation=30)
        axes[1, 0].legend(fontsize=7, loc="best")
    else:
        axes[1, 0].text(0.5, 0.5, "No total_asset data", ha="center", va="center")

    # Panel 4: Cash weight evolution per agent
    if not snapshots.empty and "cash_weight" in snapshots.columns and snapshot_time_col:
        for agent_name, grp in snapshots.groupby("agent"):
            axes[1, 1].plot(grp[snapshot_time_col], grp["cash_weight"], label=agent_name, alpha=0.8)
        axes[1, 1].set_title("Cash Weight Over Time")
        axes[1, 1].set_xlabel(snapshot_time_col.title())
        axes[1, 1].set_ylabel("Cash Weight")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].legend(fontsize=7, loc="best")
    else:
        axes[1, 1].text(0.5, 0.5, "No cash_weight data", ha="center", va="center")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "check_verification.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved verification plot to {plot_path}")

    return output_dir


def build_transaction_summary(transactions: pd.DataFrame) -> str:
    """
    Build a concise textual summary of transaction data suitable for LLM context.
    Keeps token usage minimal while conveying key patterns.
    """
    if transactions.empty:
        return "No transaction data available."

    lines = ["=== Transaction Data Summary ==="]

    # Per-agent stats
    for agent, grp in transactions.groupby("agent"):
        total = len(grp)
        buys = (grp["action_type"] == "BUY").sum() if "action_type" in grp.columns else 0
        sells = (grp["action_type"] == "SELL").sum() if "action_type" in grp.columns else 0
        holds = (grp["action_type"] == "HOLD").sum() if "action_type" in grp.columns else 0
        lines.append(f"\nAgent: {agent}")
        lines.append(f"  Total transactions: {total}")
        lines.append(f"  BUY: {buys} | SELL: {sells} | HOLD: {holds}")
        if "gross_value" in grp.columns:
            buy_val = grp.loc[grp["action_type"] == "BUY", "gross_value"].sum() if buys else 0
            sell_val = grp.loc[grp["action_type"] == "SELL", "gross_value"].sum() if sells else 0
            lines.append(f"  Total buy value: {buy_val:,.2f} | Total sell value: {sell_val:,.2f}")
        if "ticker" in grp.columns:
            active_trades = grp.loc[grp["action_type"].isin(["BUY", "SELL"])]
            if not active_trades.empty:
                top_tickers = active_trades["ticker"].value_counts().head(5)
                lines.append(f"  Most traded tickers: {dict(zip(top_tickers.index, top_tickers.values.tolist()))}")

    return "\n".join(lines)


def build_snapshot_summary(snapshots: pd.DataFrame) -> str:
    """
    Build a concise textual summary of portfolio snapshot data for LLM context.
    """
    if snapshots.empty:
        return "No snapshot data available."

    lines = ["=== Portfolio Snapshot Summary ==="]

    for agent, grp in snapshots.groupby("agent"):
        lines.append(f"\nAgent: {agent}")
        time_col = _snapshot_time_column(grp)
        if time_col:
            label = "Steps" if time_col == "step" else "Dates"
            lines.append(f"  {label} covered: {grp[time_col].min()} – {grp[time_col].max()}")
        else:
            lines.append(f"  Snapshot rows: {len(grp)}")
        if "total_asset" in grp.columns:
            start_val = grp["total_asset"].iloc[0]
            end_val = grp["total_asset"].iloc[-1]
            peak_val = grp["total_asset"].max()
            min_val = grp["total_asset"].min()
            pct_change = ((end_val - start_val) / start_val * 100) if start_val else 0
            lines.append(f"  Portfolio: {start_val:,.2f} → {end_val:,.2f} ({pct_change:+.2f}%)")
            lines.append(f"  Peak: {peak_val:,.2f} | Trough: {min_val:,.2f}")
        if "cash_weight" in grp.columns:
            min_cash = grp["cash_weight"].min()
            max_cash = grp["cash_weight"].max()
            lines.append(f"  Cash weight range: {min_cash:.2%} – {max_cash:.2%}")

        # Find ticker columns (those ending in _weight but not cash_weight)
        weight_cols = [c for c in grp.columns if c.endswith("_weight") and c != "cash_weight"]
        if weight_cols:
            # Show peak allocations across entire history (more informative than final)
            peak_weights = grp[weight_cols].max()
            top_weights = peak_weights.sort_values(ascending=False).head(3)
            formatted = {c.replace("_weight", ""): f"{v:.2%}" for c, v in top_weights.items()}
            lines.append(f"  Peak allocations (max weight ever): {formatted}")

    return "\n".join(lines)
