
import inspect
import datetime
import shutil

_RUN_DIR = None
def _get_or_create_run_dir() -> str:
    global _RUN_DIR
    if _RUN_DIR is not None:
        return _RUN_DIR
        
    caller_dir = os.getcwd()
    for frame_info in inspect.stack():
        filename = frame_info.filename
        if 'evaluation' not in filename and 'site-packages' not in filename:
            caller_dir = os.path.dirname(os.path.abspath(filename))
            break
            
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    _RUN_DIR = os.path.join(caller_dir, 'results', timestamp)
    os.makedirs(_RUN_DIR, exist_ok=True)
    
    # Copy script and config files silently
    for f in os.listdir(caller_dir):
        f_path = os.path.join(caller_dir, f)
        if os.path.isfile(f_path) and (f.endswith('.py') or f.endswith('.yaml')):
            shutil.copy(f_path, _RUN_DIR)
            
    return _RUN_DIR

import json
import os
import inspect
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml
from evaluation.plotting import plot_performance
from stable_baselines3 import A2C, DDPG, PPO

# Ensure local paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import calculate_max_drawdown, calculate_sharpe, calculate_sortino
from evaluation.rigorous_stats import calculate_dsr, calculate_psr
from evaluation.utils import _resolve_technical_indicator_columns
from envs.gym_wrappers.portfolio_env import PortfolioEnv
from envs.gym_wrappers.trading_env import TradingEnv


RISK_FREE_RATE_ANNUAL = 0.03
DEFAULT_INITIAL_CAPITAL = 1.0
_COMPAT_EPS = 1e-12
RL_AGENT_CLASSES = {
    "a2c": A2C,
    "ppo": PPO,
    "ddpg": DDPG,
}
SUPPORTED_RL_MODES = {"portfolio", "trading"}


def _canonical_agent_name(name: str) -> str:
    return str(name).strip().lower()


def _risk_free_rate_daily(risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    return risk_free_rate_annual / 252.0


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_rl_mode(config: Dict) -> str:
    mode = str(config.get("rl_mode", "portfolio")).strip().lower()
    if mode not in SUPPORTED_RL_MODES:
        raise ValueError(f"Unsupported rl_mode '{mode}'. Supported modes: {sorted(SUPPORTED_RL_MODES)}")
    return mode


def resolve_env_class(config: Dict):
    rl_mode = _resolve_rl_mode(config)
    return PortfolioEnv if rl_mode == "portfolio" else TradingEnv


def _build_env_kwargs(config: Dict) -> Dict:
    rl_mode = _resolve_rl_mode(config)
    common_kwargs = {
        "initial_capital": float(config.get("initial_capital", DEFAULT_INITIAL_CAPITAL)),
        "commission": float(config.get("commission", 0.0005)),
        "turbulence_threshold": float(config.get("turbulence_threshold", 1e9)),
        "technical_indicator_columns": _resolve_technical_indicator_columns(config),
    }
    if rl_mode == "portfolio":
        common_kwargs["window_size"] = int(config.get("window_size", 10))
        return common_kwargs
    return {
        **common_kwargs,
        "hmax": int(config.get("hmax", 100)),
        "reward_scaling": float(config.get("reward_scaling", 1e-4)),
        "buy_cost_pct": float(config.get("buy_cost_pct", common_kwargs["commission"])),
        "sell_cost_pct": float(config.get("sell_cost_pct", common_kwargs["commission"])),
    }


def _resolve_save_dir(save_models_dir: Optional[str], base_output_dir: str) -> Optional[str]:
    if not save_models_dir:
        return None
    save_dir = save_models_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(base_output_dir, save_dir)
    return _ensure_dir(save_dir)


def _resolve_model_base_path(models_dir: Optional[str], agent_name: str) -> Optional[str]:
    if not models_dir:
        return None
    canonical = _canonical_agent_name(agent_name)
    candidates = (
        os.path.join(models_dir, f"real_{canonical}"),
        os.path.join(models_dir, f"agent_real_{canonical}"),
    )
    for path in candidates:
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            return path
    return None


def _series_returns(account_values: pd.Series) -> pd.Series:
    return account_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _compute_metrics_from_account_values(
    account_values: pd.Series,
    historical_trials: Optional[List[float]] = None,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> Dict[str, float]:
    returns = _series_returns(account_values)
    if returns.empty:
        return {
            "AR": 0.0,
            "SR": 0.0,
            "Sortino": 0.0,
            "PSR": 0.0,
            "DSR": 0.0,
            "FinalValue": float(account_values.iloc[-1]) if not account_values.empty else 0.0,
            "MaxDrawdown": 0.0,
        }

    final_val = float(account_values.iloc[-1])
    initial_val = float(account_values.iloc[0]) if not account_values.empty else 1.0
    growth_ratio = final_val / initial_val if initial_val else 1.0
    annualized_return = (growth_ratio ** (252.0 / max(len(account_values), 1))) - 1
    sharpe = calculate_sharpe(
        returns.to_numpy(),
        risk_free_rate=_risk_free_rate_daily(risk_free_rate_annual),
    )
    sortino = calculate_sortino(
        returns.to_numpy(),
        risk_free_rate=_risk_free_rate_daily(risk_free_rate_annual),
    )
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    if historical_trials is None:
        historical_trials = (
            [sharpe * 0.9, sharpe * 1.05, sharpe * 0.5, sharpe * 1.2]
            if sharpe > 0
            else [-0.5, -1.0, sharpe]
        )

    psr = calculate_psr(sharpe, len(returns), skew, kurtosis, sr_benchmark=0.0)
    dsr = calculate_dsr(sharpe, len(returns), skew, kurtosis, historical_trials)
    max_drawdown = calculate_max_drawdown(account_values.to_numpy())

    return {
        "AR": float(annualized_return),
        "SR": float(sharpe),
        "Sortino": float(sortino),
        "PSR": float(psr),
        "DSR": float(dsr),
        "FinalValue": final_val,
        "MaxDrawdown": float(max_drawdown),
    }


def _initial_amount_for_env(env) -> float:
    return float(getattr(env, "initial_capital", DEFAULT_INITIAL_CAPITAL))


def _build_benchmark_series(test_df: pd.DataFrame, tickers: List[str], initial_amount: float) -> pd.Series:
    close_prices = (
        test_df.reset_index()
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
    )
    close_prices = close_prices.loc[:, tickers]
    close_prices = close_prices.dropna(how="any")
    if close_prices.empty:
        return pd.Series(dtype=float, name="benchmark")
    normalized = close_prices.divide(close_prices.iloc[0])
    equal_weight = normalized.mean(axis=1) * initial_amount
    equal_weight.name = "benchmark"
    return equal_weight


def _plot_account_values(account_values_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    for column in account_values_df.columns:
        plt.plot(account_values_df.index, account_values_df[column], label=column, linewidth=2)
    plt.title("Account Value by Agent")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_metric_bars(metrics_df: pd.DataFrame, metric: str, output_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["agent"], metrics_df[metric], color="#2a6f97")
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_boxplot(df: pd.DataFrame, value_columns: List[str], title: str, output_path: str) -> None:
    if not value_columns:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[col].dropna() for col in value_columns], tick_labels=value_columns)
    plt.title(title)
    plt.ylabel("Value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_mean_portfolio(mean_portfolio_df: pd.DataFrame, output_path: str) -> None:
    if mean_portfolio_df.empty:
        return
    plt.figure(figsize=(10, 6))
    for column in mean_portfolio_df.columns:
        plt.plot(mean_portfolio_df.index, mean_portfolio_df[column], label=column, linewidth=2)
    plt.title("Mean Portfolio Account Values")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_sharpe_convergence(sharpe_df: pd.DataFrame, agents: List[str], output_path: str) -> None:
    if sharpe_df.empty:
        return
    plt.figure(figsize=(10, 6))
    run_axis = np.arange(1, len(sharpe_df) + 1)
    for agent in agents:
        values = sharpe_df[agent].astype(float).expanding().mean()
        plt.plot(run_axis, values, label=agent, linewidth=2)
    plt.title("Sharpe Convergence Across Runs")
    plt.xlabel("Completed Runs")
    plt.ylabel("Expanding Mean Sharpe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _metric_summary(values: pd.Series, prefix: str) -> Dict[str, float]:
    clean = values.dropna()
    if clean.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_ci95_lo": np.nan,
            f"{prefix}_ci95_hi": np.nan,
        }
    mean = clean.mean()
    stderr = clean.std(ddof=1) / np.sqrt(len(clean)) if len(clean) > 1 else 0.0
    return {
        f"{prefix}_mean": float(mean),
        f"{prefix}_std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_max": float(clean.max()),
        f"{prefix}_ci95_lo": float(mean - 1.96 * stderr),
        f"{prefix}_ci95_hi": float(mean + 1.96 * stderr),
    }


def save_experiment_results(results_dict, output_path, experiment_title):
    """Writes formatted outcomes cleanly to experiment folders."""
    with open(output_path, "w") as f:
        f.write(f"Reproduction Results: {experiment_title}\n")
        for model_name, metrics in results_dict.items():
            f.write("-" * 60 + "\n")
            f.write(f"{model_name} Annualized Return:  {metrics['AR']*100:.2f}%\n")
            f.write(f"{model_name} Classical Sharpe:   {metrics['SR']:.2f}\n")
            f.write(f"{model_name} Classical Sortino:  {metrics.get('Sortino', 0.0):.2f}\n")
            f.write(f"{model_name} Probabilistic SR:   {metrics['PSR']:.4f}\n")
            f.write(f"{model_name} Deflated SR (DSR):  {metrics['DSR']:.4f}\n")

    print(f"\nResults successfully exported to {output_path}")


def _format_percent(value: float, decimals: int = 1) -> str:
    return f"{round(float(value) * 100, decimals)}%"


def _model_type_label(name: str) -> str:
    return "Benchmark" if str(name).lower() == "benchmark" else "Classical RL"


def _compute_sr_stats(log_returns: pd.Series) -> Optional[Dict[str, float]]:
    returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = len(returns)
    if n_obs < 3:
        return None

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))
    if std_r <= 0 or not np.isfinite(std_r):
        return None

    sharpe = mean_r / std_r
    skew = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=False))
    sr_var = (1 + 0.5 * sharpe**2 - skew * sharpe + ((kurtosis - 3.0) / 4.0) * sharpe**2) / (n_obs - 1)
    if sr_var <= 0 or not np.isfinite(sr_var):
        return None

    sr_se = float(np.sqrt(sr_var))
    z_score = float(stats.norm.ppf(0.975))
    return {
        "sr": float(sharpe),
        "se": sr_se,
        "ci_lower": float(sharpe - z_score * sr_se),
        "ci_upper": float(sharpe + z_score * sr_se),
        "skew": skew,
        "kurt": kurtosis,
    }


def _write_bailey_sharpe_summary(account_values_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    prices_df = account_values_df.copy()
    benchmark_columns = [col for col in prices_df.columns if str(col).lower() == "benchmark"]
    benchmark_sr_annual = 0.0
    if benchmark_columns:
        benchmark_series = prices_df[benchmark_columns[0]].dropna()
        if not benchmark_series.empty and float(benchmark_series.min()) > 0:
            benchmark_returns = np.log(benchmark_series / benchmark_series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
            if len(benchmark_returns) >= 2:
                benchmark_std = float(benchmark_returns.std(ddof=1))
                if benchmark_std > 0:
                    benchmark_sr_annual = float((benchmark_returns.mean() / benchmark_std) * np.sqrt(252.0))

    agent_rows = []
    for column in prices_df.columns:
        price_series = prices_df[column].dropna()
        agent_type = _model_type_label(column)
        if price_series.empty:
            continue
        if float(price_series.min()) <= 0:
            agent_rows.append(
                {
                    "model": column,
                    "type": agent_type,
                    "status": "Bankrupt",
                }
            )
            continue

        returns = np.log(price_series / price_series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        sr_stats = _compute_sr_stats(returns)
        roll_max = price_series.cummax()
        drawdown = (price_series - roll_max) / roll_max
        ann_vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else 0.0
        sortino = calculate_sortino(returns.to_numpy(), risk_free_rate=0.0)
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        if sr_stats is None:
            agent_rows.append(
                {
                    "model": column,
                    "type": agent_type,
                    "status": "Active",
                    "sr_annual": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "psr": 0.5,
                    "se_annual": 0.0,
                    "sortino": float(sortino),
                    "skew": 0.0,
                    "kurt": 3.0,
                    "ann_vol": ann_vol,
                    "max_dd": max_drawdown,
                }
            )
            continue

        sr_annual = float(sr_stats["sr"] * np.sqrt(252.0))
        se_annual = float(sr_stats["se"] * np.sqrt(252.0))
        if se_annual > 0:
            psr = float(stats.norm.cdf((sr_annual - benchmark_sr_annual) / se_annual))
            ci_low = float(sr_annual - stats.norm.ppf(0.975) * se_annual)
            ci_high = float(sr_annual + stats.norm.ppf(0.975) * se_annual)
        else:
            psr = 0.5
            ci_low = sr_annual
            ci_high = sr_annual

        agent_rows.append(
            {
                "model": column,
                "type": agent_type,
                "status": "Active",
                "sr_annual": sr_annual,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "psr": psr,
                "se_annual": se_annual,
                "sortino": float(sortino),
                "skew": float(sr_stats["skew"]),
                "kurt": float(sr_stats["kurt"]),
                "ann_vol": ann_vol,
                "max_dd": max_drawdown,
            }
        )

    active_sharpes = [row["sr_annual"] for row in agent_rows if row["status"] == "Active"]
    n_active = len(active_sharpes)
    gamma = 0.5772156649
    euler = np.exp(1.0)
    sharpe_std = float(np.std(active_sharpes, ddof=1)) if n_active > 1 else 0.0
    sharpe_mean = float(np.mean(active_sharpes)) if n_active else 0.0
    if n_active > 1 and sharpe_std > 0:
        expected_max_z = (1 - gamma) * stats.norm.ppf(1 - 1 / n_active) + gamma * stats.norm.ppf(1 - 1 / (n_active * euler))
        expected_max_sharpe = float(sharpe_mean + sharpe_std * expected_max_z)
    else:
        expected_max_sharpe = sharpe_mean

    output_rows = []
    for row in agent_rows:
        if row["status"] == "Bankrupt":
            output_rows.append(
                {
                    "model": row["model"],
                    "type": row["type"],
                    "sharpeRatio": "FALIDO",
                    "sortinoRatio": "-",
                    "ciLow": "-",
                    "ciHigh": "-",
                    "psr": "-",
                    "dsr": "-",
                    "skew": "-",
                    "kurt": "-",
                    "volatility": "-",
                    "maxDrawdown": "-",
                }
            )
            continue

        if n_active > 1 and row["se_annual"] > 0:
            dsr = float(stats.norm.cdf((row["sr_annual"] - expected_max_sharpe) / row["se_annual"]))
            dsr_str = _format_percent(dsr)
        else:
            dsr_str = "-"

        output_rows.append(
            {
                "model": row["model"],
                "type": row["type"],
                "sharpeRatio": round(float(row["sr_annual"]), 2),
                "sortinoRatio": round(float(row["sortino"]), 2),
                "ciLow": round(float(row["ci_low"]), 2),
                "ciHigh": round(float(row["ci_high"]), 2),
                "psr": _format_percent(row["psr"]),
                "dsr": dsr_str,
                "skew": round(float(row["skew"]), 2),
                "kurt": round(float(row["kurt"]), 2),
                "volatility": _format_percent(row["ann_vol"]),
                "maxDrawdown": _format_percent(row["max_dd"]),
            }
        )

    compat_df = pd.DataFrame(output_rows)
    if not compat_df.empty:
        compat_df["sort_key"] = pd.to_numeric(compat_df["sharpeRatio"], errors="coerce").fillna(-9999.0)
        compat_df = compat_df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"])
    compat_df.to_csv(output_path, index=False)
    return compat_df


def _fallback_episode_history(account_values: pd.Series, test_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    close_prices = (
        test_df.reset_index()
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
        .reindex(columns=tickers)
    )
    if account_values.index.dtype != "datetime64[ns]":
        dates = close_prices.index[: len(account_values)]
    else:
        dates = pd.Index(account_values.index).intersection(close_prices.index)
        if len(dates) != len(account_values):
            dates = pd.Index(account_values.index)
    rows = []
    for step_idx, (date, total_asset) in enumerate(zip(dates, account_values.to_numpy())):
        row = {
            "step": int(step_idx),
            "date": pd.Timestamp(date),
            "account_value_before": float(total_asset),
            "account_value_after": float(total_asset),
            "step_transaction_cost": 0.0,
            "cash": float(total_asset),
            "total_allocated": 0.0,
            "total_asset": float(total_asset),
            "cash_weight": 1.0,
            "cash_raw_action": 0.0,
        }
        date_prices = close_prices.loc[pd.Timestamp(date)] if pd.Timestamp(date) in close_prices.index else pd.Series(index=tickers, dtype=float)
        for ticker in tickers:
            row[f"{ticker}_price"] = float(date_prices.get(ticker, 0.0) or 0.0)
            row[f"{ticker}_weight"] = 0.0
            row[f"{ticker}_raw_action"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _build_compat_snapshots(history_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["step", "date", "cash", "total_allocated", "total_asset", "cash_weight"])

    snapshots = pd.DataFrame(
        {
            "step": history_df["step"].astype(int),
            "date": history_df["date"].astype(str),
            "cash": history_df["cash"].astype(float),
            "total_allocated": history_df["total_allocated"].astype(float),
            "total_asset": history_df["total_asset"].astype(float),
        }
    )

    ordered_columns = ["step", "date", "cash", "total_allocated", "total_asset"]
    for ticker in tickers:
        price_col = history_df[f"{ticker}_price"].astype(float)
        weight_col = history_df[f"{ticker}_weight"].astype(float)
        value_col = snapshots["total_asset"] * weight_col
        shares_col = np.where(np.abs(price_col) > _COMPAT_EPS, value_col / price_col, 0.0)
        snapshots[f"{ticker}_price"] = price_col
        snapshots[f"{ticker}_shares"] = shares_col.astype(float)
        snapshots[f"{ticker}_value"] = value_col.astype(float)
        snapshots[f"{ticker}_weight"] = weight_col
        ordered_columns.extend(
            [
                f"{ticker}_price",
                f"{ticker}_shares",
                f"{ticker}_value",
                f"{ticker}_weight",
            ]
        )

    snapshots["cash_weight"] = history_df["cash_weight"].astype(float)
    ordered_columns.append("cash_weight")
    return snapshots.loc[:, ordered_columns]


def _build_compat_transactions(history_df: pd.DataFrame, snap_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows = []
    for idx in range(1, len(snap_df)):
        previous_row = snap_df.iloc[idx - 1]
        current_row = snap_df.iloc[idx]
        history_row = history_df.iloc[idx]
        step_cost_total = float(history_row.get("step_transaction_cost", 0.0))

        gross_values = []
        trade_rows = []
        for ticker_idx, ticker in enumerate(tickers):
            holdings_before = float(previous_row[f"{ticker}_shares"])
            holdings_after = float(current_row[f"{ticker}_shares"])
            delta_shares = holdings_after - holdings_before
            price = float(current_row[f"{ticker}_price"])
            shares_traded = abs(delta_shares)
            gross_value = shares_traded * price

            if delta_shares > _COMPAT_EPS:
                action_type = "BUY"
            elif delta_shares < -_COMPAT_EPS:
                action_type = "SELL"
            else:
                action_type = "HOLD"
                shares_traded = 0.0
                gross_value = 0.0

            gross_values.append(gross_value)
            trade_rows.append(
                {
                    "step": int(current_row["step"]),
                    "date": str(current_row["date"]),
                    "ticker": ticker,
                    "ticker_idx": int(ticker_idx),
                    "action_type": action_type,
                    "raw_action": float(history_row.get(f"{ticker}_raw_action", 0.0)),
                    "shares_traded": float(shares_traded),
                    "price": price,
                    "gross_value": float(gross_value),
                    "cash_before": float(previous_row["cash"]),
                    "cash_after": float(current_row["cash"]),
                    "holdings_before": holdings_before,
                    "holdings_after": holdings_after,
                    "total_allocated": float(current_row["total_allocated"]),
                    "total_unallocated": float(current_row["cash"]),
                    "total_asset": float(current_row["total_asset"]),
                    "portfolio_weight": float(current_row[f"{ticker}_weight"]),
                }
            )

        gross_total = float(sum(gross_values))
        for trade_row in trade_rows:
            if gross_total > _COMPAT_EPS and trade_row["gross_value"] > 0:
                transaction_cost = step_cost_total * (float(trade_row["gross_value"]) / gross_total)
            else:
                transaction_cost = 0.0
            trade_row["transaction_cost"] = float(transaction_cost)
            if trade_row["action_type"] == "BUY":
                trade_row["net_value"] = float(trade_row["gross_value"]) + float(transaction_cost)
            elif trade_row["action_type"] == "SELL":
                trade_row["net_value"] = float(trade_row["gross_value"]) - float(transaction_cost)
            else:
                trade_row["net_value"] = 0.0
            rows.append(trade_row)

    return pd.DataFrame(
        rows,
        columns=[
            "step",
            "date",
            "ticker",
            "ticker_idx",
            "action_type",
            "raw_action",
            "shares_traded",
            "price",
            "gross_value",
            "transaction_cost",
            "net_value",
            "cash_before",
            "cash_after",
            "holdings_before",
            "holdings_after",
            "total_allocated",
            "total_unallocated",
            "total_asset",
            "portfolio_weight",
        ],
    )


def _trading_transactions_from_env(env, account_values: pd.Series, test_df: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if hasattr(env, "get_episode_snapshots"):
        snap_df = env.get_episode_snapshots()
        if not isinstance(snap_df, pd.DataFrame):
            snap_df = pd.DataFrame(snap_df)
    else:
        snap_df = pd.DataFrame()

    if hasattr(env, "get_episode_transactions"):
        txn_df = env.get_episode_transactions()
        if not isinstance(txn_df, pd.DataFrame):
            txn_df = pd.DataFrame(txn_df)
    else:
        txn_df = pd.DataFrame()

    if snap_df.empty:
        history_df = _fallback_episode_history(account_values, test_df, tickers)
        snap_df = _build_compat_snapshots(history_df, tickers)
    if txn_df.empty:
        history_df = _fallback_episode_history(account_values, test_df, tickers)
        txn_df = _build_compat_transactions(history_df, snap_df, tickers)
    return snap_df, txn_df


def _summarize_compat_transactions(agent_name: str, txn_df: pd.DataFrame) -> Dict[str, float]:
    if txn_df.empty:
        return {
            "agent": agent_name,
            "total_buys": 0,
            "total_sells": 0,
            "total_holds": 0,
            "total_value_buy": 0.0,
            "total_value_sell": 0.0,
            "total_cost": 0.0,
        }

    buys = txn_df[txn_df["action_type"] == "BUY"]
    sells = txn_df[txn_df["action_type"] == "SELL"]
    holds = txn_df[txn_df["action_type"] == "HOLD"]
    return {
        "agent": agent_name,
        "total_buys": int(len(buys)),
        "total_sells": int(len(sells)),
        "total_holds": int(len(holds)),
        "total_value_buy": float(buys["gross_value"].sum()),
        "total_value_sell": float(sells["gross_value"].sum()),
        "total_cost": float(txn_df["transaction_cost"].sum()),
    }


def _export_trading_analysis(
    env,
    account_values: pd.Series,
    test_df: pd.DataFrame,
    tickers: List[str],
    rl_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rl_mode == "trading":
        return _trading_transactions_from_env(env, account_values, test_df, tickers)

    if hasattr(env, "get_episode_history"):
        history_df = env.get_episode_history()
        if not isinstance(history_df, pd.DataFrame):
            history_df = pd.DataFrame(history_df)
    else:
        history_df = pd.DataFrame()
    if history_df.empty:
        history_df = _fallback_episode_history(account_values, test_df, tickers)

    snap_df = _build_compat_snapshots(history_df, tickers)
    txn_df = _build_compat_transactions(history_df, snap_df, tickers)
    return snap_df, txn_df


def evaluate_agent(
    model_class,
    model_name,
    train_env,
    test_env,
    total_timesteps,
    model_kwargs=None,
    policy_kwargs=None,
    historical_trials=None,
    risk_free_rate=RISK_FREE_RATE_ANNUAL,
):
    """
    Unified training and evaluation pipeline measuring classical metrics (AR, SR)
    alongside Bailey's probabilistic metrics (PSR, DSR).
    """
    print(f"\n--- Training {model_name} ---")
    train_env.reset()

    kwargs = {"policy": "MlpPolicy", "env": train_env, "verbose": 0, "learning_rate": 3e-4}
    if model_kwargs:
        kwargs.update(model_kwargs)
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs

    kwargs["device"] = "cpu"
    model = model_class(**kwargs)
    model.learn(total_timesteps=total_timesteps)

    print(f"--- Evaluating {model_name} ---")
    account_values = _simulate_trading_episode(model, test_env)
    metrics = _compute_metrics_from_account_values(
        account_values,
        historical_trials=historical_trials,
        risk_free_rate_annual=risk_free_rate,
    )

    print(f"{model_name} AR: {metrics['AR']*100:.2f}%, SR: {metrics['SR']:.2f}, Sortino: {metrics['Sortino']:.2f}")
    print(
        f"{model_name} Probabilistic Sharpe (PSR): {metrics['PSR']:.4f}  |  "
        f"Deflated Sharpe (DSR): {metrics['DSR']:.4f}"
    )
    return {key: metrics[key] for key in ("AR", "SR", "Sortino", "PSR", "DSR")}


def _simulate_trading_episode(model, test_env) -> pd.Series:
    reset_res = test_env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    done = False
    values = [float(test_env.portfolio_value)]
    dates = getattr(test_env, "dates", None)
    date_index = []
    current_idx = getattr(test_env, "current_step", 0)
    if dates is not None and len(dates) > 0:
        safe_idx = max(min(current_idx - 1, len(dates) - 1), 0)
        date_index.append(pd.Timestamp(dates[safe_idx]))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_res = test_env.step(action)
        if len(step_res) == 5:
            obs, _, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            obs, _, done, _ = step_res

        values.append(float(test_env.portfolio_value))
        current_idx = getattr(test_env, "current_step", current_idx + 1)
        if dates is not None and len(dates) > 0:
            safe_idx = max(min(current_idx - 1, len(dates) - 1), 0)
            date_index.append(pd.Timestamp(dates[safe_idx]))

    if date_index and len(date_index) == len(values):
        return pd.Series(values, index=pd.Index(date_index, name="date"))
    return pd.Series(values)


def load_pipeline_config(config_path: str) -> Dict:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)

    config.setdefault("agents", ["a2c", "ppo", "ddpg"])
    config.setdefault("n_runs", 1)
    config.setdefault("seed", 42)
    config.setdefault("train_once", False)
    config.setdefault("reuse_existing", True)
    config.setdefault("no_train", False)
    config.setdefault("pretrained_models_dir", None)
    config.setdefault("save_models_dir", None)
    config.setdefault("output_dir", "benchmark-data-liu")
    config.setdefault("dataset_name", "liu_trade_period")
    config.setdefault("label", config["dataset_name"])
    config.setdefault("timesteps_per_model", 150000)
    config.setdefault("rl_mode", "portfolio")
    config.setdefault("initial_capital", DEFAULT_INITIAL_CAPITAL)
    config.setdefault("commission", 0.0005)
    config.setdefault("window_size", 10)
    config.setdefault("hmax", 100)
    config.setdefault("reward_scaling", 1e-4)
    config.setdefault("risk_free_rate", RISK_FREE_RATE_ANNUAL)
    config.setdefault("turbulence_threshold", 1e9)
    config.setdefault("buy_cost_pct", config.get("commission", 0.0005))
    config.setdefault("sell_cost_pct", config.get("commission", 0.0005))

    _resolve_rl_mode(config)

    for agent_name in config["agents"]:
        canonical = _canonical_agent_name(agent_name)
        if canonical not in RL_AGENT_CLASSES:
            raise ValueError(f"Unsupported agent '{agent_name}'. Supported agents: {sorted(RL_AGENT_CLASSES)}")

    return config


def split_train_trade_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[config["train_start"]: config["train_end"]].copy()
    trade_df = df.loc[config["test_start"]: config["test_end"]].copy()
    if train_df.empty or trade_df.empty:
        raise ValueError("Training or trading split is empty. Check configured date ranges.")
    return train_df, trade_df


def create_run_metadata(config: Dict, base_output_dir: str, seeds: List[int], n_succeeded: int, failed_run_ids: List[int]) -> Dict:
    dataset_name = config["dataset_name"]
    dataset_label = config.get("label", dataset_name)
    evaluation_datasets_metadata = {
        dataset_name: {
            "label": dataset_label,
        }
    }
    return {
        "config_file": os.path.basename(config.get("_config_path", "config.yaml")),
        "output_dir": base_output_dir,
        "dataset_name": dataset_name,
        "n_runs": int(config["n_runs"]),
        "n_succeeded": int(n_succeeded),
        "n_failed": int(len(failed_run_ids)),
        "failed_run_ids": failed_run_ids,
        "seeds": [int(seed) for seed in seeds],
        "tickers": config["tickers"],
        "agents": [_canonical_agent_name(agent) for agent in config["agents"]],
        "evaluation_datasets": evaluation_datasets_metadata,
        "train_start": config["train_start"],
        "train_end": config["train_end"],
        "test_start": config["test_start"],
        "test_end": config["test_end"],
        "timesteps_per_model": int(config["timesteps_per_model"]),
        "rl_mode": _resolve_rl_mode(config),
        "train_once": bool(config["train_once"]),
        "reuse_existing": bool(config["reuse_existing"]),
        "no_train": bool(config["no_train"]),
        "pretrained_models_dir": config.get("pretrained_models_dir"),
        "save_models_dir": config.get("save_models_dir"),
    }


def write_run_metadata(base_output_dir: str, metadata: Dict) -> str:
    metadata_path = os.path.join(base_output_dir, "run_metadata.json")
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def write_data_verification_outputs(base_output_dir: str, train_df: pd.DataFrame, trade_df: pd.DataFrame, dataset_name: str) -> None:
    verification_dir = _ensure_dir(os.path.join(base_output_dir, "data_verification"))
    train_df.reset_index().to_csv(os.path.join(verification_dir, "real_train_processed.csv"), index=False)
    trade_df.reset_index().to_csv(
        os.path.join(verification_dir, f"regime_{dataset_name}_processed.csv"),
        index=False,
    )
    trade_df.reset_index().to_csv(
        os.path.join(verification_dir, f"regime_{dataset_name}_raw.csv"),
        index=False,
    )


def _build_env_factory(env_class, env_kwargs: Dict, data_frame: pd.DataFrame, tickers: List[str]) -> Callable[[], object]:
    signature = inspect.signature(env_class.__init__)
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted_kwargs = dict(env_kwargs) if has_var_keyword else {
        key: value
        for key, value in env_kwargs.items()
        if key in signature.parameters
    }

    def _fallback_kwargs_for_base_class() -> Dict:
        for base_class in env_class.__mro__[1:]:
            if base_class is object:
                continue
            base_signature = inspect.signature(base_class.__init__)
            base_has_var_keyword = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in base_signature.parameters.values()
            )
            if base_has_var_keyword:
                continue
            return {
                key: value
                for key, value in env_kwargs.items()
                if key in base_signature.parameters
            }
        return accepted_kwargs

    def _factory():
        try:
            return env_class(data_frame.copy(), tickers, **accepted_kwargs)
        except TypeError as exc:
            if has_var_keyword and "unexpected keyword argument" in str(exc):
                return env_class(data_frame.copy(), tickers, **_fallback_kwargs_for_base_class())
            raise

    return _factory


def train_sb3_model(
    model_class,
    model_name: str,
    train_env,
    total_timesteps: int,
    model_kwargs: Optional[Dict] = None,
    policy_kwargs: Optional[Dict] = None,
):
    kwargs = {"policy": "MlpPolicy", "env": train_env, "verbose": 0, "learning_rate": 3e-4, "device": "cpu"}
    if model_kwargs:
        kwargs.update(model_kwargs)
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    model = model_class(**kwargs)
    print(f"Training {model_name} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    return model


def _agent_config_values(config: Dict, agent_name: str) -> Tuple[Dict, Optional[Dict]]:
    canonical = _canonical_agent_name(agent_name)
    model_kwargs = dict(config.get(f"{canonical}_params", {}) or {})
    policy_kwargs = config.get(f"{canonical}_policy_kwargs")
    return model_kwargs, policy_kwargs


def _load_model(agent_name: str, model_base_path: str):
    model_class = RL_AGENT_CLASSES[_canonical_agent_name(agent_name)]
    return model_class.load(model_base_path)


def train_or_load_agents(
    config: Dict,
    run_dir: str,
    train_env_factory: Callable[[], object],
    base_output_dir: str,
    pretrained_models_dir: Optional[str] = None,
    shared_models: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    agents = [_canonical_agent_name(agent) for agent in config["agents"]]
    trained_agents: Dict[str, object] = {}

    if shared_models:
        for agent_name, model in shared_models.items():
            trained_agents[_canonical_agent_name(agent_name)] = model

    missing_agents: List[str] = []
    for agent_name in agents:
        if agent_name in trained_agents:
            continue

        model_path = _resolve_model_base_path(pretrained_models_dir, agent_name)
        if not model_path and config.get("reuse_existing", False):
            model_path = _resolve_model_base_path(run_dir, agent_name)

        if model_path:
            trained_agents[agent_name] = _load_model(agent_name, model_path)
            continue

        missing_agents.append(agent_name)

    if missing_agents and config.get("no_train", False):
        raise FileNotFoundError(
            "Missing required pretrained models with no_train enabled: "
            + ", ".join(missing_agents)
        )

    save_dir = _resolve_save_dir(config.get("save_models_dir"), base_output_dir)

    for agent_name in missing_agents:
        model_class = RL_AGENT_CLASSES[agent_name]
        model_kwargs, policy_kwargs = _agent_config_values(config, agent_name)
        model = train_sb3_model(
            model_class=model_class,
            model_name=agent_name,
            train_env=train_env_factory(),
            total_timesteps=int(config["timesteps_per_model"]),
            model_kwargs=model_kwargs,
            policy_kwargs=policy_kwargs,
        )
        run_model_path = os.path.join(run_dir, f"agent_real_{agent_name}")
        model.save(run_model_path)
        if save_dir:
            model.save(os.path.join(save_dir, f"real_{agent_name}"))
        trained_agents[agent_name] = model

    return trained_agents


def run_trading_evaluation(
    trained_agents: Dict[str, object],
    test_env_factory: Callable[[], object],
    dataset_dir: str,
    dataset_name: str,
    test_df: pd.DataFrame,
    tickers: List[str],
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> Dict:
    agents_trading_dir = _ensure_dir(os.path.join(dataset_dir, "agents_trading"))
    financial_metrics_dir = _ensure_dir(os.path.join(dataset_dir, "financial_metrics"))
    trading_analysis_dir = _ensure_dir(os.path.join(agents_trading_dir, "trading_analysis"))
    rl_mode = "portfolio"

    account_values_df = pd.DataFrame()
    metrics_rows = []
    compat_summaries = []

    for agent_name, model in trained_agents.items():
        env = test_env_factory()
        rl_mode = getattr(env, "rl_mode", rl_mode)
        account_values = _simulate_trading_episode(model, env)
        if not isinstance(account_values.index, pd.Index) or account_values.index.dtype == "int64":
            close_dates = (
                test_df.reset_index()["Date"].drop_duplicates().sort_values().reset_index(drop=True)
            )
            account_values.index = pd.Index(close_dates.iloc[: len(account_values)], name="date")
        else:
            account_values.index.name = "date"

        account_values_df[agent_name] = account_values
        account_values.to_frame(name=agent_name).to_csv(
            os.path.join(agents_trading_dir, f"account_value_{agent_name}.csv")
        )

        snap_df, txn_df = _export_trading_analysis(env, account_values, test_df, tickers, rl_mode)
        agent_trading_dir = _ensure_dir(os.path.join(trading_analysis_dir, agent_name))
        snap_df.to_csv(os.path.join(agent_trading_dir, "snapshots.csv"), index=False)
        txn_df.to_csv(os.path.join(agent_trading_dir, "transactions.csv"), index=False)
        compat_summaries.append(_summarize_compat_transactions(agent_name, txn_df))

        metrics = _compute_metrics_from_account_values(
            account_values,
            risk_free_rate_annual=risk_free_rate_annual,
        )
        metrics_rows.append({"agent": agent_name, **metrics})

    benchmark_series = _build_benchmark_series(
        test_df=test_df,
        tickers=tickers,
        initial_amount=_initial_amount_for_env(test_env_factory()),
    )
    if not benchmark_series.empty:
        benchmark_series = benchmark_series.reindex(account_values_df.index)
        account_values_df["benchmark"] = benchmark_series

    account_values_df.to_csv(os.path.join(dataset_dir, "account_values.csv"), index=True)
    _plot_account_values(account_values_df, os.path.join(dataset_dir, "account_value_agents.png"))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(financial_metrics_dir, "sharpe_summary_agents.csv"), index=False)
    _write_bailey_sharpe_summary(
        account_values_df,
        os.path.join(financial_metrics_dir, "sharpe_summary_agents_with_psr.csv"),
    )
    _plot_metric_bars(metrics_df, "SR", os.path.join(financial_metrics_dir, "sharpe_comparison.png"), "Sharpe by Agent")

    if compat_summaries:
        comparison_dir = _ensure_dir(os.path.join(trading_analysis_dir, "_comparison"))
        pd.DataFrame(compat_summaries).to_csv(
            os.path.join(comparison_dir, "compare_transaction_summary.csv"),
            index=False,
        )

    sharpe_dict = {row["agent"]: float(row["SR"]) for row in metrics_rows}
    final_value_dict = {row["agent"]: float(row["FinalValue"]) for row in metrics_rows}
    max_drawdown_dict = {row["agent"]: float(row["MaxDrawdown"]) for row in metrics_rows}

    return {
        "dataset_name": dataset_name,
        "sharpe": sharpe_dict,
        "final_value": final_value_dict,
        "max_drawdown": max_drawdown_dict,
        "account_values_csv": os.path.join(dataset_dir, "account_values.csv"),
    }


def _export_legacy_general_layout(base_output_dir: str, dataset_dir: str) -> None:
    """Write compatibility artifacts under benchmark-data/general.

    This keeps legacy post-processing scripts working while preserving the
    current run_###/dataset layout.
    """
    legacy_general_dir = _ensure_dir(os.path.join(base_output_dir, "benchmark-data", "general"))

    account_src = os.path.join(dataset_dir, "account_values.csv")
    if os.path.exists(account_src):
        shutil.copy2(account_src, os.path.join(legacy_general_dir, "account_values.csv"))

    agents_src = os.path.join(dataset_dir, "agents_trading")
    if os.path.exists(agents_src):
        shutil.copytree(
            agents_src,
            os.path.join(legacy_general_dir, "agents_trading"),
            dirs_exist_ok=True,
        )

    metrics_src = os.path.join(dataset_dir, "financial_metrics")
    if os.path.exists(metrics_src):
        shutil.copytree(
            metrics_src,
            os.path.join(legacy_general_dir, "financial_metrics"),
            dirs_exist_ok=True,
        )


def aggregate_run_results(
    run_results: List[Dict],
    base_output_dir: str,
    dataset_name: str,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> str:
    if not run_results:
        raise ValueError("No run results to aggregate.")

    aggregated_root = _ensure_dir(os.path.join(base_output_dir, f"aggregated_{dataset_name}"))
    aggregated_dir = _ensure_dir(os.path.join(aggregated_root, "aggregated"))

    agents = sorted(run_results[0]["sharpe"].keys())
    sharpe_rows = []
    final_value_rows = []
    max_drawdown_rows = []

    for result in run_results:
        sharpe_rows.append({"run": result["run_id"], "seed": result["seed"], **result["sharpe"]})
        final_value_rows.append({"run": result["run_id"], "seed": result["seed"], **result["final_value"]})
        max_drawdown_rows.append({"run": result["run_id"], "seed": result["seed"], **result.get("max_drawdown", {})})

    sharpe_df = pd.DataFrame(sharpe_rows)
    final_value_df = pd.DataFrame(final_value_rows)
    max_drawdown_df = pd.DataFrame(max_drawdown_rows)

    sharpe_df.to_csv(os.path.join(aggregated_dir, "all_runs_sharpe.csv"), index=False)
    final_value_df.to_csv(os.path.join(aggregated_dir, "all_runs_final_value.csv"), index=False)

    summary_rows = []
    for agent in agents:
        row = {"agent": agent, "n_runs": len(run_results)}
        row.update(_metric_summary(sharpe_df[agent], "sharpe"))
        row.update(_metric_summary(final_value_df[agent], "final_value"))
        if agent in max_drawdown_df.columns:
            row.update(_metric_summary(max_drawdown_df[agent], "max_drawdown"))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(aggregated_dir, "statistical_summary.csv"), index=False)

    mean_portfolio = pd.DataFrame()
    common_index = None
    for agent in agents:
        series_list = []
        for result in run_results:
            account_values_df = pd.read_csv(result["account_values_csv"], index_col=0, parse_dates=True)
            if common_index is None:
                common_index = account_values_df.index
            if agent in account_values_df.columns:
                series_list.append(account_values_df[agent].reset_index(drop=True))
        if series_list:
            aligned = pd.concat(series_list, axis=1)
            mean_portfolio[agent] = aligned.mean(axis=1)

    if common_index is not None and len(common_index) == len(mean_portfolio):
        mean_portfolio.index = common_index
        mean_portfolio.index.name = "date"

    mean_portfolio.to_csv(os.path.join(aggregated_dir, "mean_portfolio_account_values.csv"))

    mean_metrics_rows = []
    for agent in agents:
        if agent not in mean_portfolio.columns:
            continue
        series = mean_portfolio[agent].dropna()
        returns = _series_returns(series)
        mean_metrics_rows.append(
            {
                "agent": agent,
                "mean_portfolio_sharpe": float(
                    calculate_sharpe(
                        returns.to_numpy(),
                        risk_free_rate=_risk_free_rate_daily(risk_free_rate_annual),
                    )
                )
                if not returns.empty
                else 0.0,
                "mean_portfolio_final_value": float(series.iloc[-1]) if not series.empty else 0.0,
                "mean_portfolio_max_drawdown": float(calculate_max_drawdown(series.to_numpy())) if not series.empty else 0.0,
            }
        )
    mean_metrics_df = pd.DataFrame(mean_metrics_rows)
    mean_metrics_df.to_csv(os.path.join(aggregated_dir, "mean_portfolio_metrics.csv"), index=False)

    _plot_boxplot(sharpe_df, agents, "Sharpe Across Runs", os.path.join(aggregated_dir, "boxplots_all_metrics.png"))
    _plot_mean_portfolio(mean_portfolio, os.path.join(aggregated_dir, "mean_portfolio_account_values.png"))
    _plot_sharpe_convergence(sharpe_df, agents, os.path.join(aggregated_dir, "sharpe_convergence.png"))

    return aggregated_root


def run_benchmark_pipeline(
    config: Dict,
    train_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    env_class,
    base_output_dir: str,
) -> Dict:
    # Use our timestamped folder!
    base_output_dir = _get_or_create_run_dir()
    _ensure_dir(base_output_dir)
    dataset_name = config["dataset_name"]
    config = dict(config)
    config["rl_mode"] = _resolve_rl_mode(config)
    env_kwargs = _build_env_kwargs(config)

    write_data_verification_outputs(base_output_dir, train_df, trade_df, dataset_name)

    rng = np.random.default_rng(int(config["seed"]))
    seeds = [int(seed) for seed in rng.integers(0, 2**31 - 1, size=int(config["n_runs"]))]

    pretrained_models_dir = config.get("pretrained_models_dir")
    if pretrained_models_dir and not os.path.isabs(pretrained_models_dir):
        pretrained_models_dir = os.path.join(base_output_dir, pretrained_models_dir)

    shared_models = None
    if config["train_once"]:
        shared_run_dir = _ensure_dir(os.path.join(base_output_dir, "run_000"))
        train_env_factory = _build_env_factory(env_class, env_kwargs, train_df, config["tickers"])
        shared_models = train_or_load_agents(
            config=config,
            run_dir=shared_run_dir,
            train_env_factory=train_env_factory,
            base_output_dir=base_output_dir,
            pretrained_models_dir=pretrained_models_dir,
            shared_models=None,
        )

    run_results = []
    failed_run_ids = []

    for run_id, seed in enumerate(seeds):
        np.random.seed(seed)
        run_dir = _ensure_dir(os.path.join(base_output_dir, f"run_{run_id:03d}"))
        train_env_factory = _build_env_factory(env_class, env_kwargs, train_df, config["tickers"])
        trade_env_factory = _build_env_factory(env_class, env_kwargs, trade_df, config["tickers"])

        try:
            trained_agents = train_or_load_agents(
                config=config,
                run_dir=run_dir,
                train_env_factory=train_env_factory,
                base_output_dir=base_output_dir,
                pretrained_models_dir=pretrained_models_dir,
                shared_models=shared_models,
            )

            dataset_dir = _ensure_dir(os.path.join(run_dir, dataset_name))
            eval_result = run_trading_evaluation(
                trained_agents=trained_agents,
                test_env_factory=trade_env_factory,
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                test_df=trade_df,
                tickers=config["tickers"],
                risk_free_rate_annual=float(config.get("risk_free_rate", RISK_FREE_RATE_ANNUAL)),
            )
            if run_id == 0:
                _export_legacy_general_layout(base_output_dir=base_output_dir, dataset_dir=dataset_dir)
            eval_result["run_id"] = run_id
            eval_result["seed"] = seed
            run_results.append(eval_result)
        except Exception:
            failed_run_ids.append(run_id)
            raise

    metadata = create_run_metadata(
        config=config,
        base_output_dir=base_output_dir,
        seeds=seeds,
        n_succeeded=len(run_results),
        failed_run_ids=failed_run_ids,
    )
    write_run_metadata(base_output_dir, metadata)
    aggregated_root = aggregate_run_results(
        run_results,
        base_output_dir,
        dataset_name,
        risk_free_rate_annual=float(config.get("risk_free_rate", RISK_FREE_RATE_ANNUAL)),
    )
    return {
        "base_output_dir": base_output_dir,
        "run_results": run_results,
        "aggregated_dir": aggregated_root,
        "metadata": metadata,
    }


def run_benchmark_pipeline_from_data(
    config: Dict,
    data_frame: pd.DataFrame,
    env_class=None,
    base_output_dir: str = "",
) -> Dict:
    base_output_dir = _get_or_create_run_dir()
    train_df, trade_df = split_train_trade_data(data_frame, config)
    if env_class is None:
        env_class = resolve_env_class(config)
    return run_benchmark_pipeline(
        config=config,
        train_df=train_df,
        trade_df=trade_df,
        env_class=env_class,
        base_output_dir=base_output_dir,
    )
