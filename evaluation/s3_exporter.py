import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import boto3
from botocore.exceptions import NoCredentialsError
from typing import Dict, Optional

from evaluation.rigorous_stats import calculate_dsr, calculate_psr


def _annualized_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    mean_excess = clean.mean() - (risk_free_rate / 252.0)
    std = clean.std() + 1e-8
    return float((mean_excess / std) * np.sqrt(252.0))


def _annualized_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    mean_excess = clean.mean() - (risk_free_rate / 252.0)
    downside = clean[clean < 0.0]
    downside_std = downside.std() + 1e-8
    return float((mean_excess / downside_std) * np.sqrt(252.0))


def _daily_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    mean_excess = clean.mean() - (risk_free_rate / 252.0)
    std = clean.std() + 1e-8
    return float(mean_excess / std)


def _bailey_ci_annualized_from_daily_sr(sr_daily: float, t: int, skewness: float, kurtosis: float) -> tuple[float, float]:
    if t < 3:
        sr_annual = float(sr_daily * np.sqrt(252.0))
        return sr_annual, sr_annual

    # Bailey & López de Prado (2012) standard error for Sharpe, on daily scale.
    # kurtosis here is excess kurtosis (pandas default), matching Bailey's γ₄.
    var_daily = (1.0 - skewness * sr_daily + (kurtosis / 4.0) * (sr_daily ** 2)) / max(t - 1, 1)
    var_daily = max(float(var_daily), 1e-12)
    se_daily = float(np.sqrt(var_daily))
    sr_annual = float(sr_daily * np.sqrt(252.0))
    se_annual = float(se_daily * np.sqrt(252.0))
    z = float(stats.norm.ppf(0.975))
    return float(sr_annual - z * se_annual), float(sr_annual + z * se_annual)


def _psr_dsr_vs_benchmark(agent_returns: pd.Series, benchmark_sr: float) -> tuple[float, float]:
    clean = agent_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return 0.0, 0.0

    agent_sr = _annualized_sharpe(clean, risk_free_rate=0.0)
    skewness = float(clean.skew()) if len(clean) > 2 else 0.0
    kurtosis = float(clean.kurtosis()) if len(clean) > 3 else 3.0
    if not np.isfinite(skewness):
        skewness = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 3.0

    t = int(len(clean))
    psr = float(calculate_psr(agent_sr, t, skewness, kurtosis, sr_benchmark=benchmark_sr))
    dsr = float(calculate_dsr(agent_sr, t, skewness, kurtosis, trials_sr=[agent_sr, benchmark_sr]))
    if not np.isfinite(psr):
        psr = 0.0
    if not np.isfinite(dsr):
        dsr = 0.0
    return float(np.clip(psr, 0.0, 1.0)), float(np.clip(dsr, 0.0, 1.0))


def _compute_probabilistic_metrics(
    returns: pd.Series,
    benchmark_sr_annual: float,
    risk_free_rate: float,
) -> tuple[float, float, float, float]:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        sr_annual = _annualized_sharpe(clean, risk_free_rate=risk_free_rate)
        return sr_annual, 0.5, sr_annual, sr_annual

    sr_daily = _daily_sharpe(clean, risk_free_rate=risk_free_rate)
    benchmark_sr_daily = float(benchmark_sr_annual / np.sqrt(252.0))
    skewness = float(clean.skew()) if len(clean) > 2 else 0.0
    kurtosis = float(clean.kurtosis()) if len(clean) > 3 else 3.0
    if not np.isfinite(skewness):
        skewness = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 3.0

    t = int(len(clean))
    psr = float(calculate_psr(sr_daily, t, skewness, kurtosis, sr_benchmark=benchmark_sr_daily))
    if not np.isfinite(psr):
        psr = 0.5
    psr = float(np.clip(psr, 0.0, 1.0))

    ci_low, ci_high = _bailey_ci_annualized_from_daily_sr(sr_daily, t, skewness, kurtosis)
    sr_annual = float(sr_daily * np.sqrt(252.0))
    return sr_annual, psr, ci_low, ci_high


def format_sharpe_summary(
    results_dict: Dict,
    benchmark_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    records = []

    benchmark_sr = 0.0
    benchmark_sortino = 0.0
    if benchmark_series is not None and not benchmark_series.empty:
        benchmark_returns = pd.to_numeric(benchmark_series, errors="coerce").pct_change().dropna()
        benchmark_sr = _annualized_sharpe(benchmark_returns, risk_free_rate=risk_free_rate)
        benchmark_sortino = _annualized_sortino(benchmark_returns, risk_free_rate=risk_free_rate)

    for model, metrics in results_dict.items():
        psr_val = float(metrics.get("PSR", 0.0))
        dsr_val = float(metrics.get("DSR", 0.0))
        ci_low = float(metrics.get("CI_Low", 0.0))
        ci_high = float(metrics.get("CI_High", 0.0))
        sr_val = float(metrics.get("SR", 0.0))
        account_values = metrics.get("AccountValue")
        if account_values is not None and benchmark_series is not None and not benchmark_series.empty:
            agent_returns = pd.to_numeric(account_values, errors="coerce").pct_change().dropna()
            sr_val, psr_val, ci_low, ci_high = _compute_probabilistic_metrics(
                agent_returns,
                benchmark_sr_annual=benchmark_sr,
                risk_free_rate=risk_free_rate,
            )
            _, dsr_val = _psr_dsr_vs_benchmark(agent_returns, benchmark_sr)

        records.append({
            "model": model,
            "sharpeRatio": sr_val,
            "sortinoRatio": metrics.get("Sortino", 0.0),
            "benchmarkSharpeRatio": benchmark_sr,
            "benchmarkSortinoRatio": benchmark_sortino,
            "psr": psr_val,
            "dsr": dsr_val,
            "ciLow": ci_low,
            "ciHigh": ci_high
        })

    if benchmark_series is not None and not benchmark_series.empty:
        benchmark_returns = pd.to_numeric(benchmark_series, errors="coerce").pct_change().dropna()
        benchmark_sr_row, benchmark_psr, benchmark_ci_low, benchmark_ci_high = _compute_probabilistic_metrics(
            benchmark_returns,
            benchmark_sr_annual=benchmark_sr,
            risk_free_rate=risk_free_rate,
        )
        benchmark_dsr = float(
            calculate_dsr(
                benchmark_sr_row / np.sqrt(252.0),
                int(max(len(benchmark_returns), 2)),
                float(benchmark_returns.skew()) if len(benchmark_returns) > 2 else 0.0,
                float(benchmark_returns.kurtosis()) if len(benchmark_returns) > 3 else 3.0,
                [benchmark_sr_row / np.sqrt(252.0)],
            )
        )
        if not np.isfinite(benchmark_dsr):
            benchmark_dsr = 0.5
        benchmark_dsr = float(np.clip(benchmark_dsr, 0.0, 1.0))

        records.append(
            {
                "model": "Benchmark",
                "sharpeRatio": benchmark_sr_row,
                "sortinoRatio": benchmark_sortino,
                "benchmarkSharpeRatio": benchmark_sr,
                "benchmarkSortinoRatio": benchmark_sortino,
                "psr": benchmark_psr,
                "dsr": benchmark_dsr,
                "ciLow": benchmark_ci_low,
                "ciHigh": benchmark_ci_high,
            }
        )

    return pd.DataFrame(records)

def format_compare_transactions(results_dict: Dict) -> pd.DataFrame:
    records = []
    for agent, metrics in results_dict.items():
        txns = metrics.get("Transactions")
        if txns is not None and not txns.empty:
            records.append({
                "agent": agent,
                "total_buys": (txns["action_type"] == "BUY").sum(),
                "total_sells": (txns["action_type"] == "SELL").sum(),
                "total_holds": (txns["action_type"] == "HOLD").sum()
            })
        else:
            records.append({
                "agent": agent, "total_buys": 0, "total_sells": 0, "total_holds": 0
            })
    return pd.DataFrame(records)

def format_account_values(results_dict: Dict, benchmark_series: Optional[pd.Series] = None) -> pd.DataFrame:
    df_list = []
    for model, metrics in results_dict.items():
        acc = metrics.get("AccountValue")
        if acc is not None:
            acc_df = acc.to_frame(name=model)
            acc_df.index.name = "date"
            df_list.append(acc_df)
    
    if df_list:
        combined = pd.concat(df_list, axis=1)
        if benchmark_series is not None:
            bench_df = benchmark_series.to_frame(name=benchmark_series.name or "Benchmark")
            combined = combined.join(bench_df, how="left")
            
        combined = combined.reset_index()
        combined["date"] = combined["date"].astype(str)
        return combined
    return pd.DataFrame()

def prepare_s3_payloads(
    results_dict: Dict,
    run_dir: str,
    benchmark_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
):
    gen_path = os.path.join(run_dir, "benchmark-data", "general")
    agg_path = os.path.join(run_dir, "benchmark-data", "aggregated_general", "stock_analysis", "02_consensus")
    
    fin_mets = os.path.join(gen_path, "financial_metrics")
    comp_txns = os.path.join(gen_path, "agents_trading", "trading_analysis", "_comparison")
    
    os.makedirs(fin_mets, exist_ok=True)
    os.makedirs(comp_txns, exist_ok=True)
    os.makedirs(agg_path, exist_ok=True)
    
    df_sharpe = format_sharpe_summary(
        results_dict,
        benchmark_series=benchmark_series,
        risk_free_rate=risk_free_rate,
    )
    df_sharpe.to_csv(os.path.join(fin_mets, "sharpe_summary_agents_with_psr.csv"), index=False)
    
    df_comp_txn = format_compare_transactions(results_dict)
    df_comp_txn.to_csv(os.path.join(comp_txns, "compare_transaction_summary.csv"), index=False)
    
    df_acc = format_account_values(results_dict, benchmark_series=benchmark_series)
    if not df_acc.empty:
        df_acc.to_csv(os.path.join(gen_path, "account_values.csv"), index=False)
        
    all_final_weights = []
    all_agent_means = []
    
    for model, metrics in results_dict.items():
        agent_dir = os.path.join(gen_path, "agents_trading", "trading_analysis", model)
        os.makedirs(agent_dir, exist_ok=True)
        
        txns = metrics.get("Transactions")
        if txns is None or txns.empty:
            pd.DataFrame(columns=["date", "ticker", "price", "action_type", "shares_traded"]).to_csv(os.path.join(agent_dir, "transactions.csv"), index=False)
        else:
            txns = txns.copy()
            if txns.index.name != "date" and "date" not in txns.columns:
                txns.index.name = "date"
            txns = txns.reset_index()
            txns["date"] = txns["date"].astype(str)
            expected_cols = ["date", "ticker", "price", "action_type", "shares_traded"]
            existing_cols = [c for c in expected_cols if c in txns.columns]
            if not existing_cols:
                pd.DataFrame(columns=expected_cols).to_csv(os.path.join(agent_dir, "transactions.csv"), index=False)
            else:
                txns[existing_cols].to_csv(os.path.join(agent_dir, "transactions.csv"), index=False)
            
        snaps = metrics.get("AssetWeights")
        if snaps is None or snaps.empty:
            pd.DataFrame(columns=["date", "cash_weight"]).to_csv(os.path.join(agent_dir, "snapshots.csv"), index=False)
        else:
            snaps = snaps.copy()
            if snaps.index.name != "date" and "date" not in snaps.columns:
                snaps.index.name = "date"
            snaps = snaps.reset_index()
            snaps["date"] = snaps["date"].astype(str)
            # Filter only date and _weight columns
            weight_cols = [c for c in snaps.columns if c.endswith('_weight')]
            if "cash_weight" not in weight_cols and "cash" in snaps.columns:
                # Fallback if cash weight was omitted somehow
                snaps["cash_weight"] = snaps["cash"] / snaps["total_asset"]
                weight_cols.append("cash_weight")
                
            cols_to_keep = ["date"] + weight_cols
            snaps = snaps[cols_to_keep].copy()
            
            snaps.to_csv(os.path.join(agent_dir, "snapshots.csv"), index=False)
            
            asset_cols = [c for c in snaps.columns if c != "date"]
            if not snaps.empty:
                final_row = snaps.iloc[-1].copy()
                mean_row = snaps[asset_cols].mean()
                for c in asset_cols:
                    asset_name = c.replace("_weight", "")
                    all_agent_means.append({"ticker": asset_name, "agent_key": model, "mean_weight": float(mean_row[c])})
                rec = {"run": 0, "agent_key": model}
                for c in asset_cols:
                    rec[c.replace("_weight", "")] = float(final_row[c])
                all_final_weights.append(rec)

    if all_agent_means:
        df_consensus_agent = pd.DataFrame(all_agent_means)
        df_consensus_agent.to_csv(os.path.join(agg_path, "allocation_consensus_by_agent.csv"), index=False)
        df_overall = df_consensus_agent.groupby("ticker")["mean_weight"].agg(['mean', 'std']).reset_index()
        df_overall = df_overall.rename(columns={"mean": "mean_weight", "std": "std_weight"})
        df_overall["std_weight"] = df_overall["std_weight"].fillna(0.0)
        df_overall["conviction_score"] = 1.0 - df_overall["std_weight"]
        df_overall = df_overall[["ticker", "mean_weight", "std_weight", "conviction_score"]]
        df_overall.to_csv(os.path.join(agg_path, "allocation_consensus_overall.csv"), index=False)
    else:
        pd.DataFrame(columns=["ticker", "agent_key", "mean_weight"]).to_csv(os.path.join(agg_path, "allocation_consensus_by_agent.csv"), index=False)
        pd.DataFrame(columns=["ticker", "mean_weight", "std_weight", "conviction_score"]).to_csv(os.path.join(agg_path, "allocation_consensus_overall.csv"), index=False)

    if all_final_weights:
        df_final_weights = pd.DataFrame(all_final_weights)
        df_final_weights.to_csv(os.path.join(agg_path, "final_weights_all_runs.csv"), index=False)
    else:
        pd.DataFrame(columns=["run", "agent_key", "cash"]).to_csv(os.path.join(agg_path, "final_weights_all_runs.csv"), index=False)


def export_run_to_s3(run_dir: str, upload: bool = True, bucket_name: str = "some-leaderboard"):
    if not upload:
        return
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    s3_client = boto3.client('s3')
    base_upload_dir = os.path.join(run_dir, "benchmark-data")
    
    if not os.path.exists(base_upload_dir):
        print("No benchmark-data found to upload to S3.")
        return
        
    print(f"\nUploading data to s3://{bucket_name}/datasource-website/benchmark-data/ ...")
    
    upload_count = 0
    for root, _, files in os.walk(base_upload_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, base_upload_dir)
            s3_key = f"datasource-website/benchmark-data/{relative_path}".replace("\\", "/")
            
            try:
                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded S3: {s3_key}")
                upload_count += 1
            except NoCredentialsError:
                print("\n  [S3 Error]: AWS credentials not found!")
                continue
            except Exception as e:
                print(f"\n  [S3 Error]: Failed to upload {file}: {str(e)}")
                continue
    print(f"Total files uploaded: {upload_count}")

