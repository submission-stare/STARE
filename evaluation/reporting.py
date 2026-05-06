import os
import shutil
import inspect
import datetime
import json
import yaml
import pandas as pd
from typing import Dict, Optional

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

def save_experiment_results(results_dict: Dict, output_filename: str, experiment_title: str, upload_to_s3: bool = False, bucket_name: str = "some-leaderboard", benchmark_series: Optional[pd.Series] = None, risk_free_rate: float = 0.0) -> None:
    """Writes formatted outcomes cleanly to experiment folders and exports to S3."""
    run_dir = _get_or_create_run_dir()
    output_path = os.path.join(run_dir, os.path.basename(output_filename))
    
    with open(output_path, "w") as f:
        f.write(f"Reproduction Results: {experiment_title}\n")
        
        for model_name, metrics in results_dict.items():
            f.write("-" * 60 + "\n")
            f.write(f"{model_name} Annualized Return:  {metrics.get('AR', 0.0)*100:.2f}%\n")
            f.write(f"{model_name} Classical Sharpe:   {metrics.get('SR', 0.0):.2f}\n")
            f.write(f"{model_name} Classical Sortino:  {metrics.get('Sortino', 0.0):.2f}\n")
            f.write(f"{model_name} Probabilistic SR:   {metrics.get('PSR', 0.0):.4f}\n")
            f.write(f"{model_name} Deflated SR (DSR):  {metrics.get('DSR', 0.0):.4f}\n")
            if 'MaxDrawdown' in metrics:
                f.write(f"{model_name} Max Drawdown:       {metrics['MaxDrawdown']*100:.2f}%\n")
                
    print(f"\nResults successfully exported to {output_path}")

    from evaluation.compute_profiler import save_compute_log
    save_compute_log(run_dir, results_dict)

    from evaluation.s3_exporter import prepare_s3_payloads, export_run_to_s3
    prepare_s3_payloads(
        results_dict,
        run_dir,
        benchmark_series=benchmark_series,
        risk_free_rate=risk_free_rate,
    )
    export_run_to_s3(run_dir, upload=upload_to_s3, bucket_name=bucket_name)

def _summarize_compat_transactions(agent_name: str, txn_df: pd.DataFrame) -> Dict[str, float]:
    if txn_df is None or txn_df.empty:
        return {}
    buys = txn_df[txn_df["action_type"] == "BUY"]
    sells = txn_df[txn_df["action_type"] == "SELL"]
    holds = txn_df[txn_df["action_type"] == "HOLD"]
    return {
        "agent": agent_name,
        "total_buys": int(len(buys)),
        "total_sells": int(len(sells)),
        "total_holds": int(len(holds)),
        "total_value_buy": float(buys["gross_value"].sum()) if not buys.empty else 0.0,
        "total_value_sell": float(sells["gross_value"].sum()) if not sells.empty else 0.0,
        "total_cost": float(txn_df["transaction_cost"].sum()) if "transaction_cost" in txn_df.columns else 0.0,
    }
