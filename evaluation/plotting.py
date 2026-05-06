import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import List, Optional

def plot_performance(
    model_name: str,
    account_values: pd.Series,
    metrics: dict,
    run_dir: str,
    historical_trials: Optional[List[float]] = None
) -> None:
    """Plots the 4-panel dashboard with account value, drawdown, and Bailey's CI."""
    if account_values.empty:
        return
        
    vals = account_values.values
    dates_pd = pd.to_datetime(account_values.index) if not account_values.index.empty else pd.Series(range(len(vals)))
    
    returns = account_values.pct_change().dropna()
    sk = returns.skew()
    ku = returns.kurtosis()
    sr = metrics.get('SR', 0.0)
    
    # Calculate Bailey's approximated Confidence Interval for SR
    # Variance of SR: (1 + 0.5 * SR^2 - skew * SR + (kurtosis / 4) * SR^2) / (N - 1)
    n_ret = len(returns)
    if n_ret > 1:
        std_sr = np.sqrt((1 + 0.5 * sr**2 - sk * sr + (ku / 4) * sr**2) / (n_ret - 1))
        ci_lower = sr - 1.96 * std_sr
        ci_upper = sr + 1.96 * std_sr
    else:
        ci_lower = ci_upper = sr
    
    rolling_max = pd.Series(vals).cummax()
    drawdown = (pd.Series(vals) - rolling_max) / (rolling_max + 1e-8)
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    axes[0].plot(dates_pd, vals, color='blue')
    axes[0].set_title(f'{model_name} Account Value')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].plot(dates_pd, drawdown, color='red')
    axes[1].fill_between(dates_pd, drawdown, 0, color='red', alpha=0.3) # type: ignore
    axes[1].set_title(f'{model_name} Drawdown')
    axes[1].set_ylabel('Drawdown')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot Sharpe Ratio with Bailey's Confidence Intervals
    axes[2].errorbar(['Sharpe Ratio'], [sr], yerr=[[sr - ci_lower], [ci_upper - sr]], fmt='D', color='green', markersize=8, capsize=8, capthick=2)
    axes[2].axhline(y=np.max(historical_trials) if historical_trials else 0.0, color='gray', linestyle='--', label='Benchmark')
    axes[2].set_title('Sharpe Ratio (95% CI)')
    axes[2].grid(True, axis='y')
    axes[2].legend()
    
    axes[3].axis('off')
    info_text = (
        f"Algorithm: {model_name}\n\n"
        f"Annualized Return: {metrics.get('AR', 0)*100:.2f}%\n"
        f"Max Drawdown: {metrics.get('MaxDrawdown', 0)*100:.2f}%\n"
        f"Sharpe Ratio: {sr:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]\n"
        f"PSR: {metrics.get('PSR', 0):.4f}\n"
        f"DSR: {metrics.get('DSR', 0):.4f}"
    )
    axes[3].text(0.0, 0.5, info_text, fontsize=14, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{model_name}_performance.png"))
    plt.close(fig)
