import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import sys
import inspect
import datetime
import shutil
import time
import matplotlib.pyplot as plt
import torch

from envs.gym_wrappers.portfolio_env import PortfolioEnv

try:
    from envs.gym_wrappers.stock_trading_env import StockTradingEnv as TradingEnv
except ImportError:
    TradingEnv = None

# Ensure local paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.rigorous_stats import calculate_psr, calculate_dsr, compute_benchmark_sr_daily
from evaluation.metrics import calculate_sortino

_RUN_DIR = None

def _get_or_create_run_dir():
    global _RUN_DIR
    if _RUN_DIR is not None:
        return _RUN_DIR

    frames = inspect.stack()
    caller_dir = os.getcwd()

    # Prefer the experiment entrypoint to avoid internal callers (for example
    # agents/llms) creating results in the wrong subtree.
    for frame_info in frames:
        filename = os.path.abspath(frame_info.filename)
        if (
            os.sep + "experiments" + os.sep in filename
            and "site-packages" not in filename
        ):
            caller_dir = os.path.dirname(filename)
            break
    else:
        for frame_info in frames:
            filename = os.path.abspath(frame_info.filename)
            if "site-packages" in filename:
                continue
            if os.sep + "evaluation" + os.sep in filename:
                continue
            if os.sep + "agents" + os.sep + "llms" + os.sep in filename:
                continue
            caller_dir = os.path.dirname(filename)
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

def evaluate_agent(model_class, model_name, train_env, test_env, total_timesteps, model_kwargs=None, policy_kwargs=None, historical_trials=None, risk_free_rate=0.0, pretrained_model=None, trials=1):
    """
    Unified training and evaluation pipeline measuring classical metrics (AR, SR)
    alongside Bailey's probabilistic metrics (PSR, DSR) evaluating tail risks.

    If *pretrained_model* is provided, training is skipped entirely and the
    given model is evaluated directly on *test_env*. The argument can be an
    already-instantiated model object or a filesystem path to a persisted
    model (SB3 .zip or pickled .pkl).
    """
    run_dir = _get_or_create_run_dir()

    _model_path = None

    if pretrained_model is not None:
        # ---- eval-only path: skip training completely ----
        if isinstance(pretrained_model, (str, os.PathLike)):
            pretrained_path = str(pretrained_model)
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained model path not found: {pretrained_path}")
            if pretrained_path.endswith(".pkl"):
                import pickle as _pkl
                with open(pretrained_path, "rb") as _mf:
                    model = _pkl.load(_mf)
            else:
                if model_class is None or not hasattr(model_class, "load"):
                    raise ValueError(
                        "pretrained_model is a path, but model_class has no load() method."
                    )
                model = model_class.load(pretrained_path)
        else:
            model = pretrained_model
        _training_time_s = 0.0
        print(f"\n--- Using pretrained model for {model_name} (skipping training) ---")
    else:
        # ---- standard path: train from scratch ----
        print(f"\n--- Training {model_name} ---")
        train_env.reset()
        
        kwargs = {"policy": "MlpPolicy", "env": train_env, "verbose": 0, "learning_rate": 3e-4}
        if model_kwargs:
            kwargs.update(model_kwargs)
        if policy_kwargs:
            kwargs["policy_kwargs"] = policy_kwargs
            
        # Optional safety bypass flag used only by this project wrapper.
        # Removed from kwargs so it is never forwarded to SB3 model constructors.
        allow_unsafe_cuda_onpolicy_mlp = bool(kwargs.pop("allow_unsafe_cuda_onpolicy_mlp", False))

        # Smart device selection:
        # - CNN policy + CUDA: use GPU
        # - Off-policy + MLP + CUDA: use GPU
        # - On-policy + MLP: use CPU (SB3 #1245)
        policy = kwargs.get("policy", "MlpPolicy")
        is_cnn = (isinstance(policy, str) and "Cnn" in policy) or (
            not isinstance(policy, str) and hasattr(policy, '__name__') and "Cnn" in policy.__name__
        )

        from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
        is_model_class = isinstance(model_class, type)
        is_on_policy = is_model_class and issubclass(model_class, OnPolicyAlgorithm)
        is_off_policy = is_model_class and issubclass(model_class, OffPolicyAlgorithm)

        if "device" not in kwargs:
            if is_cnn and torch.cuda.is_available():
                kwargs["device"] = "cuda"
            elif torch.cuda.is_available() and is_off_policy:
                kwargs["device"] = "cuda"
            else:
                kwargs["device"] = "cpu"
            print(f"  -> Auto-selected device: {kwargs['device']} (policy={policy})")
        else:
            explicit_device = str(kwargs["device"]).lower()
            if explicit_device.startswith("cuda") and is_on_policy and not is_cnn and not allow_unsafe_cuda_onpolicy_mlp:
                print("  -> Explicit CUDA ignored for on-policy + MLP (SB3 recommends CPU). Using device=cpu.")
                print("  -> To force CUDA anyway, set allow_unsafe_cuda_onpolicy_mlp=true in model_kwargs.")
                kwargs["device"] = "cpu"
            else:
                print(f"  -> Using explicit device: {kwargs['device']}")

        _t_train_start = time.perf_counter()
        model = model_class(**kwargs)
        model.learn(total_timesteps=total_timesteps)
        _t_train_end = time.perf_counter()
        _training_time_s = round(_t_train_end - _t_train_start, 2)

        # Persist trained model (SB3 .zip or generic pickle).
        try:
            if hasattr(model, "save"):
                # SB3 models have a native .save() that produces a .zip
                _model_path = os.path.join(run_dir, f"{model_name}_model.zip")
                model.save(_model_path)
                print(f"  -> Saved model: {_model_path}")
            else:
                # Fallback for non-SB3 models (e.g. LLM wrappers): pickle
                import pickle as _pkl
                _model_path = os.path.join(run_dir, f"{model_name}_model.pkl")
                with open(_model_path, "wb") as _mf:
                    _pkl.dump(model, _mf)
                print(f"  -> Saved model (pickle): {_model_path}")
        except Exception as _save_err:
            print(f"  -> WARNING: could not save model: {_save_err}")

    # Keep model-side environment context aligned with evaluation period.
    # Custom agents (for example LLM wrappers) read day/transactions from
    # model.env inside predict(), so they must point to test_env here.
    if hasattr(model, "set_env"):
        try:
            model.set_env(test_env)
        except Exception:
            if hasattr(model, "env"):
                model.env = test_env
    elif hasattr(model, "env"):
        model.env = test_env
    
    print(f"--- Evaluating {model_name} ---")
    _t_eval_start = time.perf_counter()

    # Check if reset returns a tuple (Gymnasium) or just obs (old Gym)
    reset_res = test_env.reset()
    if isinstance(reset_res, tuple):
        obs = reset_res[0]
    else:
        obs = reset_res
        
    done = False
    vals = []
    dates = []
    env_dates = getattr(test_env, "dates", None)
    fallback_step_idx = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if env_dates is not None and len(env_dates) > 0 and hasattr(test_env, "current_step"):
            safe_idx = max(min(int(getattr(test_env, "current_step", 0)), len(env_dates) - 1), 0)
            current_date = env_dates[safe_idx]
        else:
            current_date = pd.Timestamp("1970-01-01") + pd.Timedelta(days=fallback_step_idx)
        
        step_res = test_env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            obs, reward, done, _ = step_res
            
        vals.append(test_env.portfolio_value)
        dates.append(current_date)
        fallback_step_idx += 1
        
    final_val = float(test_env.portfolio_value)
    initial_val = float(getattr(test_env, 'initial_capital', 1.0))
    growth_ratio = (final_val / initial_val) if initial_val > 0 else 1.0
    total_return = growth_ratio - 1.0
    n_steps = max(len(vals), 1)
    # Annualized return on trading-day basis.
    ar = (growth_ratio ** (252.0 / n_steps)) - 1.0
    
    returns = pd.Series(vals).pct_change().dropna()
    sk = returns.skew()
    ku = returns.kurtosis()
    mean_ret = returns.mean() - (risk_free_rate / 252)
    std_ret = returns.std() + 1e-8
    sr_daily = mean_ret / std_ret
    sr = sr_daily * np.sqrt(252)
    sortino = calculate_sortino(returns.to_numpy(), risk_free_rate=(risk_free_rate / 252))
    
    # Probabilistic limits
    # DSR deflation: generate N synthetic trial SRs to penalize for multiple testing.
    # N = trials (from config, default 1 = no deflation).
    if not historical_trials:
        if trials <= 1:
            historical_trials = [sr_daily]
        else:
            rng = np.random.RandomState(42)
            historical_trials = (sr_daily * rng.uniform(0.5, 1.2, size=trials)).tolist()
    else:
        # Convert annualized historical Sharpe assumptions into daily Sharpe scale.
        historical_trials = [float(x) / np.sqrt(252.0) for x in historical_trials]

    # PSR benchmark: use buy-and-hold equal-weight SR from test_env
    _sr_benchmark_daily = compute_benchmark_sr_daily(test_env)
    psr = calculate_psr(sr_daily, len(returns), sk, ku, sr_benchmark=_sr_benchmark_daily)
    dsr = calculate_dsr(sr_daily, len(returns), sk, ku, historical_trials)
    
    # Bailey & López de Prado (2012) Confidence Interval for SR (daily scale)
    # Var(SR) = (1 - γ₃·SR + γ₄/4·SR²) / (T-1), where γ₄ is excess kurtosis
    std_sr_daily = np.sqrt(max((1 - sk * sr_daily + (ku / 4) * sr_daily**2) / (len(returns) - 1), 1e-12))
    ci_lower = sr - 1.96 * (std_sr_daily * np.sqrt(252.0))
    ci_upper = sr + 1.96 * (std_sr_daily * np.sqrt(252.0))
    
    rolling_max = pd.Series(vals).cummax()
    drawdown = (pd.Series(vals) - rolling_max) / rolling_max
    
    dates_pd = pd.to_datetime(dates)
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    axes[0].plot(dates_pd, vals, color='blue')
    axes[0].set_title(f'{model_name} Account Value Over Time')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].plot(dates_pd, drawdown, color='red')
    axes[1].fill_between(dates_pd, drawdown, 0, color='red', alpha=0.3)
    axes[1].set_title(f'{model_name} Drawdown Over Time')
    axes[1].set_ylabel('Drawdown')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot Sharpe Ratio with Bailey's Confidence Intervals
    axes[2].errorbar(['Sharpe Ratio'], [sr], yerr=[[sr - ci_lower], [ci_upper - sr]], fmt='D', color='green', markersize=8, capsize=8, capthick=2)
    axes[2].axhline(y=np.max(historical_trials) if historical_trials else 0.0, color='gray', linestyle='--', label='Benchmark')
    axes[2].set_title(f'Sharpe Ratio (95% CI)')
    axes[2].grid(True, axis='y')
    axes[2].legend()
    
    axes[3].axis('off')
    info_text = (
        f"Algorithm: {model_name}\n\n"
        f"Annualized Return: {ar*100:.2f}%\n"
        f"Annualized STD: {std_ret * np.sqrt(252) * 100:.2f}%\n"
        f"Sharpe Ratio: {sr:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]\n"
        f"Sortino Ratio: {sortino:.2f}\n"
        f"PSR: {psr:.4f}\n"
        f"DSR: {dsr:.4f}"
    )
    axes[3].text(0.0, 0.5, info_text, fontsize=14, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{model_name}_performance.png"))
    plt.close(fig)
    
    print(
        f"{model_name} Final Value: {final_val:,.2f} | Total Return: {total_return*100:.2f}% | "
        f"AR (annualized): {ar*100:.2f}%, SR: {sr:.2f}, Sortino: {sortino:.2f}"
    )
    _t_eval_end = time.perf_counter()
    _evaluation_time_s = round(_t_eval_end - _t_eval_start, 2)

    print(f"{model_name} PSR (vs Benchmark SR_daily={_sr_benchmark_daily:.4f}): {psr:.4f}  |  DSR: {dsr:.4f}")
    print(f"{model_name} Training: {_training_time_s:.1f}s  |  Evaluation: {_evaluation_time_s:.1f}s")
    
    return {
        "AR": ar,
        "TotalReturn": float(total_return),
        "SR": sr, 
        "Sortino": sortino,
        "PSR": psr, 
        "DSR": dsr,
        "CI_Low": ci_lower,
        "CI_High": ci_upper,
        "training_time_s": _training_time_s,
        "evaluation_time_s": _evaluation_time_s,
        "model_path": _model_path,
        "AccountValue": pd.Series(vals, index=pd.to_datetime(dates)).astype(float),
        "Transactions": getattr(test_env, 'get_episode_transactions', lambda: None)(),
        "AssetWeights": getattr(test_env, 'get_episode_snapshots', lambda: None)()
    }

def save_experiment_results(results_dict, output_path, experiment_title):
    """ Writes formatted outcomes cleanly to experiment folders. """
    run_dir = _get_or_create_run_dir()
    final_output_path = os.path.join(run_dir, os.path.basename(output_path))
    
    with open(final_output_path, "w") as f:
        f.write(f"Reproduction Results: {experiment_title}\n")
        
        for model_name, metrics in results_dict.items():
            f.write("-" * 60 + "\n")
            f.write(f"{model_name} Annualized Return:  {metrics['AR']*100:.2f}%\n")
            f.write(f"{model_name} Classical Sharpe:   {metrics['SR']:.2f}\n")
            f.write(f"{model_name} Classical Sortino:  {metrics.get('Sortino', 0.0):.2f}\n")
            f.write(f"{model_name} Probabilistic SR:   {metrics['PSR']:.4f}\n")
            f.write(f"{model_name} Deflated SR (DSR):  {metrics['DSR']:.4f}\n")
            
    print(f"\nResults successfully exported to {final_output_path}")

    from evaluation.compute_profiler import save_compute_log
    save_compute_log(run_dir, results_dict)

# --- Merged from feat_trading_leaderboard ---
from typing import Dict, List, Optional, Tuple, Callable

SUPPORTED_RL_MODES = {"portfolio", "trading"}
RL_AGENT_CLASSES = {}
try:
    from stable_baselines3 import A2C, PPO, DDPG
    RL_AGENT_CLASSES = {
        "a2c": A2C,
        "ppo": PPO,
        "ddpg": DDPG,
    }
except ImportError:
    pass

def resolve_env_class(config: dict):
    mode = str(config.get("rl_mode", "portfolio")).strip().lower()
    return PortfolioEnv if mode == "portfolio" else TradingEnv


from evaluation import pipeline as _pipeline  # noqa: E402
from evaluation.pipeline import (  # noqa: E402
    aggregate_run_results,
    load_pipeline_config,
    run_benchmark_pipeline_from_data,
    run_trading_evaluation,
    train_or_load_agents,
)

RL_AGENT_CLASSES = _pipeline.RL_AGENT_CLASSES


__all__ = [
    "aggregate_run_results",
    "evaluate_agent",
    "load_pipeline_config",
    "resolve_env_class",
    "run_benchmark_pipeline_from_data",
    "run_trading_evaluation",
    "train_or_load_agents",
]

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
        "total_cost": float(txn_df["transaction_cost"].sum()) if "transaction_cost" in txn_df else 0.0,
    }
