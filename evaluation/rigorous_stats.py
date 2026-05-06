import scipy.stats as stats
import numpy as np
import pandas as pd


def compute_benchmark_sr_daily(test_env) -> float:
    """Compute daily Sharpe ratio of an equal-weight buy-and-hold benchmark.

    Uses the close-price matrix stored in TradingEnv (attribute C) to build
    an equal-weight portfolio and compute its daily simple-return SR.
    Returns 0.0 if the benchmark cannot be computed.
    """
    close_matrix = getattr(test_env, "C", None)  # shape (T, n_assets)
    if close_matrix is None or len(close_matrix) < 2:
        return 0.0
    close_matrix = np.asarray(close_matrix, dtype=float)
    first_row = close_matrix[0]
    if np.any(first_row <= 0):
        return 0.0
    normalized = close_matrix / first_row  # (T, n_assets)
    portfolio = normalized.mean(axis=1)     # (T,)
    returns = pd.Series(portfolio).pct_change().dropna().to_numpy()
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=0)) + 1e-8
    return float(np.mean(returns) / std)

def calculate_expected_max_sr(trials_sr: list) -> float:
    """
    Euler-Mascheroni deterministic approximation of the expected maximum SR 
    resulting from N optimization trials.
    """
    if len(trials_sr) == 0:
        return 0.0
    N = len(trials_sr)
    std_sr = np.std(trials_sr) + 1e-8
    gamma = 0.5772156649  # Euler-Mascheroni constant
    expected_max = np.mean(trials_sr) + std_sr * ((1 - gamma) * stats.norm.ppf(1 - 1/N) + gamma * stats.norm.ppf(1 - 1/(N * np.e)))
    return expected_max

def calculate_psr(sr: float, t: int, skewness: float, kurtosis: float, sr_benchmark: float = 0.0) -> float:
    """ 
    Probabilistic Sharpe Ratio (Bailey et al. 2012).
    Estimates probability that observed SR genuinely exceeds a benchmark threshold 
    under non-normality constraints (skew & fat tails).
    """
    if t < 2:
        return 0.5

    sr_diff = sr - sr_benchmark
    variance = (1 - skewness * sr + (kurtosis - 1) / 4 * sr ** 2) / (t - 1)
    variance = max(float(variance), 1e-12)
    denominator = np.sqrt(variance)
    z_stat = sr_diff / (denominator + 1e-8)
    return float(stats.norm.cdf(z_stat))

def calculate_dsr(sr: float, t: int, skewness: float, kurtosis: float, trials_sr: list) -> float:
    """ 
    Deflated Sharpe Ratio (Bailey et al. 2014).
    Calculates the PSR using the Expected Maximum SR of N historical trials 
    as the inflated benchmark to combat Selection Bias under multiple testing.
    """
    expected_max_sr = calculate_expected_max_sr(trials_sr)
    return calculate_psr(sr, t, skewness, kurtosis, sr_benchmark=expected_max_sr)
