import numpy as np

def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    mean_return = np.mean(returns) - risk_free_rate
    std_return = np.std(returns) + 1e-8
    return (mean_return / std_return) * np.sqrt(252)

def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - roll_max) / roll_max
    return np.min(drawdowns)

def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    clean_returns = np.asarray(returns, dtype=float)
    if clean_returns.size == 0:
        return 0.0
    mean_return = np.mean(clean_returns) - risk_free_rate
    downside = np.minimum(clean_returns - risk_free_rate, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 1e-8:
        return 0.0
    return float((mean_return / downside_std) * np.sqrt(252))
