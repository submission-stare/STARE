import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gym_wrappers.portfolio_env import PortfolioEnv


def _build_market_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for d_idx, date in enumerate(dates):
        for t_idx, ticker in enumerate(["AAA", "BBB"]):
            base = 100.0 + d_idx + t_idx
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Open": base,
                    "High": base + 1.0,
                    "Low": base - 1.0,
                    "Close": base + 0.5,
                    "Volume": 1000 + d_idx,
                    "macd": 0.1,
                    "boll_ub": base + 2.0,
                    "boll_lb": base - 2.0,
                    "rsi_30": 50.0,
                    "cci_30": 75.0,
                    "dx_30": 20.0,
                    "close_30_sma": base + 0.25,
                    "close_60_sma": base + 0.1,
                    "atr_14": 1.2,
                    "stoch_k": 40.0,
                    "stoch_d": 35.0,
                    "obv": 5000.0,
                    "MACD": 0.1,
                    "RSI": 50.0,
                    "SO": 40.0,
                    "FR": 0.2,
                }
            )
    return pd.DataFrame(rows).set_index("Date")


def test_portfolio_env_transactions_include_all_tickers():
    data = _build_market_frame()
    env = PortfolioEnv(data=data, tickers=["AAA", "BBB"], window_size=2, initial_capital=1.0)

    env.reset()
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env.step(action)

    tx = env.get_episode_transactions()

    assert not tx.empty
    assert set(tx["ticker"].unique().tolist()) == {"AAA", "BBB"}
    assert {"date", "ticker", "price", "action_type", "shares_traded"}.issubset(tx.columns)
    assert (tx["price"] > 0).all()
