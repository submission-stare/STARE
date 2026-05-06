import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessors.features import (
    TECHNICAL_INDICATOR_COLUMNS,
    calculate_technical_features,
    normalize_technical_indicator_selection,
)
from envs.gym_wrappers.portfolio_env import PortfolioEnv
from envs.gym_wrappers.trading_env import TradingEnv


def _build_raw_market_frame(periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    rows = []
    for ticker_idx, ticker in enumerate(["AAA", "BBB"]):
        base = 100.0 + (ticker_idx * 5.0)
        for day_idx, date in enumerate(dates):
            close = base + day_idx * 0.4 + np.sin(day_idx / 5.0)
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Open": close - 0.6,
                    "High": close + 1.2,
                    "Low": close - 1.4,
                    "Close": close,
                    "Volume": 1_000_000 + 1000 * day_idx + (ticker_idx * 10_000),
                }
            )
    return pd.DataFrame(rows).set_index("Date")


def test_calculate_technical_features_adds_requested_columns_and_legacy_aliases():
    raw = _build_raw_market_frame()

    featured = calculate_technical_features(raw, ["AAA", "BBB"])

    required_columns = {
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
        "atr_14",
        "stoch_k",
        "stoch_d",
        "obv",
        "MACD",
        "RSI",
        "SO",
        "FR",
        "Turbulence",
    }

    assert required_columns.issubset(featured.columns)
    assert set(TECHNICAL_INDICATOR_COLUMNS).issubset(featured.columns)

    tail = featured.groupby("Ticker").tail(20)
    assert np.isfinite(tail[list(required_columns - {"Turbulence"})].to_numpy()).all()

    counts = featured.groupby(level=0)["Ticker"].nunique()
    assert (counts == 2).all()


def test_expanded_feature_frames_work_in_both_envs():
    raw = _build_raw_market_frame()
    featured = calculate_technical_features(raw, ["AAA", "BBB"])

    portfolio_env = PortfolioEnv(featured, ["AAA", "BBB"], window_size=5, initial_capital=1.0)
    trading_env = TradingEnv(featured, ["AAA", "BBB"], initial_capital=1000.0)

    portfolio_state, _ = portfolio_env.reset()
    trading_state, _ = trading_env.reset()

    assert portfolio_state.shape == portfolio_env.observation_space.shape
    assert trading_state.shape == trading_env.observation_space.shape

    next_portfolio_state, _, _, _, _ = portfolio_env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    next_trading_state, _, _, _, _ = trading_env.step(np.array([0.2, -0.1], dtype=np.float32))

    assert next_portfolio_state.shape == portfolio_env.observation_space.shape
    assert next_trading_state.shape == trading_env.observation_space.shape


def test_custom_technical_indicator_subset_changes_state_shape():
    raw = _build_raw_market_frame()
    featured = calculate_technical_features(raw, ["AAA", "BBB"])
    selected = ["MACD", "RSI", "dx_30"]

    portfolio_env = PortfolioEnv(
        featured,
        ["AAA", "BBB"],
        window_size=5,
        initial_capital=1.0,
        technical_indicator_columns=selected,
    )
    trading_env = TradingEnv(
        featured,
        ["AAA", "BBB"],
        initial_capital=1000.0,
        technical_indicator_columns=selected,
    )

    assert portfolio_env.technical_indicator_columns == ["macd", "rsi_30", "dx_30"]
    assert trading_env.technical_indicator_columns == ["macd", "rsi_30", "dx_30"]
    assert portfolio_env.state_dim == 5 * (((5 + 3) * 2) + 2)
    assert trading_env.observation_space.shape == (2 * (3 + 2) + 1,)


def test_normalize_technical_indicator_selection_rejects_unknown_names():
    assert normalize_technical_indicator_selection(None)
    assert normalize_technical_indicator_selection(["MACD", "RSI"]) == ["macd", "rsi_30"]

    try:
        normalize_technical_indicator_selection(["unknown_indicator"])
    except ValueError as exc:
        assert "unknown_indicator" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown indicator selection")