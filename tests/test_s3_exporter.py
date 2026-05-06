import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from unittest.mock import patch, MagicMock

from evaluation.s3_exporter import format_sharpe_summary, format_compare_transactions, format_account_values


def test_format_sharpe_summary():
    results = {
        "A2C": {"SR": 1.52, "Sortino": 1.91, "PSR": 95.0, "DSR": 90.2, "CI_Low": 1.21, "CI_High": 1.83},
        "PPO": {"SR": 1.31, "Sortino": 1.48, "PSR": 88.5, "DSR": 82.1, "CI_Low": 1.01, "CI_High": 1.60}
    }
    df = format_sharpe_summary(results)
    assert len(df) == 2
    assert "model" in df.columns
    assert "sharpeRatio" in df.columns
    assert "sortinoRatio" in df.columns
    assert "benchmarkSharpeRatio" in df.columns
    assert "benchmarkSortinoRatio" in df.columns
    assert "psr" in df.columns
    assert "dsr" in df.columns
    assert "ciLow" in df.columns
    assert "ciHigh" in df.columns
    assert df.loc[0, "model"] == "A2C"
    assert df.loc[0, "sharpeRatio"] == 1.52
    assert df.loc[0, "sortinoRatio"] == 1.91

def test_format_compare_transactions():
    results = {
        "A2C": {"Transactions": pd.DataFrame({"action_type": ["BUY", "SELL", "HOLD", "BUY"]})},
        "PPO": {"Transactions": pd.DataFrame({"action_type": ["HOLD", "HOLD", "HOLD"]})}
    }
    df = format_compare_transactions(results)
    assert len(df) == 2
    assert list(df.columns) == ["agent", "total_buys", "total_sells", "total_holds"]
    assert df.loc[0, "total_buys"] == 2
    assert df.loc[1, "total_buys"] == 0

def test_format_account_values():
    results = {
        "A2C": {"AccountValue": pd.Series([100000, 100100], index=["2020-01-01", "2020-01-02"])},
        "PPO": {"AccountValue": pd.Series([100000, 100200], index=["2020-01-01", "2020-01-02"])}
    }
    df = format_account_values(results, benchmark_series=pd.Series([100000, 100500], index=["2020-01-01", "2020-01-02"], name="DJIA"))
    assert len(df) == 2
    assert "date" in df.columns
    assert "A2C" in df.columns
    assert "PPO" in df.columns
    assert "DJIA" in df.columns
    assert df.loc[df["date"] == "2020-01-02", "A2C"].values[0] == 100100


def test_format_sharpe_summary_recomputes_psr_and_ci_vs_benchmark():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    benchmark_values = pd.Series([100, 101, 103, 104, 106, 108, 109, 111], index=dates, name="benchmark")
    ppo_values = pd.Series([100, 101, 100, 101, 100, 101, 100, 101], index=dates)

    results = {
        "PPO": {
            "SR": 10.0,
            "Sortino": 1.0,
            "PSR": 1.0,
            "DSR": 1.0,
            "CI_Low": -99.0,
            "CI_High": 99.0,
            "AccountValue": ppo_values,
        }
    }

    df = format_sharpe_summary(results, benchmark_series=benchmark_values, risk_free_rate=0.0)

    ppo_row = df.loc[df["model"] == "PPO"].iloc[0]
    benchmark_row = df.loc[df["model"] == "Benchmark"].iloc[0]

    assert 0.0 <= float(ppo_row["psr"]) <= 1.0
    assert float(ppo_row["psr"]) < 0.5
    assert float(ppo_row["ciLow"]) < float(ppo_row["ciHigh"])

    assert float(benchmark_row["psr"]) == pytest.approx(0.5, abs=1e-8)
    assert float(benchmark_row["ciLow"]) < float(benchmark_row["ciHigh"])

