import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import runner
from evaluation.runner import (
    aggregate_run_results,
    evaluate_agent,
    load_pipeline_config,
    resolve_env_class,
    run_benchmark_pipeline_from_data,
    run_trading_evaluation,
    train_or_load_agents,
)
from envs.gym_wrappers.trading_env import TradingEnv


class DummyModel:
    saved_paths = []
    loaded_paths = []

    def __init__(self, *args, **kwargs):
        self.learn_calls = []

    def learn(self, total_timesteps):
        self.learn_calls.append(total_timesteps)

    def predict(self, obs, deterministic=True):
        return [0.6, 0.4], None

    def save(self, path):
        self.saved_paths.append(path)
        save_path = path if str(path).endswith(".zip") else path + ".zip"
        Path(save_path).write_text("dummy-model", encoding="utf-8")

    @classmethod
    def load(cls, path):
        cls.loaded_paths.append(path)
        return cls()


class DummyEvalEnv:
    def __init__(self):
        self.portfolio_value = 1.0
        self.step_idx = 0

    def reset(self):
        self.portfolio_value = 1.0
        self.step_idx = 0
        return [0.0]

    def step(self, action):
        self.step_idx += 1
        self.portfolio_value *= 1.01
        done = self.step_idx > 10
        reward = 1.0
        return [0.0], reward, done, {}


class DummyPortfolioEnv:
    WEIGHT_PATTERN = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.60, 0.25, 0.15], dtype=float),
        np.array([0.55, 0.35, 0.10], dtype=float),
        np.array([0.50, 0.30, 0.20], dtype=float),
        np.array([0.58, 0.22, 0.20], dtype=float),
        np.array([0.52, 0.33, 0.15], dtype=float),
    )
    RETURN_PATTERN = (1.01, 1.015, 0.99, 1.02, 1.005, 0.995)

    def __init__(
        self,
        data,
        tickers,
        window_size=2,
        initial_capital=1000000.0,
        commission=0.0005,
        turbulence_threshold=1e9,
    ):
        self.data = data
        self.tickers = tickers
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.commission = commission
        self.turbulence_threshold = turbulence_threshold
        self.price_frame = (
            data.reset_index()
            .pivot(index="Date", columns="Ticker", values="Close")
            .sort_index()
            .reindex(columns=tickers)
        )
        self.unique_dates = list(self.price_frame.index)
        self.dates = self.unique_dates
        self.current_step = 1
        self.portfolio_value = initial_capital
        self.weights = self._weights_for_step(0)
        self._episode_history = []

    @classmethod
    def _weights_for_step(cls, step_idx):
        return cls.WEIGHT_PATTERN[min(step_idx, len(cls.WEIGHT_PATTERN) - 1)].copy()

    @classmethod
    def _return_factor_for_step(cls, step_idx):
        return float(cls.RETURN_PATTERN[step_idx % len(cls.RETURN_PATTERN)])

    @classmethod
    def expected_step_transaction_costs(cls, n_steps, initial_capital, commission):
        previous_weights = cls._weights_for_step(0)
        portfolio_value = float(initial_capital)
        costs = []
        for step_idx in range(1, n_steps + 1):
            next_weights = cls._weights_for_step(step_idx)
            step_cost = portfolio_value * commission * np.abs(next_weights - previous_weights).sum()
            costs.append(step_cost)
            portfolio_value = (portfolio_value - step_cost) * cls._return_factor_for_step(step_idx - 1)
            previous_weights = next_weights
        return np.asarray(costs, dtype=float)

    def _prices_for_date(self, date_idx):
        row = self.price_frame.iloc[date_idx]
        return {ticker: float(row[ticker]) for ticker in self.tickers}

    def _record_history_row(self, step, date_idx, account_value_before, account_value_after, raw_action, step_transaction_cost):
        total_asset = float(account_value_after)
        cash_weight = float(self.weights[0])
        row = {
            "step": int(step),
            "date": pd.Timestamp(self.unique_dates[date_idx]),
            "account_value_before": float(account_value_before),
            "account_value_after": total_asset,
            "step_transaction_cost": float(step_transaction_cost),
            "cash": float(total_asset * cash_weight),
            "total_allocated": float(total_asset * (1.0 - cash_weight)),
            "total_asset": total_asset,
            "cash_weight": cash_weight,
            "cash_raw_action": float(raw_action[0]),
        }
        date_prices = self._prices_for_date(date_idx)
        for ticker_idx, ticker in enumerate(self.tickers):
            row[f"{ticker}_price"] = date_prices[ticker]
            row[f"{ticker}_weight"] = float(self.weights[ticker_idx + 1])
            row[f"{ticker}_raw_action"] = float(raw_action[ticker_idx + 1])
        self._episode_history.append(row)

    def get_episode_history(self):
        return pd.DataFrame(self._episode_history).copy()

    def reset(self, seed=None, options=None):
        self.current_step = 1
        self.portfolio_value = self.initial_capital
        self.weights = self._weights_for_step(0)
        self._episode_history = []
        self._record_history_row(
            step=0,
            date_idx=0,
            account_value_before=self.initial_capital,
            account_value_after=self.initial_capital,
            raw_action=np.zeros(len(self.tickers) + 1, dtype=float),
            step_transaction_cost=0.0,
        )
        return [0.0], {}

    def step(self, action):
        previous_value = self.portfolio_value
        previous_weights = self.weights.copy()
        next_history_step = len(self._episode_history)
        self.weights = self._weights_for_step(next_history_step)
        step_transaction_cost = previous_value * self.commission * np.abs(self.weights - previous_weights).sum()
        self.portfolio_value = (previous_value - step_transaction_cost) * self._return_factor_for_step(next_history_step - 1)
        self.current_step += 1
        raw_action = np.linspace(-0.2, 0.2, len(self.tickers) + 1, dtype=float) + next_history_step
        self._record_history_row(
            step=next_history_step,
            date_idx=min(self.current_step - 1, len(self.unique_dates) - 1),
            account_value_before=previous_value,
            account_value_after=self.portfolio_value,
            raw_action=raw_action,
            step_transaction_cost=step_transaction_cost,
        )
        done = self.current_step >= len(self.unique_dates)
        return [0.0], 1.0, done, False, {}


def _sample_market_frame():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for date_idx, date in enumerate(dates):
        for ticker_idx, ticker in enumerate(["AAA", "BBB"]):
            price = 10 + date_idx + ticker_idx
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Open": price,
                    "High": price + 1,
                    "Low": price - 1,
                    "Close": price + 0.5,
                    "Volume": 1000 + date_idx,
                    "MACD": 0.1,
                    "RSI": 50.0,
                    "SO": 50.0,
                    "FR": 0.5,
                }
            )
    return pd.DataFrame(rows).set_index("Date")


@pytest.fixture(autouse=True)
def reset_dummy_model_state():
    DummyModel.saved_paths = []
    DummyModel.loaded_paths = []
    yield


def test_evaluate_agent():
    train_env = DummyEvalEnv()
    test_env = DummyEvalEnv()
    mock_model_class = MagicMock(return_value=DummyModel())

    metrics = evaluate_agent(
        model_class=mock_model_class,
        model_name="Test-Mock-Agent",
        train_env=train_env,
        test_env=test_env,
        total_timesteps=100,
    )

    assert "AR" in metrics
    assert "SR" in metrics
    assert "Sortino" in metrics
    assert "PSR" in metrics
    assert "DSR" in metrics
    assert metrics["AR"] > 0
    assert "model_path" in metrics
    assert metrics["model_path"] is not None
    assert isinstance(metrics["PSR"], float)


def test_evaluate_agent_with_pretrained_model():
    """evaluate_agent skips training when pretrained_model is provided."""
    test_env = DummyEvalEnv()
    dummy = DummyModel()

    metrics = evaluate_agent(
        model_class=None,  # not used
        model_name="Pretrained-Agent",
        train_env=None,  # not used
        test_env=test_env,
        total_timesteps=0,  # not used
        pretrained_model=dummy,
    )

    assert dummy.learn_calls == []  # training must be skipped
    assert "AR" in metrics
    assert "SR" in metrics
    assert metrics["model_path"] is None  # no new save for pretrained
    assert metrics["training_time_s"] == 0.0


def test_evaluate_agent_with_pretrained_model_path(tmp_path):
    """evaluate_agent loads a pretrained model when a file path is provided."""
    test_env = DummyEvalEnv()
    pretrained_zip = tmp_path / "A2C_model.zip"
    pretrained_zip.write_text("dummy-model", encoding="utf-8")

    metrics = evaluate_agent(
        model_class=DummyModel,
        model_name="Pretrained-Agent-Path",
        train_env=None,
        test_env=test_env,
        total_timesteps=0,
        pretrained_model=str(pretrained_zip),
    )

    assert DummyModel.loaded_paths == [str(pretrained_zip)]
    assert "AR" in metrics
    assert "SR" in metrics
    assert metrics["model_path"] is None
    assert metrics["training_time_s"] == 0.0


def test_train_or_load_agents_reuses_pretrained_models(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "a2c", DummyModel)

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    (pretrained_dir / "real_a2c.zip").write_text("saved", encoding="utf-8")

    config = {
        "agents": ["a2c"],
        "reuse_existing": True,
        "no_train": False,
        "timesteps_per_model": 5,
        "save_models_dir": None,
    }

    agents = train_or_load_agents(
        config=config,
        run_dir=str(tmp_path / "run_000"),
        train_env_factory=DummyEvalEnv,
        base_output_dir=str(tmp_path),
        pretrained_models_dir=str(pretrained_dir),
    )

    assert "a2c" in agents
    assert DummyModel.loaded_paths == [str(pretrained_dir / "real_a2c")]
    assert DummyModel.saved_paths == []


def test_train_or_load_agents_raises_for_missing_no_train(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "a2c", DummyModel)

    config = {
        "agents": ["a2c"],
        "reuse_existing": False,
        "no_train": True,
        "timesteps_per_model": 5,
        "save_models_dir": None,
    }

    with pytest.raises(FileNotFoundError):
        train_or_load_agents(
            config=config,
            run_dir=str(tmp_path / "run_000"),
            train_env_factory=DummyEvalEnv,
            base_output_dir=str(tmp_path),
            pretrained_models_dir=None,
        )


def test_run_trading_evaluation_writes_expected_outputs(tmp_path):
    dataset_dir = tmp_path / "run_000" / "sample_period"
    dataset_dir.mkdir(parents=True)
    market_df = _sample_market_frame()

    result = run_trading_evaluation(
        trained_agents={"a2c": DummyModel(), "ppo": DummyModel()},
        test_env_factory=lambda: DummyPortfolioEnv(market_df, ["AAA", "BBB"]),
        dataset_dir=str(dataset_dir),
        dataset_name="sample_period",
        test_df=market_df,
        tickers=["AAA", "BBB"],
    )

    assert (dataset_dir / "account_values.csv").exists()
    assert (dataset_dir / "agents_trading" / "account_value_a2c.csv").exists()
    assert (dataset_dir / "agents_trading" / "account_value_ppo.csv").exists()
    assert (dataset_dir / "financial_metrics" / "sharpe_summary_agents.csv").exists()
    assert (dataset_dir / "financial_metrics" / "sharpe_summary_agents_with_psr.csv").exists()
    assert (dataset_dir / "agents_trading" / "trading_analysis" / "a2c" / "transactions.csv").exists()
    assert (dataset_dir / "agents_trading" / "trading_analysis" / "a2c" / "snapshots.csv").exists()
    assert (dataset_dir / "agents_trading" / "trading_analysis" / "_comparison" / "compare_transaction_summary.csv").exists()

    account_values_df = pd.read_csv(dataset_dir / "account_values.csv")
    sharpe_compat_df = pd.read_csv(dataset_dir / "financial_metrics" / "sharpe_summary_agents_with_psr.csv")
    snapshots_df = pd.read_csv(dataset_dir / "agents_trading" / "trading_analysis" / "a2c" / "snapshots.csv")
    transactions_df = pd.read_csv(dataset_dir / "agents_trading" / "trading_analysis" / "a2c" / "transactions.csv")
    comparison_df = pd.read_csv(dataset_dir / "agents_trading" / "trading_analysis" / "_comparison" / "compare_transaction_summary.csv")

    assert list(account_values_df.columns) == ["date", "a2c", "ppo", "benchmark"]
    assert list(sharpe_compat_df.columns) == [
        "model",
        "type",
        "sharpeRatio",
        "sortinoRatio",
        "ciLow",
        "ciHigh",
        "psr",
        "dsr",
        "skew",
        "kurt",
        "volatility",
        "maxDrawdown",
    ]
    assert list(snapshots_df.columns) == [
        "step",
        "date",
        "cash",
        "total_allocated",
        "total_asset",
        "AAA_price",
        "AAA_shares",
        "AAA_value",
        "AAA_weight",
        "BBB_price",
        "BBB_shares",
        "BBB_value",
        "BBB_weight",
        "cash_weight",
    ]
    assert list(transactions_df.columns) == [
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
    ]
    assert list(comparison_df.columns) == [
        "agent",
        "total_buys",
        "total_sells",
        "total_holds",
        "total_value_buy",
        "total_value_sell",
        "total_cost",
    ]

    assert np.allclose(snapshots_df["cash"] + snapshots_df["total_allocated"], snapshots_df["total_asset"])
    assert np.allclose(snapshots_df[["AAA_weight", "BBB_weight", "cash_weight"]].sum(axis=1), 1.0)

    holdings_delta = transactions_df["holdings_after"] - transactions_df["holdings_before"]
    assert ((holdings_delta > 1e-12) == (transactions_df["action_type"] == "BUY")).all()
    assert ((holdings_delta < -1e-12) == (transactions_df["action_type"] == "SELL")).all()
    assert ((holdings_delta.abs() <= 1e-12) == (transactions_df["action_type"] == "HOLD")).all()

    expected_costs = DummyPortfolioEnv.expected_step_transaction_costs(
        n_steps=len(snapshots_df) - 1,
        initial_capital=1_000_000.0,
        commission=0.0005,
    )
    observed_costs = (
        transactions_df.groupby("step")["transaction_cost"].sum().sort_index().to_numpy()
    )
    assert np.allclose(observed_costs, expected_costs[: len(observed_costs)])
    assert result["dataset_name"] == "sample_period"


def test_load_pipeline_config_defaults_trading_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tickers": ["AAA", "BBB"],
                "train_start": "2020-01-01",
                "train_end": "2020-01-03",
                "test_start": "2020-01-04",
                "test_end": "2020-01-06",
            }
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(str(config_path))

    assert config["rl_mode"] == "portfolio"
    assert config["hmax"] == 100
    assert config["reward_scaling"] == 1e-4
    assert config["buy_cost_pct"] == config["commission"]
    assert config["sell_cost_pct"] == config["commission"]
    assert resolve_env_class(config) is runner.PortfolioEnv


def test_run_trading_evaluation_supports_real_trading_mode_outputs(tmp_path):
    dataset_dir = tmp_path / "run_000" / "sample_period"
    dataset_dir.mkdir(parents=True)
    market_df = _sample_market_frame()

    result = run_trading_evaluation(
        trained_agents={"a2c": DummyModel()},
        test_env_factory=lambda: TradingEnv(
            market_df,
            ["AAA", "BBB"],
            initial_capital=1_000_000.0,
            commission=0.0005,
            turbulence_threshold=200.0,
            hmax=100,
            reward_scaling=1e-4,
        ),
        dataset_dir=str(dataset_dir),
        dataset_name="sample_period",
        test_df=market_df,
        tickers=["AAA", "BBB"],
    )

    transactions_df = pd.read_csv(dataset_dir / "agents_trading" / "trading_analysis" / "a2c" / "transactions.csv")
    snapshots_df = pd.read_csv(dataset_dir / "agents_trading" / "trading_analysis" / "a2c" / "snapshots.csv")

    assert result["dataset_name"] == "sample_period"
    assert not transactions_df.empty
    assert np.allclose(transactions_df["shares_traded"], transactions_df["shares_traded"].round())
    assert (transactions_df["shares_traded"] >= 0).all()
    assert np.allclose(snapshots_df["cash"] + snapshots_df["total_allocated"], snapshots_df["total_asset"])


def test_aggregate_run_results_creates_aggregated_dataset(tmp_path):
    run_results = []
    for run_id in range(2):
        run_dir = tmp_path / f"run_{run_id:03d}" / "sample_period"
        run_dir.mkdir(parents=True)
        account_values = pd.DataFrame(
            {
                "a2c": [100.0, 101.0 + run_id, 102.0 + run_id],
                "ppo": [100.0, 100.5 + run_id, 101.0 + run_id],
            },
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        account_values.to_csv(run_dir / "account_values.csv")
        run_results.append(
            {
                "run_id": run_id,
                "seed": 123 + run_id,
                "sharpe": {"a2c": 1.0 + run_id, "ppo": 0.5 + run_id},
                "final_value": {"a2c": 102.0 + run_id, "ppo": 101.0 + run_id},
                "max_drawdown": {"a2c": -0.1, "ppo": -0.2},
                "account_values_csv": str(run_dir / "account_values.csv"),
            }
        )

    aggregated_root = aggregate_run_results(run_results, str(tmp_path), "sample_period")
    aggregated_dir = Path(aggregated_root) / "aggregated"

    assert aggregated_dir.exists()
    assert (aggregated_dir / "all_runs_sharpe.csv").exists()
    assert (aggregated_dir / "all_runs_final_value.csv").exists()
    assert (aggregated_dir / "statistical_summary.csv").exists()
    assert (aggregated_dir / "mean_portfolio_account_values.csv").exists()
    assert (aggregated_dir / "mean_portfolio_metrics.csv").exists()


def test_run_benchmark_pipeline_from_data_writes_run_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "a2c", DummyModel)
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "ppo", DummyModel)
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "ddpg", DummyModel)

    config = {
        "tickers": ["AAA", "BBB"],
        "agents": ["a2c", "ppo", "ddpg"],
        "start_date": "2020-01-01",
        "train_start": "2020-01-01",
        "train_end": "2020-01-03",
        "test_start": "2020-01-04",
        "test_end": "2020-01-06",
        "dataset_name": "sample_period",
        "label": "Sample Period",
        "n_runs": 1,
        "seed": 7,
        "timesteps_per_model": 5,
        "train_once": False,
        "reuse_existing": True,
        "no_train": False,
        "pretrained_models_dir": None,
        "save_models_dir": "saved_models",
        "window_size": 2,
        "initial_capital": 1000000.0,
        "commission": 0.0005,
        "turbulence_threshold": 200.0,
        "risk_free_rate": 0.0,
    }

    result = run_benchmark_pipeline_from_data(
        config=config,
        data_frame=_sample_market_frame(),
        env_class=DummyPortfolioEnv,
        base_output_dir=str(tmp_path / "outputs"),
    )

    base_output = Path(result["base_output_dir"])
    metadata = yaml.safe_load((base_output / "run_metadata.json").read_text(encoding="utf-8"))

    assert (base_output / "run_000" / "sample_period" / "account_values.csv").exists()
    assert (base_output / "benchmark-data" / "general" / "account_values.csv").exists()
    assert (
        base_output
        / "run_000"
        / "sample_period"
        / "agents_trading"
        / "trading_analysis"
        / "a2c"
        / "transactions.csv"
    ).exists()
    assert (
        base_output
        / "benchmark-data"
        / "general"
        / "agents_trading"
        / "trading_analysis"
        / "a2c"
        / "transactions.csv"
    ).exists()
    assert (
        base_output
        / "run_000"
        / "sample_period"
        / "financial_metrics"
        / "sharpe_summary_agents_with_psr.csv"
    ).exists()
    assert (base_output / "run_metadata.json").exists()
    assert metadata["evaluation_datasets"] == {"sample_period": {"label": "Sample Period"}}
    assert (base_output / "data_verification" / "real_train_processed.csv").exists()
    assert (base_output / "aggregated_sample_period" / "aggregated" / "statistical_summary.csv").exists()
    assert (base_output / "saved_models" / "real_a2c.zip").exists()


def test_run_benchmark_pipeline_passes_turbulence_threshold(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "a2c", DummyModel)

    captured_thresholds = []

    class CapturingEnv(DummyPortfolioEnv):
        def __init__(self, *args, **kwargs):
            captured_thresholds.append(kwargs.get("turbulence_threshold"))
            super().__init__(*args, **kwargs)

    config = {
        "tickers": ["AAA", "BBB"],
        "agents": ["a2c"],
        "start_date": "2020-01-01",
        "train_start": "2020-01-01",
        "train_end": "2020-01-03",
        "test_start": "2020-01-04",
        "test_end": "2020-01-06",
        "dataset_name": "sample_period",
        "n_runs": 1,
        "seed": 7,
        "timesteps_per_model": 5,
        "train_once": False,
        "reuse_existing": True,
        "no_train": False,
        "pretrained_models_dir": None,
        "save_models_dir": None,
        "window_size": 2,
        "initial_capital": 1000000.0,
        "commission": 0.0005,
        "turbulence_threshold": 200.0,
        "risk_free_rate": 0.0,
    }

    run_benchmark_pipeline_from_data(
        config=config,
        data_frame=_sample_market_frame(),
        env_class=CapturingEnv,
        base_output_dir=str(tmp_path / "outputs"),
    )

    assert captured_thresholds
    assert set(captured_thresholds) == {200.0}


def test_run_benchmark_pipeline_from_data_resolves_trading_mode(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.RL_AGENT_CLASSES, "a2c", DummyModel)

    config = {
        "tickers": ["AAA", "BBB"],
        "agents": ["a2c"],
        "start_date": "2020-01-01",
        "train_start": "2020-01-01",
        "train_end": "2020-01-03",
        "test_start": "2020-01-04",
        "test_end": "2020-01-06",
        "dataset_name": "sample_period",
        "label": "Sample Period",
        "n_runs": 1,
        "seed": 7,
        "timesteps_per_model": 5,
        "train_once": False,
        "reuse_existing": True,
        "no_train": False,
        "pretrained_models_dir": None,
        "save_models_dir": None,
        "rl_mode": "trading",
        "initial_capital": 1000000.0,
        "commission": 0.0005,
        "turbulence_threshold": 200.0,
        "risk_free_rate": 0.0,
        "hmax": 100,
        "reward_scaling": 1e-4,
    }

    result = run_benchmark_pipeline_from_data(
        config=config,
        data_frame=_sample_market_frame(),
        base_output_dir=str(tmp_path / "outputs"),
    )

    base_output = Path(result["base_output_dir"])
    metadata = yaml.safe_load((base_output / "run_metadata.json").read_text(encoding="utf-8"))
    transactions_df = pd.read_csv(
        base_output
        / "run_000"
        / "sample_period"
        / "agents_trading"
        / "trading_analysis"
        / "a2c"
        / "transactions.csv"
    )

    assert metadata["rl_mode"] == "trading"
    assert np.allclose(transactions_df["shares_traded"], transactions_df["shares_traded"].round())



