"""Tests for evaluation.multiseed – multi-seed DRL runner."""

import json
import os
import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from evaluation.multiseed import (
    AGENT_CLASSES,
    bootstrap_ci,
    compute_aggregate_stats,
    compute_mintrl,
    run_single_seed,
)
from evaluation.experiment_setup import set_global_seed


class TestSetGlobalSeed:
    """Ensures reproducibility primitives work."""

    def test_sets_numpy_seed(self):
        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_values(self):
        set_global_seed(1)
        a = np.random.rand(5)
        set_global_seed(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


class TestBootstrapCI:
    """Tests for bootstrap confidence interval utility."""

    def test_returns_tuple_of_two_floats(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(vals)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi

    def test_single_value_returns_same(self):
        lo, hi = bootstrap_ci([3.0])
        assert lo == hi == 3.0

    def test_empty_returns_nan(self):
        lo, hi = bootstrap_ci([])
        assert np.isnan(lo) and np.isnan(hi)


class TestComputeMinTRL:
    """MinTRL (Minimum Track Record Length) from Bailey & López de Prado."""

    def test_positive_sr_returns_positive(self):
        m = compute_mintrl(sr=1.5, skewness=0.0, kurtosis=3.0)
        assert m > 0

    def test_zero_sr_returns_inf_or_large(self):
        m = compute_mintrl(sr=0.0, skewness=0.0, kurtosis=3.0)
        assert m == float("inf") or m > 1e6

    def test_higher_sr_needs_fewer_observations(self):
        m_low = compute_mintrl(sr=0.5, skewness=0.0, kurtosis=3.0)
        m_high = compute_mintrl(sr=2.0, skewness=0.0, kurtosis=3.0)
        assert m_high < m_low


class TestLoadScenarioConfig:
    """Validates config resolution for each scenario."""

    def _load(self, scenario):
        import yaml
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments", f"neurips2026_{scenario}",
        )
        with open(os.path.join(config_dir, "config.yaml")) as f:
            return yaml.safe_load(f)

    def test_turb_scenario_has_turbulence_200(self):
        cfg = self._load("turb")
        assert cfg["turbulence_threshold"] == 200.0
        assert cfg.get("data_source") == "yahoo"

    def test_noturb_scenario_has_high_threshold(self):
        cfg = self._load("noturb")
        assert float(cfg["turbulence_threshold"]) >= 1e5
        assert cfg.get("data_source") == "yahoo"

    def test_synth_scenario_uses_synthetic_csv(self):
        cfg = self._load("synth")
        assert cfg.get("train_data_source") == "yahoo"
        assert cfg.get("test_data_source") == "synthetic_csv"

    def test_all_scenarios_have_agent_params(self):
        for sc in ("turb", "noturb", "synth"):
            cfg = self._load(sc)
            assert "a2c_params" in cfg
            assert "timesteps_per_model" in cfg


class TestComputeAggregateStats:
    """Validates aggregation of per-seed metrics."""

    def _make_records(self, n=5):
        rng = np.random.default_rng(99)
        return [
            {
                "SR": rng.normal(1.0, 0.2),
                "Sortino": rng.normal(1.5, 0.3),
                "PSR": rng.uniform(0.6, 1.0),
                "DSR": rng.uniform(0.0, 0.5),
                "AR": rng.normal(0.1, 0.05),
                "TotalReturn": rng.normal(0.15, 0.05),
                "MaxDrawdown": rng.uniform(-0.3, -0.05),
                "MinTRL": rng.uniform(10, 100),
            }
            for _ in range(n)
        ]

    def test_returns_dict_with_mean_std_ci(self):
        records = self._make_records()
        agg = compute_aggregate_stats(records)
        assert "SR_mean" in agg
        assert "SR_std" in agg
        assert "SR_ci_lo" in agg
        assert "SR_ci_hi" in agg

    def test_mean_is_correct(self):
        records = self._make_records(3)
        agg = compute_aggregate_stats(records)
        expected_mean = np.mean([r["SR"] for r in records])
        assert abs(agg["SR_mean"] - expected_mean) < 1e-8


class TestAgentsAndScenarios:
    """Constants are correct."""

    def test_agents_list(self):
        assert set(AGENT_CLASSES.keys()) == {"a2c", "ppo", "ddpg", "td3"}


class TestRunSingleSeed:
    """Integration-level test with heavy mocking to avoid real training."""

    @mock.patch("evaluation.multiseed.evaluate_agent")
    @mock.patch("evaluation.multiseed.build_envs")
    def test_returns_metrics_dict(self, mock_build, mock_eval):
        mock_build.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_eval.return_value = {
            "AR": 0.10,
            "TotalReturn": 0.15,
            "SR": 1.2,
            "Sortino": 1.5,
            "PSR": 0.95,
            "DSR": 0.30,
            "CI_Low": 0.8,
            "CI_High": 1.6,
            "training_time_s": 10.0,
            "evaluation_time_s": 2.0,
            "AccountValue": pd.Series([1e6, 1.01e6, 1.02e6]),
            "Transactions": None,
            "AssetWeights": None,
        }
        result = run_single_seed({"a2c_params": {}}, "a2c", seed=42, timesteps=100)
        assert "SR" in result
        assert "seed" in result
        assert result["seed"] == 42
        assert "MaxDrawdown" in result
        assert "MinTRL" in result
