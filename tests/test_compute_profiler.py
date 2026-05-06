"""Tests for evaluation.compute_profiler – hardware info collection and log persistence."""

import json
import os
import time
from unittest import mock

import pytest

from evaluation.compute_profiler import collect_hardware_info, save_compute_log


class TestCollectHardwareInfo:
    """Verifies that hardware/software metadata is captured correctly."""

    def test_returns_dict_with_required_keys(self):
        info = collect_hardware_info()
        required = [
            "platform",
            "python_version",
            "processor",
            "cpu_count_physical",
            "cpu_count_logical",
            "ram_total_gb",
            "cuda_available",
        ]
        for key in required:
            assert key in info, f"Missing key: {key}"

    def test_cpu_counts_are_positive_integers(self):
        info = collect_hardware_info()
        assert isinstance(info["cpu_count_physical"], int)
        assert isinstance(info["cpu_count_logical"], int)
        assert info["cpu_count_logical"] >= 1

    def test_ram_is_positive_float(self):
        info = collect_hardware_info()
        assert isinstance(info["ram_total_gb"], float)
        assert info["ram_total_gb"] > 0.0

    def test_includes_torch_version_when_available(self):
        info = collect_hardware_info()
        # torch is installed in this env, so the key must exist
        assert "torch_version" in info

    def test_includes_sb3_version_when_available(self):
        info = collect_hardware_info()
        assert "stable_baselines3_version" in info

    def test_gpu_fields_present_when_cuda_available(self):
        info = collect_hardware_info()
        if info["cuda_available"]:
            assert "gpu_name" in info
            assert "gpu_count" in info
            assert info["gpu_count"] >= 1
            assert "gpu_memory_gb" in info

    def test_cpu_model_is_populated(self):
        info = collect_hardware_info()
        assert "cpu_model" in info
        assert isinstance(info["cpu_model"], str)


class TestSaveComputeLog:
    """Verifies that compute_log.json is written with correct structure."""

    def _make_results_dict(self):
        """Minimal results_dict with timing keys inserted by evaluate_agent."""
        return {
            "A2C": {
                "AR": 0.10,
                "SR": 1.5,
                "training_time_s": 120.55,
                "evaluation_time_s": 3.21,
            },
            "PPO": {
                "AR": 0.08,
                "SR": 1.2,
                "training_time_s": 180.12,
                "evaluation_time_s": 3.05,
            },
        }

    def test_creates_json_file(self, tmp_path):
        save_compute_log(str(tmp_path), self._make_results_dict())
        assert (tmp_path / "compute_log.json").exists()

    def test_json_is_valid_and_has_sections(self, tmp_path):
        save_compute_log(str(tmp_path), self._make_results_dict())
        with open(tmp_path / "compute_log.json") as f:
            data = json.load(f)
        assert "hardware" in data
        assert "agent_timings" in data
        assert "total_experiment_time_s" in data

    def test_agent_timings_extracted_from_results(self, tmp_path):
        results = self._make_results_dict()
        save_compute_log(str(tmp_path), results)
        with open(tmp_path / "compute_log.json") as f:
            data = json.load(f)
        assert "A2C" in data["agent_timings"]
        assert data["agent_timings"]["A2C"]["training_time_s"] == 120.55
        assert data["agent_timings"]["A2C"]["evaluation_time_s"] == 3.21
        assert "PPO" in data["agent_timings"]

    def test_total_time_is_sum_of_agents(self, tmp_path):
        results = self._make_results_dict()
        save_compute_log(str(tmp_path), results)
        with open(tmp_path / "compute_log.json") as f:
            data = json.load(f)
        expected = 120.55 + 3.21 + 180.12 + 3.05
        assert abs(data["total_experiment_time_s"] - expected) < 0.01

    def test_custom_total_time_overrides_sum(self, tmp_path):
        results = self._make_results_dict()
        save_compute_log(str(tmp_path), results, total_experiment_time_s=999.0)
        with open(tmp_path / "compute_log.json") as f:
            data = json.load(f)
        assert data["total_experiment_time_s"] == 999.0

    def test_handles_agents_without_timing_keys(self, tmp_path):
        results = {
            "OLD_AGENT": {"AR": 0.05, "SR": 0.8},
        }
        save_compute_log(str(tmp_path), results)
        with open(tmp_path / "compute_log.json") as f:
            data = json.load(f)
        # Agent should still appear but with no timing
        assert "OLD_AGENT" not in data["agent_timings"] or data["agent_timings"]["OLD_AGENT"] == {}

    def test_custom_filename(self, tmp_path):
        save_compute_log(str(tmp_path), self._make_results_dict(), filename="my_log.json")
        assert (tmp_path / "my_log.json").exists()
        assert not (tmp_path / "compute_log.json").exists()
