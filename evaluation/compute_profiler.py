"""Collect hardware/software info and per-agent timing for NeurIPS reproducibility."""

import json
import os
import platform
from typing import Dict, Optional

import psutil


def collect_hardware_info() -> dict:
    """Return a dict describing the current machine and key library versions."""
    info: dict = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # CPU model name (Linux-specific fallback when platform.processor() is vague)
    cpu_model = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info["cpu_model"] = cpu_model

    # GPU / CUDA via PyTorch
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        info["torch_version"] = torch.__version__
    except ImportError:
        info["cuda_available"] = False

    # Stable-Baselines3
    try:
        import stable_baselines3

        info["stable_baselines3_version"] = stable_baselines3.__version__
    except ImportError:
        pass

    return info


def save_compute_log(
    run_dir: str,
    results_dict: Dict,
    total_experiment_time_s: Optional[float] = None,
    filename: str = "compute_log.json",
) -> str:
    """Write ``compute_log.json`` into *run_dir* with hardware info and agent timings.

    Per-agent timing is extracted from the *results_dict* values produced by
    ``evaluate_agent`` (keys ``training_time_s`` and ``evaluation_time_s``).

    Returns the path of the written file.
    """
    agent_timings: Dict[str, dict] = {}
    summed_time = 0.0

    for agent_name, metrics in results_dict.items():
        t_train = metrics.get("training_time_s")
        t_eval = metrics.get("evaluation_time_s")
        if t_train is not None or t_eval is not None:
            entry: dict = {}
            if t_train is not None:
                entry["training_time_s"] = round(t_train, 2)
                summed_time += t_train
            if t_eval is not None:
                entry["evaluation_time_s"] = round(t_eval, 2)
                summed_time += t_eval
            agent_timings[agent_name] = entry

    if total_experiment_time_s is None:
        total_experiment_time_s = round(summed_time, 2)

    payload = {
        "hardware": collect_hardware_info(),
        "agent_timings": agent_timings,
        "total_experiment_time_s": total_experiment_time_s,
    }

    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Compute log saved to {path}")
    return path
