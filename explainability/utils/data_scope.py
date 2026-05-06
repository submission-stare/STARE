from __future__ import annotations

import os
from pathlib import Path

from utils.config import cfg


def get_data_scope() -> dict[str, str]:
    """Return the configured benchmark subdirectories to inspect."""
    return {
        "run_dir": cfg("data_scope.run_dir", "run_000"),
        "trading_regime_dir": cfg("data_scope.trading_regime_dir", "interleaved"),
        "aggregated_dir": cfg("data_scope.aggregated_dir", "aggregated_interleaved"),
    }


def resolve_data_scope(base_path: str) -> dict[str, str]:
    """Resolve the configured benchmark subdirectories for a given root path."""
    scope = get_data_scope()
    root = _coerce_benchmark_root(os.path.abspath(base_path))
    trading_dir, aggregated_dir, run_dir, run_dir_name, trading_regime_name, aggregated_dir_name = _select_layout(root, scope)
    trading_analysis_dir = os.path.join(trading_dir, "agents_trading", "trading_analysis")
    comparison_dir = os.path.join(trading_analysis_dir, "_comparison")
    aggregated_nested_dir = os.path.join(aggregated_dir, "aggregated")
    aggregated_summary_dir = aggregated_nested_dir if os.path.isdir(aggregated_nested_dir) else aggregated_dir

    return {
        "root": root,
        "run_dir_name": run_dir_name,
        "trading_regime_name": trading_regime_name,
        "aggregated_dir_name": aggregated_dir_name,
        "run_dir": run_dir,
        "trading_dir": trading_dir,
        "aggregated_dir": aggregated_dir,
        "trading_analysis_dir": trading_analysis_dir,
        "comparison_dir": comparison_dir,
        "aggregated_summary_dir": aggregated_summary_dir,
    }


def _coerce_benchmark_root(path: str) -> str:
    """Accept either a benchmark-data folder or its containing result root."""
    if os.path.basename(path.rstrip(os.sep)) == "benchmark-data":
        return path
    benchmark_child = os.path.join(path, "benchmark-data")
    if os.path.isdir(benchmark_child):
        return benchmark_child
    return path


def _select_layout(root: str, scope: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    """Pick the first existing benchmark layout, falling back to configured names."""
    candidates = [
        (scope["run_dir"], scope["trading_regime_dir"], scope["aggregated_dir"]),
        ("run_000", "interleaved", "aggregated_interleaved"),
        ("", "general", "aggregated_general"),
    ]

    for run_name, trading_name, aggregated_name in candidates:
        run_dir = os.path.join(root, run_name) if run_name else root
        if run_name and not os.path.isdir(run_dir):
            continue
        trading_dir = os.path.join(run_dir, trading_name)
        aggregated_dir = os.path.join(root, aggregated_name)
        if os.path.isdir(trading_dir) or os.path.isdir(aggregated_dir):
            return trading_dir, aggregated_dir, run_dir, run_name, trading_name, aggregated_name

    run_dir = os.path.join(root, scope["run_dir"])
    trading_dir = os.path.join(run_dir, scope["trading_regime_dir"])
    aggregated_dir = os.path.join(root, scope["aggregated_dir"])
    return (
        trading_dir,
        aggregated_dir,
        run_dir,
        scope["run_dir"],
        scope["trading_regime_dir"],
        scope["aggregated_dir"],
    )


def scoped_walk_roots(base_path: str) -> list[str]:
    """Return the existing directories that should be scanned for benchmark data."""
    paths = resolve_data_scope(base_path)
    candidates = [
        paths["trading_dir"],
        paths["aggregated_dir"],
    ]
    return [p for p in candidates if os.path.isdir(p)]


def scoped_csv_files(base_path: str) -> list[str]:
    """Return CSV files under the configured trading and aggregated directories only."""
    csvs: list[str] = []
    for root in scoped_walk_roots(base_path):
        for path in sorted(Path(root).rglob("*.csv")):
            csvs.append(str(path))
    return csvs
