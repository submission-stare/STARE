from __future__ import annotations

import glob
import json
import os
import re
import shutil
from dataclasses import dataclass, field

import pandas as pd

from utils.data_scope import resolve_data_scope


SUPPLEMENTAL_METADATA_FILENAMES = {
    "compute_log.json",
    "config.yaml",
    "config-turb.yaml",
    "config-noturb.yaml",
    "config-synthetic.yaml",
}


@dataclass(frozen=True)
class AgentDataSource:
    label: str
    source_label: str
    source_dir: str
    transactions_path: str
    snapshots_path: str


@dataclass
class ResultRun:
    input_path: str
    result_root: str
    benchmark_root: str
    run_label: str
    trading_analysis_dir: str
    account_values_path: str | None = None
    agents: list[AgentDataSource] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationIssue:
    input_path: str
    message: str


class DataSourceValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        lines = ["Invalid data source selection:"]
        for issue in issues:
            lines.append(f"- {issue.input_path}: {issue.message}")
        super().__init__("\n".join(lines))


def expand_data_paths(raw_args) -> list[str]:
    """Flatten argparse data-path values and expand quoted shell globs."""
    if raw_args is None:
        return []
    if isinstance(raw_args, str):
        values = [raw_args]
    else:
        values = []
        for item in raw_args:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                values.extend(str(value) for value in item if value is not None)
            else:
                values.append(str(item))

    expanded: list[str] = []
    for value in values:
        path = os.path.expanduser(value)
        if glob.has_magic(path):
            expanded.extend(sorted(glob.glob(path)))
        else:
            expanded.append(path)
    return expanded


def discover_result_run(path: str) -> ResultRun:
    """Resolve a result root or benchmark-data root into a structured run."""
    input_path = os.path.abspath(os.path.expanduser(path))
    benchmark_root = _resolve_benchmark_root(input_path)
    result_root = os.path.dirname(benchmark_root) if os.path.basename(benchmark_root) == "benchmark-data" else input_path
    run_label = _safe_name(os.path.basename(result_root.rstrip(os.sep)) or "run")
    scope = resolve_data_scope(benchmark_root)
    trading_analysis_dir = scope["trading_analysis_dir"]
    account_values_path = os.path.join(scope["trading_dir"], "account_values.csv")
    if not os.path.isfile(account_values_path):
        account_values_path = None

    agents = _discover_agents(trading_analysis_dir, result_root)
    return ResultRun(
        input_path=input_path,
        result_root=result_root,
        benchmark_root=benchmark_root,
        run_label=run_label,
        trading_analysis_dir=trading_analysis_dir,
        account_values_path=account_values_path,
        agents=agents,
    )


def validate_result_runs(runs: list[ResultRun]) -> None:
    """Fail fast when selected runs cannot provide agent transaction data."""
    issues: list[ValidationIssue] = []
    if not runs:
        issues.append(ValidationIssue("<data-path>", "no data paths were provided or matched"))

    for run in runs:
        if not os.path.isdir(run.input_path):
            issues.append(ValidationIssue(run.input_path, "path does not exist or is not a directory"))
            continue
        if _looks_like_result_root_without_benchmark_data(run.input_path):
            issues.append(ValidationIssue(run.input_path, "benchmark-data directory was not found"))
            continue
        if not os.path.isdir(run.benchmark_root):
            issues.append(ValidationIssue(run.input_path, "benchmark-data directory was not found"))
            continue
        if not os.path.isdir(run.trading_analysis_dir):
            issues.append(
                ValidationIssue(
                    run.input_path,
                    f"trading analysis directory was not found at {run.trading_analysis_dir}",
                )
            )
            continue
        if not run.agents:
            partials = _partial_agent_file_issues(run.trading_analysis_dir)
            if partials:
                for partial in partials:
                    issues.append(ValidationIssue(run.input_path, partial))
                continue
            issues.append(
                ValidationIssue(
                    run.input_path,
                    "no agent folders containing both transactions.csv and snapshots.csv were found",
                )
            )

    if issues:
        raise DataSourceValidationError(issues)


def stage_multi_run_dataset(runs: list[ResultRun], output_dir: str) -> str:
    """Copy multiple result runs into a single benchmark-data root."""
    validate_result_runs(runs)

    staged_root = os.path.abspath(output_dir)
    if os.path.exists(staged_root):
        shutil.rmtree(staged_root)

    trading_dir = os.path.join(staged_root, "general")
    staged_analysis_dir = os.path.join(trading_dir, "agents_trading", "trading_analysis")
    metadata_dir = os.path.join(staged_root, "source_runs")
    os.makedirs(staged_analysis_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    used_labels: set[str] = set()
    account_frames: list[pd.DataFrame] = []
    manifest: list[dict] = []

    for run in runs:
        run_meta_dir = os.path.join(metadata_dir, run.run_label)
        os.makedirs(run_meta_dir, exist_ok=True)
        _copy_metadata(run.result_root, run_meta_dir)

        account_df = _read_account_values(run.account_values_path)
        for agent in run.agents:
            label = _dedupe_label(agent.label, run.run_label, used_labels)
            dest_dir = os.path.join(staged_analysis_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(agent.transactions_path, os.path.join(dest_dir, "transactions.csv"))
            shutil.copy2(agent.snapshots_path, os.path.join(dest_dir, "snapshots.csv"))

            agent_account = _extract_agent_account_values(account_df, agent.label, label)
            if agent_account is not None:
                account_frames.append(agent_account)

            manifest.append(
                {
                    "staged_agent": label,
                    "source_agent": agent.label,
                    "source_run": run.run_label,
                    "source_dir": agent.source_dir,
                }
            )

    merged_accounts = _merge_account_values(account_frames)
    if merged_accounts is not None:
        merged_accounts.to_csv(os.path.join(trading_dir, "account_values.csv"), index=False)

    with open(os.path.join(metadata_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return staged_root


def _resolve_benchmark_root(path: str) -> str:
    if os.path.basename(path.rstrip(os.sep)) == "benchmark-data":
        return path
    candidate = os.path.join(path, "benchmark-data")
    if os.path.isdir(candidate):
        return candidate
    return path


def _looks_like_result_root_without_benchmark_data(path: str) -> bool:
    if os.path.basename(path.rstrip(os.sep)) == "benchmark-data":
        return False
    if os.path.isdir(os.path.join(path, "benchmark-data")):
        return False
    scope = resolve_data_scope(path)
    return not os.path.isdir(scope["trading_dir"]) and not os.path.isdir(scope["aggregated_dir"])


def _discover_agents(trading_analysis_dir: str, result_root: str) -> list[AgentDataSource]:
    if not os.path.isdir(trading_analysis_dir):
        return []

    compute_labels = _agent_labels_from_compute_log(result_root)
    agents: list[AgentDataSource] = []
    for root, _dirs, files in os.walk(trading_analysis_dir):
        if "transactions.csv" not in files or "snapshots.csv" not in files:
            continue
        rel = os.path.relpath(root, trading_analysis_dir)
        folder_label = rel.replace(os.sep, "/")
        label = compute_labels[0] if len(compute_labels) == 1 and len(agents) == 0 else folder_label
        agents.append(
            AgentDataSource(
                label=_safe_name(label),
                source_label=folder_label,
                source_dir=root,
                transactions_path=os.path.join(root, "transactions.csv"),
                snapshots_path=os.path.join(root, "snapshots.csv"),
            )
        )
    return agents


def _partial_agent_file_issues(trading_analysis_dir: str) -> list[str]:
    issues: list[str] = []
    if not os.path.isdir(trading_analysis_dir):
        return issues
    for root, _dirs, files in os.walk(trading_analysis_dir):
        has_transactions = "transactions.csv" in files
        has_snapshots = "snapshots.csv" in files
        if has_transactions == has_snapshots:
            continue
        rel = os.path.relpath(root, trading_analysis_dir)
        missing = "snapshots.csv" if has_transactions else "transactions.csv"
        issues.append(f"agent folder {rel} is missing {missing}")
    return issues


def _agent_labels_from_compute_log(result_root: str) -> list[str]:
    path = os.path.join(result_root, "compute_log.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    timings = payload.get("agent_timings")
    if not isinstance(timings, dict):
        return []
    return [_safe_name(str(label)) for label in timings.keys()]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().replace("/", "_"))
    return cleaned.strip("._") or "agent"


def _dedupe_label(label: str, run_label: str, used_labels: set[str]) -> str:
    if label not in used_labels:
        used_labels.add(label)
        return label

    prefixed = f"{run_label}_{label}"
    candidate = prefixed
    counter = 2
    while candidate in used_labels:
        candidate = f"{prefixed}_{counter}"
        counter += 1
    used_labels.add(candidate)
    return candidate


def _copy_metadata(result_root: str, dest_dir: str) -> None:
    for filename in sorted(SUPPLEMENTAL_METADATA_FILENAMES):
        source = os.path.join(result_root, filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(dest_dir, filename))


def _read_account_values(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "date" not in df.columns:
        return None
    return df


def _extract_agent_account_values(
    account_df: pd.DataFrame | None,
    source_label: str,
    staged_label: str,
) -> pd.DataFrame | None:
    if account_df is None:
        return None
    candidates = [source_label, source_label.upper(), source_label.lower(), staged_label]
    source_column = next((column for column in candidates if column in account_df.columns), None)
    if not source_column:
        non_benchmark = [column for column in account_df.columns if column.lower() not in {"date", "benchmark"}]
        if len(non_benchmark) == 1:
            source_column = non_benchmark[0]
    if not source_column:
        return None
    return account_df[["date", source_column]].rename(columns={source_column: staged_label})


def _merge_account_values(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    if not frames:
        return None
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)
