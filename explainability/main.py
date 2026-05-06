import os
import argparse
import logging
import subprocess
import re
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from state import AgentState

from nodes.data_analyst import data_analyst
from nodes.hypothesis_forum import hypothesis_forum
from nodes.hypothesis_investigator import hypothesis_investigator
from nodes.code_generator import code_generator
from nodes.code_executor import code_executor
from nodes.consensus_forum import consensus_forum
from nodes.code_report_flow import (
    code_claim_explainer,
    code_consensus_forum,
    code_context_builder,
    code_hypothesis_forum,
    code_hypothesis_investigator,
    code_report_generator,
)
from nodes.report_generator import report_generator
from utils.config import cfg
from utils.data_sources import (
    DataSourceValidationError,
    ValidationIssue,
    discover_result_run,
    expand_data_paths,
    stage_multi_run_dataset,
    validate_result_runs,
)
from utils.data_scope import resolve_data_scope
from utils.step_logger import init_step_logger
import sys

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def route_next_node(state: AgentState):
    """
    Reads the 'next_node' from the state and returns the appropriate edge transition.
    """
    next_node = state.get("next_node")
    if not next_node:
        return "data_analyst" # Fallback
    
    if next_node == "report_generator":
        return "report_generator"
    elif next_node == "consensus_forum":
        return "consensus_forum"
    elif next_node == "hypothesis_maker":
        return "hypothesis_maker"
    elif next_node == "hypothesis_investigator":
        return "hypothesis_investigator"
    elif next_node == "code_generator":
        return "code_generator"
    elif next_node == "code_hypothesis_forum":
        return "code_hypothesis_forum"
    elif next_node == "code_claim_explainer":
        return "code_claim_explainer"
    elif next_node == "code_hypothesis_investigator":
        return "code_hypothesis_investigator"
    elif next_node == "code_consensus_forum":
        return "code_consensus_forum"
    elif next_node == "code_report_generator":
        return "code_report_generator"
    elif next_node == "data_analyst":
        return "data_analyst"
    else:
        return END

def create_agent_graph(analysis_mode: str = "report"):
    if analysis_mode == "both":
        analysis_mode = "report"
    workflow = StateGraph(AgentState)

    if analysis_mode == "code-report":
        workflow.add_node("code_context_builder", code_context_builder)
        workflow.add_node("code_claim_explainer", code_claim_explainer)
        workflow.add_node("code_hypothesis_forum", code_hypothesis_forum)
        workflow.add_node("code_hypothesis_investigator", code_hypothesis_investigator)
        workflow.add_node("code_consensus_forum", code_consensus_forum)
        workflow.add_node("code_report_generator", code_report_generator)

        workflow.set_entry_point("code_context_builder")
        workflow.add_edge("code_context_builder", "code_claim_explainer")
        workflow.add_edge("code_claim_explainer", "code_hypothesis_forum")
        workflow.add_edge("code_hypothesis_forum", "code_hypothesis_investigator")
        workflow.add_edge("code_hypothesis_investigator", "code_consensus_forum")
        workflow.add_edge("code_consensus_forum", "code_report_generator")
        workflow.add_edge("code_report_generator", END)
        return workflow.compile()

    # Report mode graph
    workflow.add_node("code_context_builder", code_context_builder)
    workflow.add_node("code_claim_explainer", code_claim_explainer)
    workflow.add_node("code_hypothesis_investigator", code_hypothesis_investigator)
    workflow.add_node("data_analyst", data_analyst)
    workflow.add_node("hypothesis_maker", hypothesis_forum)
    workflow.add_node("hypothesis_investigator", hypothesis_investigator)
    workflow.add_node("code_generator", code_generator)
    workflow.add_node("code_executor", code_executor)
    workflow.add_node("consensus_forum", consensus_forum)
    workflow.add_node("report_generator", report_generator)

    workflow.set_entry_point("code_context_builder")
    workflow.add_edge("code_context_builder", "data_analyst")
    workflow.add_conditional_edges(
        "data_analyst",
        route_next_node,
        {
            "hypothesis_maker": "hypothesis_maker",
            "hypothesis_investigator": "hypothesis_investigator",
            "code_hypothesis_investigator": "code_hypothesis_investigator",
            "report_generator": "consensus_forum",
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "hypothesis_investigator",
        route_next_node,
        {
            "code_generator": "code_generator",
            "data_analyst": "data_analyst",
        }
    )
    workflow.add_edge("hypothesis_maker", "code_claim_explainer")
    workflow.add_edge("code_claim_explainer", "data_analyst")
    workflow.add_edge("code_generator", "code_executor")
    workflow.add_conditional_edges(
        "code_executor",
        route_next_node,
        {
            "hypothesis_investigator": "hypothesis_investigator",
            "code_generator": "code_generator",
        }
    )
    workflow.add_conditional_edges(
        "code_hypothesis_investigator",
        route_next_node,
        {
            "consensus_forum": "consensus_forum",
        }
    )
    workflow.add_edge("consensus_forum", "report_generator")
    workflow.add_edge("report_generator", END)
    return workflow.compile()


def _normalize_analysis_mode(analysis_mode: str) -> str:
    if analysis_mode == "both":
        logger.warning(
            "analysis-mode=both is deprecated and now runs the integrated single-pass report flow. "
            "Use --analysis-mode report going forward."
        )
        return "report"
    return analysis_mode


def _infer_data_path() -> str | None:
    spec_file = "specification/important_paths.md"
    if not os.path.exists(spec_file):
        return None
    logger.info("No data path provided. Inferring from %s...", spec_file)
    with open(spec_file, "r", encoding="utf-8") as handle:
        content = handle.read()
    match = re.search(r"(s3://[a-zA-Z0-9.\-_/]+)", content)
    return match.group(1) if match else None


def _sync_s3_if_needed(data_path: str) -> str:
    if not data_path.startswith("s3://"):
        return data_path
    local_cache_dir = ".data_cache"
    os.makedirs(local_cache_dir, exist_ok=True)
    logger.info("Syncing S3 data from %s to %s...", data_path, local_cache_dir)
    subprocess.run(["aws", "s3", "sync", data_path, local_cache_dir], check=True)
    logger.info("Successfully synced S3 data.")
    return local_cache_dir


def _prepare_report_data_path(raw_data_args, step_logger) -> str:
    has_explicit_data_args = bool(raw_data_args)
    data_paths = expand_data_paths(raw_data_args)
    if not data_paths and not has_explicit_data_args:
        inferred = _infer_data_path()
        if inferred:
            data_paths = [inferred]

    if not data_paths:
        raise DataSourceValidationError(
            [
                ValidationIssue(
                    "<data-path>",
                    "no data paths were provided or matched"
                    if has_explicit_data_args
                    else "no data path provided and no inferable data source was found",
                )
            ]
        )

    synced_paths = [_sync_s3_if_needed(path) for path in data_paths]
    directory_paths = [path for path in synced_paths if os.path.isdir(path)]
    if not directory_paths:
        return synced_paths[0]

    if len(synced_paths) != len(directory_paths):
        if len(synced_paths) == 1:
            return synced_paths[0]
        raise DataSourceValidationError(
            [
                ValidationIssue(path, "multi-directory analysis only accepts benchmark result directories")
                for path in synced_paths
                if not os.path.isdir(path)
            ]
        )

    runs = [discover_result_run(path) for path in synced_paths]
    validate_result_runs(runs)
    if len(runs) == 1:
        return synced_paths[0]

    staged_path = stage_multi_run_dataset(
        runs,
        os.path.join(step_logger.run_dir, "staged_benchmark_data"),
    )
    logger.info("Staged %s data sources into %s", len(runs), staged_path)
    step_logger.log_step(
        "data_sources",
        {
            "source_count": len(runs),
            "source_paths": synced_paths,
            "staged_path": staged_path,
        },
    )
    return staged_path


def _build_initial_state(
    *,
    analysis_mode: str,
    data_path: str,
    report_path: str,
    benchmark_entry: str,
    benchmark_config: str,
    code_scope_root: str,
) -> AgentState:
    return {
        "messages": [HumanMessage(content=f"Starting {analysis_mode} analysis")],
        "raw_data_path": data_path,
        "hypotheses": [],
        "generated_code": "",
        "plot_paths": [],
        "final_report": "",
        "next_node": "",
        "investigation_tests": [],
        "code_fix_retries": 0,
        "transaction_summary": "",
        "snapshot_summary": "",
        "consensus_answers": {},
        "analysis_mode": analysis_mode,
        "report_path": report_path,
        "benchmark_entry": benchmark_entry,
        "benchmark_config": benchmark_config,
        "code_scope_root": code_scope_root,
        "report_claims": [],
        "code_dependency_graph": {},
        "code_context_bundle": {},
        "code_enriched_claims": [],
        "code_hypotheses": [],
        "code_hypotheses_data": [],
        "code_investigation_tasks": [],
        "code_evidence_results": [],
        "code_consensus_answers": {},
        "code_recommendations": {},
        "final_code_report": "",
    }


def _load_transaction_context(initial_state: AgentState, step_logger) -> None:
    data_path = initial_state["raw_data_path"]
    if not data_path:
        return
    from utils.s3_data_loader import (
        build_snapshot_summary,
        build_transaction_summary,
        generate_check_input_data,
        load_all,
    )

    trading_analysis_dir = resolve_data_scope(data_path)["trading_analysis_dir"]
    if not os.path.isdir(trading_analysis_dir):
        logger.warning("Trading analysis directory not found at %s. Proceeding without transaction data.", trading_analysis_dir)
        return

    logger.info("Loading transaction data from %s...", trading_analysis_dir)
    transactions_df, snapshots_df = load_all(base_dir=trading_analysis_dir)
    if transactions_df.empty:
        logger.warning("No transaction data found in the expected benchmark structure.")
        return

    tx_summary = build_transaction_summary(transactions_df)
    sn_summary = build_snapshot_summary(snapshots_df)
    initial_state["transaction_summary"] = tx_summary
    initial_state["snapshot_summary"] = sn_summary
    check_dir = generate_check_input_data(transactions_df, snapshots_df)
    logger.info("Transaction summary: %s chars, Snapshot summary: %s chars", len(tx_summary), len(sn_summary))
    logger.info("Check plots saved to: %s", check_dir)
    step_logger.log_step(
        "s3_data_loader",
        {
            "transactions_rows": len(transactions_df),
            "snapshots_rows": len(snapshots_df),
            "agents": list(transactions_df["agent"].unique()),
            "check_plots_dir": check_dir,
        },
    )


def _merge_state(current_state: AgentState, node_update: dict) -> None:
    for key, value in node_update.items():
        if key in {"messages", "plot_paths"}:
            current_state[key] = current_state.get(key, []) + value
        else:
            current_state[key] = value


def _run_graph(initial_state: AgentState, analysis_mode: str):
    from langgraph.errors import GraphRecursionError

    graph = create_agent_graph(analysis_mode=analysis_mode)
    logger.info("Starting %s agent graph...", analysis_mode)

    current_state = initial_state.copy()
    try:
        for state in graph.stream(initial_state, {"recursion_limit": cfg("graph.recursion_limit", 50)}):
            node_name = list(state.keys())[0]
            logger.info("--- Node Executed: %s ---", node_name)
            _merge_state(current_state, state[node_name])
    except GraphRecursionError:
        logger.warning("%s graph recursion limit reached; forcing terminal report generation.", analysis_mode)
        if analysis_mode == "code-report":
            _merge_state(current_state, code_report_generator(current_state))
        else:
            _merge_state(current_state, report_generator(current_state))
    return current_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Trading Post-Mortem Agent")
    parser.add_argument(
        "--data-path",
        nargs="+",
        action="append",
        default=None,
        help=(
            "Path(s) to benchmark result folders, benchmark-data folders, or a CSV/JSON file. "
            "Repeat the flag or pass shell-expanded globs for multi-directory analysis."
        ),
    )
    parser.add_argument("--analysis-mode", choices=["report", "code-report", "both"], default="report")
    parser.add_argument("--report-path", default="report.md")
    parser.add_argument("--benchmark-entry", default="")
    parser.add_argument("--benchmark-config", default="")
    parser.add_argument("--code-scope-root", default="")
    parser.add_argument("--clear-report", action="store_true", help="Delete the existing report.md before starting.")
    args = parser.parse_args()
    args.analysis_mode = _normalize_analysis_mode(args.analysis_mode)
    
    if args.clear_report and os.path.exists("report.md"):
        os.remove("report.md")
        logger.info("Cleared existing report.md")

    step_logger = init_step_logger()
    logger.info(f"Logs will be saved to: {step_logger.run_dir}")

    provided_data_paths = expand_data_paths(args.data_path)
    data_path = provided_data_paths[0] if provided_data_paths else None
    if args.analysis_mode == "report":
        try:
            data_path = _prepare_report_data_path(args.data_path, step_logger)
        except subprocess.CalledProcessError as exc:
            logger.error("Error syncing from S3: %s", exc)
            sys.exit(1)
        except DataSourceValidationError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    report_state = None
    if args.analysis_mode == "report":
        report_state = _build_initial_state(
            analysis_mode="report",
            data_path=data_path or "",
            report_path=args.report_path,
            benchmark_entry=args.benchmark_entry,
            benchmark_config=args.benchmark_config,
            code_scope_root=args.code_scope_root,
        )
        report_state["messages"] = [HumanMessage(content=f"Starting analysis on data from {data_path}")]
        _load_transaction_context(report_state, step_logger)
        report_state = _run_graph(report_state, analysis_mode="report")

    if args.analysis_mode == "code-report":
        report_path = args.report_path
        if not os.path.exists(report_path):
            logger.error("code-report mode requires an existing report at %s", report_path)
            sys.exit(1)
        code_state = _build_initial_state(
            analysis_mode="code-report",
            data_path=data_path or "",
            report_path=report_path,
            benchmark_entry=args.benchmark_entry,
            benchmark_config=args.benchmark_config,
            code_scope_root=args.code_scope_root,
        )
        code_state["messages"] = [HumanMessage(content=f"Starting code-grounded analysis from {report_path}")]
        _run_graph(code_state, analysis_mode="code-report")

    logger.info("✅ Analysis Complete! Check the output directory, report.md, and/or code_report.md")
