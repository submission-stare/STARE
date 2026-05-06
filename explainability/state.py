from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """
    Represents the state of the LangGraph trading post-mortem agent.
    """
    messages: Annotated[list, operator.add]
    raw_data_path: str
    hypotheses: list[str]  # Overwrite instead of append; can contain up to 20 hypotheses
    generated_code: str
    plot_paths: Annotated[list[str], operator.add]
    final_report: str
    next_node: str
    # ── Investigation tracking ──────────────────────────────────────
    investigation_tests: list[str]  # Feasible tests determined by the investigator
    code_fix_retries: int           # Counter for code-fix retry loops
    # ── Transaction-level data from S3 ──────────────────────────────
    transaction_summary: str   # Concise text summary of transaction data for LLM context
    snapshot_summary: str      # Concise text summary of portfolio snapshots for LLM context
    # ── Consensus Forum answers ─────────────────────────────────────
    consensus_answers: dict    # {"what": ..., "how": ..., "why": ...}
    # ── Code-report analysis branch ────────────────────────────────
    analysis_mode: str
    report_path: str
    benchmark_entry: str
    benchmark_config: str
    code_scope_root: str
    report_claims: list[dict]  # Parsed report claims in code-report mode; live in-flight claims in report mode
    code_dependency_graph: dict
    code_context_bundle: dict
    code_enriched_claims: list[dict]  # Claim-to-code mappings reused by both report and code-report modes
    code_hypotheses: list[str]
    code_hypotheses_data: list[dict]
    code_investigation_tasks: list[dict]
    code_evidence_results: list[dict]  # Hypothesis-level code evidence used inline in report mode and fully in code-report mode
    code_consensus_answers: dict
    code_recommendations: dict
    final_code_report: str
