from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from state import AgentState
from utils.config import cfg
from utils.code_report import summarize_enriched_claims
from utils.data_scope import scoped_csv_files
from utils.llm import get_llm
from utils.step_logger import get_step_logger
from langchain_core.output_parsers import PydanticOutputParser
import os
import logging

logger = logging.getLogger(__name__)

# Track how many times the investigator has been called to prevent infinite loops
_investigator_call_count = 0
_MAX_INVESTIGATOR_CALLS: int = cfg("investigator.max_calls", 3)
_MIN_PLOTS: int = cfg("investigator.min_plots", 3)


class FeasibleTest(BaseModel):
    test_description: str = Field(description="Clear description of the test to perform.")
    hypothesis_index: int = Field(description="1-based index of the hypothesis this test validates.")
    data_files_needed: list[str] = Field(description="List of data file paths required for this test.")
    feasible: bool = Field(description="Whether this test can be performed with the available data.")
    reason: str = Field(description="Why this test is feasible or not.")


class InvestigationPlan(BaseModel):
    feasible_tests: list[FeasibleTest] = Field(description="List of tests evaluated for feasibility.")
    next_node: str = Field(description="The next node to execute. Must be 'code_generator' or 'data_analyst'.")
    reasoning: str = Field(description="Brief reasoning for the overall decision.")


def _build_data_inventory(data_path: str) -> str:
    """Build a summary of available data files in the data cache directory."""
    abs_path = os.path.abspath(data_path)
    if not os.path.isdir(abs_path):
        return f"Data directory not found: {abs_path}\n"

    csvs = scoped_csv_files(abs_path)
    if not csvs:
        return f"No scoped CSV files found under {abs_path}\n"

    lines = [f"Data root: {abs_path}", f"CSV files ({len(csvs)} total):"]
    for c in csvs:
        rel = os.path.relpath(c, abs_path)
        # Read header for schema info
        try:
            with open(c) as f:
                header = f.readline().rstrip()
            lines.append(f"  - {rel}  [columns: {header}]")
        except Exception:
            lines.append(f"  - {rel}")
    return "\n".join(lines)


def _write_investigation_log(tests: list[FeasibleTest], run_dir: str | None):
    """Write the investigation_tests.md log file."""
    log_dir = run_dir or os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "investigation_tests.md")
    from datetime import datetime
    with open(path, "w") as f:
        f.write("# Investigation Tests Plan\n\n")
        f.write(f"_Generated at {datetime.now().isoformat()}_\n\n")
        feasible = [t for t in tests if t.feasible]
        not_feasible = [t for t in tests if not t.feasible]
        f.write(f"**Total tests evaluated:** {len(tests)}\n")
        f.write(f"**Feasible:** {len(feasible)}\n")
        f.write(f"**Not feasible:** {len(not_feasible)}\n\n")
        if feasible:
            f.write("## Tests to Execute\n\n")
            for i, t in enumerate(feasible, 1):
                f.write(f"### Test {i} (Hypothesis {t.hypothesis_index})\n")
                f.write(f"**Description:** {t.test_description}\n")
                f.write(f"**Data files:** {', '.join(t.data_files_needed)}\n")
                f.write(f"**Reason:** {t.reason}\n\n---\n\n")
        if not_feasible:
            f.write("## Tests NOT Feasible\n\n")
            for t in not_feasible:
                f.write(f"- (Hypothesis {t.hypothesis_index}) {t.test_description}\n")
                f.write(f"  Reason: {t.reason}\n\n")
    logger.info(f"Investigation tests log written to: {path}")


def hypothesis_investigator(state: AgentState):
    global _investigator_call_count
    _investigator_call_count += 1
    logger.info(f"Hypothesis Investigator started. (call #{_investigator_call_count})")

    hypotheses = state.get("hypotheses", [])
    plot_paths = state.get("plot_paths", [])
    messages = state.get("messages", [])
    raw_data_path = state.get("raw_data_path", "")

    logger.info(f"Investigator State: {len(hypotheses)} hypotheses, {len(plot_paths)} plots.")

    step_logger = get_step_logger()

    # ── Safety guards to prevent infinite loops ──────────────────────
    if _investigator_call_count >= _MAX_INVESTIGATOR_CALLS:
        next_node = "data_analyst"
        reasoning = (
            f"Safety: {_investigator_call_count} investigation cycles completed "
            f"with {len(plot_paths)} plots. Proceeding to report."
        )
        logger.info(f"Investigator safety exit: {reasoning}")
        if step_logger:
            step_logger.log_router_decision(next_node, reasoning, {
                "hypotheses_count": len(hypotheses),
                "plots_count": len(plot_paths),
                "call_count": _investigator_call_count,
            })
        msg = HumanMessage(content=f"Investigator Decision: {next_node}. Reasoning: {reasoning}")
        return {"next_node": next_node, "messages": [msg]}

    # If we have at least min_plots, investigation is complete
    if len(plot_paths) >= _MIN_PLOTS:
        next_node = "data_analyst"
        reasoning = f"{len(plot_paths)} plots generated covering the hypotheses. Investigation complete."
        logger.info(f"Investigator: sufficient plots. {reasoning}")
        if step_logger:
            step_logger.log_router_decision(next_node, reasoning, {
                "hypotheses_count": len(hypotheses),
                "plots_count": len(plot_paths),
            })
        msg = HumanMessage(content=f"Investigator Decision: {next_node}. Reasoning: {reasoning}")
        return {"next_node": next_node, "messages": [msg]}

    # ── Build data inventory from .data_cache and check_input_data ───
    data_inventory = _build_data_inventory(raw_data_path)
    check_dir = os.path.join(os.getcwd(), "check_input_data")
    if os.path.isdir(check_dir):
        data_inventory += "\n\nConsolidated data directory:\n"
        for f in sorted(os.listdir(check_dir)):
            if f.endswith(".csv"):
                fpath = os.path.join(check_dir, f)
                try:
                    with open(fpath) as fh:
                        header = fh.readline().rstrip()
                    data_inventory += f"  - check_input_data/{f}  [columns: {header}]\n"
                except Exception:
                    data_inventory += f"  - check_input_data/{f}\n"

    # ── Use LLM to evaluate hypotheses and determine feasible tests ──
    llm = get_llm(temperature=cfg("temperatures.hypothesis_investigator", 0.0))
    parser = PydanticOutputParser(pydantic_object=InvestigationPlan)

    recent_context = "\n".join([f"{m.type}: {str(m.content)[:500]}" for m in messages[-6:]])
    code_grounding = summarize_enriched_claims(state.get("code_enriched_claims", []), limit=8)

    system_prompt = (
        "You are the Hypothesis Investigator for a quantitative post-mortem trading agent.\n"
        "Your job is to:\n"
        "1. Evaluate each hypothesis and its proposed tests.\n"
        "2. Check which tests are FEASIBLE given the available data files listed below.\n"
        "3. A test is feasible ONLY if the required data columns and files exist.\n"
        "4. List all tests with their feasibility assessment.\n"
        "5. Decide the next step:\n"
        "   - 'code_generator': If there are feasible tests to implement (most common on first call).\n"
        "   - 'data_analyst': If all tests have been executed and plots generated, OR no feasible tests exist.\n\n"
        "When code-grounding context is available, use it to prefer tests that can connect observed behavior to concrete "
        "modules, config keys, and execution flow.\n\n"
        f"Available Data Files:\n{data_inventory}\n\n"
        f"{parser.get_format_instructions()}"
    )

    user_prompt = (
        f"Hypotheses to Investigate:\n{chr(10).join(hypotheses)}\n\n"
        f"Claim-to-code grounding:\n{code_grounding}\n\n"
        f"Plots Successfully Generated So Far: {len(plot_paths)}\n"
        f"{chr(10).join(plot_paths) if plot_paths else '(none)'}\n\n"
        f"Recent Execution Context:\n{recent_context}\n\n"
        "Evaluate the feasibility of each test and decide the next step."
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    try:
        plan = parser.invoke(response)
        next_node = plan.next_node
        reasoning = plan.reasoning
        feasible_tests = plan.feasible_tests

        # Write investigation tests log
        run_dir = step_logger.run_dir if step_logger else None
        _write_investigation_log(feasible_tests, run_dir)

        # Build list of feasible test descriptions for the code generator
        feasible_descriptions = [
            t.test_description for t in feasible_tests if t.feasible
        ]

        # Log the tests to console for visibility
        logger.info(f"Investigation plan: {len(feasible_descriptions)} feasible tests "
                     f"out of {len(feasible_tests)} evaluated.")
        for i, desc in enumerate(feasible_descriptions, 1):
            logger.info(f"  Test {i}: {desc}")

        # Build message with the tests for the code generator
        tests_msg = "Feasible tests to implement:\n"
        for i, desc in enumerate(feasible_descriptions, 1):
            tests_msg += f"  {i}. {desc}\n"

    except Exception as e:
        logger.warning(f"Structured output failed: {e}. Using fallback.")
        feasible_descriptions = []
        if plot_paths:
            next_node = "data_analyst"
            reasoning = f"Fallback: plots exist, proceeding to report. Parse error: {e}"
        else:
            next_node = "code_generator"
            reasoning = f"Fallback routing due to parsing error. Raw response: {response.content}"
        tests_msg = "Could not parse feasible tests. Code generator should analyze hypotheses directly."

    if step_logger:
        step_logger.log_router_decision(next_node, reasoning, {
            "hypotheses_count": len(hypotheses),
            "plots_count": len(plot_paths),
            "call_count": _investigator_call_count,
            "feasible_tests": len(feasible_descriptions),
        })

    msg = HumanMessage(
        content=f"Investigator Decision: {next_node}. Reasoning: {reasoning}\n\n{tests_msg}"
    )

    return {
        "next_node": next_node,
        "messages": [msg],
        "investigation_tests": feasible_descriptions,
    }
