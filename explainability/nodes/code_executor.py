from langchain_core.messages import HumanMessage
from state import AgentState
from utils.code_helpers import execute_python_code
from utils.config import cfg
from utils.step_logger import get_step_logger
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_MAX_CODE_RETRIES: int = cfg("investigator.max_code_retries", 3)

# Track execution history across iterations (success / retrying)
_execution_history: list[dict] = []


def _write_execution_status_log(run_dir: str | None):
    """Write/overwrite the code_execution_status.md log each iteration."""
    log_dir = run_dir or os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "code_execution_status.md")

    succeeded = [e for e in _execution_history if e["status"] == "success"]
    retrying = [e for e in _execution_history if e["status"] == "retrying"]
    failed = [e for e in _execution_history if e["status"] == "failed"]

    with open(path, "w") as f:
        f.write("# Code Execution Status\n\n")
        f.write(f"_Last updated: {datetime.now().isoformat()}_\n\n")
        f.write(f"**Total executions:** {len(_execution_history)}\n")
        f.write(f"**Succeeded:** {len(succeeded)} | "
                f"**Retrying:** {len(retrying)} | "
                f"**Failed (gave up):** {len(failed)}\n\n")

        if succeeded:
            f.write("## Succeeded\n\n")
            for e in succeeded:
                f.write(f"- Iteration {e['iteration']}: "
                        f"{len(e.get('plots', []))} plots generated\n")
                for p in e.get("plots", []):
                    f.write(f"  - {p}\n")
                f.write("\n")

        if retrying:
            f.write("## Being Corrected (retrying)\n\n")
            for e in retrying:
                f.write(f"- Iteration {e['iteration']} (retry {e['retry_count']}/"
                        f"{_MAX_CODE_RETRIES})\n")
                f.write(f"  Error: {e.get('error_summary', 'unknown')}\n\n")

        if failed:
            f.write("## Failed (max retries reached)\n\n")
            for e in failed:
                f.write(f"- Iteration {e['iteration']}: gave up after "
                        f"{e['retry_count']} retries\n")
                f.write(f"  Last error: {e.get('error_summary', 'unknown')}\n\n")

    logger.info(f"Execution status log written to: {path}")


def code_executor(state: AgentState):
    logger.info("Starting Code Executor node...")
    code = state.get("generated_code", "")
    current_retries = state.get("code_fix_retries", 0)

    if not code:
        return {
            "messages": [HumanMessage(content="No code found to execute.")],
            "next_node": "hypothesis_investigator",
        }

    execution_result = execute_python_code(code, working_dir="./generated_code_results")

    output = execution_result["output"]
    error = execution_result["error"]
    new_plots = execution_result["new_plots"]

    result_str = ""
    if error:
        result_str += f"Execution Errors:\n{error}\n"
        # Also include the stdout output which may contain partial prints before crash
        if output:
            result_str += f"\nStdout before error:\n{output}\n"
    elif output:
        result_str += f"Execution Output:\n{output}\n"
    if new_plots:
        result_str += f"\nGenerated Plots: {', '.join(new_plots)}\n"
    elif not error:
        result_str += "\nNo plots were generated.\n"

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_code_execution(output, error, new_plots)

    # ── Routing: success → hypothesis_investigator, error → code_generator ──
    iteration = len(_execution_history) + 1

    if error:
        new_retries = current_retries + 1
        if new_retries >= _MAX_CODE_RETRIES:
            # Max retries reached, give up and move on
            next_node = "hypothesis_investigator"
            status = "failed"
            logger.warning(
                f"Code fix retry limit reached ({new_retries}/{_MAX_CODE_RETRIES}). "
                f"Moving to hypothesis_investigator."
            )
            result_str += (
                f"\n⚠️ Max code fix retries reached ({new_retries}/{_MAX_CODE_RETRIES}). "
                f"Moving on despite errors.\n"
            )
        else:
            # Send back to code_generator for fixing
            next_node = "code_generator"
            status = "retrying"
            logger.info(
                f"Code execution failed (retry {new_retries}/{_MAX_CODE_RETRIES}). "
                f"Sending back to code_generator for fix."
            )
            result_str += (
                f"\n🔄 Code execution failed. Retry {new_retries}/{_MAX_CODE_RETRIES}. "
                f"Sending to code_generator for correction.\n"
            )

        error_lines = error.strip().split("\n")
        error_summary = error_lines[-1] if error_lines else "unknown error"

        _execution_history.append({
            "iteration": iteration,
            "status": status,
            "retry_count": new_retries,
            "error_summary": error_summary[:200],
        })
    elif not new_plots:
        # Code ran without errors but produced NO plots — treat as soft failure
        new_retries = current_retries + 1
        error = "0 plots generated"  # set error so it shows in traceback extraction
        if new_retries >= _MAX_CODE_RETRIES:
            next_node = "hypothesis_investigator"
            status = "failed"
            logger.warning(
                f"Code ran but generated 0 plots. Max retries reached "
                f"({new_retries}/{_MAX_CODE_RETRIES}). Moving on."
            )
            result_str += (
                f"\nExecution Errors:\n"
                f"RuntimeError: 0 plots generated. The script ran but did not create any .png files.\n"
                f"⚠️ Max retries reached ({new_retries}/{_MAX_CODE_RETRIES}).\n"
            )
        else:
            next_node = "code_generator"
            status = "retrying"
            logger.info(
                f"Code ran but generated 0 plots (retry {new_retries}/{_MAX_CODE_RETRIES}). "
                f"Sending back to code_generator."
            )
            result_str += (
                f"\nExecution Errors:\n"
                f"RuntimeError: 0 plots generated. The script ran but did not create any .png files.\n"
                f"The script MUST use plt.savefig('name.png', dpi=100, bbox_inches='tight') "
                f"followed by plt.close() for EVERY plot.\n"
                f"NEVER use plt.show() — it produces no output in this environment.\n"
                f"Retry {new_retries}/{_MAX_CODE_RETRIES}.\n"
            )

        _execution_history.append({
            "iteration": iteration,
            "status": status,
            "retry_count": new_retries,
            "error_summary": "0 plots generated",
        })
    else:
        # Success - reset retries and go to hypothesis_investigator
        next_node = "hypothesis_investigator"
        new_retries = 0
        _execution_history.append({
            "iteration": iteration,
            "status": "success",
            "retry_count": current_retries,
            "plots": [os.path.basename(p) for p in new_plots],
        })
        logger.info(
            f"Code executed successfully. {len(new_plots)} plots generated. "
            f"Routing to hypothesis_investigator."
        )

    # Write/overwrite the status log
    run_dir = step_logger.run_dir if step_logger else None
    _write_execution_status_log(run_dir)

    msg = HumanMessage(content="Code Execution Results:\n" + result_str)
    return {
        "messages": [msg],
        "plot_paths": new_plots,
        "next_node": next_node,
        "code_fix_retries": new_retries,
    }
