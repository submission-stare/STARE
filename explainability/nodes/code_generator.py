"""
Code Generator Node — generates and fixes Python analysis scripts.

Strategy: Pre-load all key DataFrames in the injected preamble so the LLM
never needs to reference file paths, DATA_FILES keys, or pd.read_csv().
The LLM receives pre-loaded variables (df_transactions, df_snapshots, etc.)
and only writes analysis + plotting code.
"""

import os
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from utils.config import cfg
from utils.code_report import summarize_enriched_claims
from utils.data_scope import resolve_data_scope
from utils.llm import get_llm, invoke_llm_with_retries
from utils.step_logger import get_step_logger

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Data mapping: variable name → (description, file finder function)
# ═══════════════════════════════════════════════════════════════════

def _discover_dataframes(data_path: str) -> list[dict]:
    """Discover available CSV files and map them to DataFrame variable names.

    Returns a list of dicts with keys:
      - var: Python variable name (e.g., 'df_transactions')
      - path: absolute path to the CSV
      - description: human-readable description
      - columns: header line from the CSV
      - sample: first 2 data rows
    """
    abs_path = os.path.abspath(data_path)
    cwd = os.getcwd()
    scoped = resolve_data_scope(abs_path)
    frames: list[dict] = []

    def _read_header_and_sample(path: str) -> tuple[str, str]:
        """Read CSV header + 2 sample rows."""
        try:
            with open(path) as f:
                header = f.readline().rstrip()
                rows = []
                for _ in range(2):
                    line = f.readline().rstrip()
                    if line:
                        rows.append(line)
            return header, "\n".join(rows)
        except Exception:
            return "(could not read)", ""

    def _add(var: str, path: str, description: str):
        if os.path.isfile(path):
            cols, sample = _read_header_and_sample(path)
            frames.append({
                "var": var,
                "path": path,
                "description": description,
                "columns": cols,
                "sample": sample,
            })

    # ── Core consolidated files (PREFERRED) ─────────────────────────
    check_dir = os.path.join(cwd, "check_input_data")
    _add("df_transactions", os.path.join(check_dir, "transactions_all.csv"),
         "ALL agents' transactions merged (has 'agent' column)")
    _add("df_snapshots", os.path.join(check_dir, "snapshots_all.csv"),
         "ALL agents' portfolio snapshots merged (has 'agent' column)")

    # ── Account values ──────────────────────────────────────────────
    _add("df_account_values",
         os.path.join(scoped["trading_dir"], "account_values.csv"),
         "Daily portfolio value for all agents + benchmark")

    # ── Financial metrics ───────────────────────────────────────────
    _add("df_sharpe",
         os.path.join(scoped["trading_dir"],
                      "financial_metrics", "sharpe_summary_agents.csv"),
         "Sharpe ratio per agent")
    _add("df_advanced_stats",
         os.path.join(scoped["aggregated_summary_dir"],
                      "advanced_stats_summary.csv"),
         "Advanced stats: PSR, DSR, confidence intervals")

    # ── Aggregated multi-run data ───────────────────────────────────
    agg_dir = scoped["aggregated_summary_dir"]
    if os.path.isdir(agg_dir):
        _add("df_mean_portfolio_values",
             os.path.join(agg_dir, "mean_portfolio_account_values.csv"),
             "Mean portfolio account values across runs")
        _add("df_mean_portfolio_metrics",
             os.path.join(agg_dir, "mean_portfolio_metrics.csv"),
             "Mean portfolio metrics (Sharpe, final value, max drawdown)")
        _add("df_statistical_summary",
             os.path.join(agg_dir, "statistical_summary.csv"),
             "Statistical summary with CIs for Sharpe, VaR, ES, final value")
        _add("df_all_runs_sharpe",
             os.path.join(agg_dir, "all_runs_sharpe.csv"),
             "Sharpe ratios per run per agent")
        _add("df_all_runs_final_value",
             os.path.join(agg_dir, "all_runs_final_value.csv"),
             "Final portfolio values per run per agent")

    # ── Stock-level analysis ────────────────────────────────────────
    if os.path.isdir(agg_dir):
        _add("df_ticker_pnl",
             os.path.join(agg_dir,
                          "stock_analysis_01_profitability_ticker_pnl_by_agent.csv"),
             "Per-ticker PnL by agent (mean, std, median)")
        _add("df_ticker_ranking",
             os.path.join(agg_dir,
                          "stock_analysis_01_profitability_ticker_ranking_overall.csv"),
             "Overall ticker profitability ranking")
        _add("df_consensus_by_agent",
             os.path.join(agg_dir,
                          "stock_analysis_02_consensus_allocation_consensus_by_agent.csv"),
             "Portfolio allocation consensus by agent per ticker")
        _add("df_consensus_overall",
             os.path.join(agg_dir,
                          "stock_analysis_02_consensus_allocation_consensus_overall.csv"),
             "Overall allocation consensus per ticker")
        _add("df_final_weights",
             os.path.join(agg_dir,
                          "stock_analysis_02_consensus_final_weights_all_runs.csv"),
             "Final portfolio weights across all runs")

    # ── Comparison summary ──────────────────────────────────────────
    comp_dir = scoped["comparison_dir"]
    if os.path.isdir(comp_dir):
        _add("df_compare_transactions",
             os.path.join(comp_dir, "compare_transaction_summary.csv"),
             "Transaction summary comparison across agents")

    return frames


def _build_preamble(frames: list[dict]) -> str:
    """Build the Python preamble that pre-loads all DataFrames.

    This code is injected BEFORE the LLM-generated script so all
    variables are available without any pd.read_csv() calls.
    """
    lines = [
        "# === AUTO-INJECTED PREAMBLE (do not edit) ===",
        "import pandas as pd",
        "import matplotlib",
        "matplotlib.use('Agg')  # Non-interactive backend",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "# Pre-loaded DataFrames — use these directly, do NOT call pd.read_csv()",
    ]
    for f in frames:
        lines.append(f"# {f['var']}: {f['description']}")
        lines.append(f"{f['var']} = pd.read_csv(r\"{f['path']}\")")
        lines.append("")

    # Print available variables for debugging
    lines.append("print('=== Pre-loaded DataFrames ===')")
    for f in frames:
        lines.append(
            f"print(f\"  {f['var']}: {{len({f['var']})}} rows, "
            f"columns: {{{f['var']}.columns.tolist()}}\")"
        )
    lines.append("print('=== End pre-loaded ===')")
    lines.append("print()")
    lines.append("# === END AUTO-INJECTED PREAMBLE ===")
    lines.append("")

    return "\n".join(lines)


def _build_schema_reference(frames: list[dict]) -> str:
    """Build a human-readable schema reference for the LLM prompt.

    Shows variable names, descriptions, column names, and sample data
    so the LLM knows exactly what's available and what columns to use.
    """
    lines = []
    for f in frames:
        lines.append(f"  {f['var']} — {f['description']}")
        lines.append(f"    columns: {f['columns']}")
        if f['sample']:
            sample_lines = f['sample'].split('\n')
            lines.append(f"    sample:  {sample_lines[0]}")
            for sl in sample_lines[1:]:
                lines.append(f"             {sl}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Post-processing: strip preamble + re-inject
# ═══════════════════════════════════════════════════════════════════

def _strip_preamble(code: str) -> str:
    """Remove any auto-injected preamble from the code."""
    # Remove our preamble block
    code = re.sub(
        r'# === AUTO-INJECTED PREAMBLE.*?# === END AUTO-INJECTED PREAMBLE ===\s*',
        '',
        code,
        flags=re.DOTALL,
    )
    # Also remove old-style DATA_FILES/SafeDataFiles blocks (from previous versions)
    code = re.sub(
        r'# === AUTO-INJECTED DATA_FILES.*?# === END AUTO-INJECTED ===\s*',
        '',
        code,
        flags=re.DOTALL,
    )
    # Remove any LLM-generated pd.read_csv calls that re-load our pre-loaded variables
    # (e.g., df_transactions = pd.read_csv(...))
    code = re.sub(
        r'^(df_\w+)\s*=\s*pd\.read_csv\([^)]+\)\s*$',
        r'# \1 already pre-loaded',
        code,
        flags=re.MULTILINE,
    )
    return code.strip()


def _fix_redundant_imports(code: str) -> str:
    """Remove import statements for modules already imported in the preamble."""
    # These are always in the preamble
    for pattern in [
        r'^import pandas as pd\s*$',
        r'^import matplotlib\.pyplot as plt\s*$',
        r'^import numpy as np\s*$',
        r'^import matplotlib\s*$',
        r"^matplotlib\.use\(['\"]Agg['\"]\)\s*$",
        r"^warnings\.filterwarnings\([^)]+\)\s*$",
        r'^import warnings\s*$',
    ]:
        code = re.sub(pattern, '', code, flags=re.MULTILINE)
    return code.strip()


_SAVEFIG_COUNTER = 0


def _fix_plt_show(code: str) -> str:
    """Replace plt.show() calls with plt.savefig() + plt.close().

    The Agg backend ignores plt.show(), so plots are never written to disk.
    This post-processor auto-generates unique filenames for each plot.
    """
    global _SAVEFIG_COUNTER

    def _replacement(match: re.Match) -> str:
        global _SAVEFIG_COUNTER
        _SAVEFIG_COUNTER += 1
        indent = match.group(1)
        return (
            f"{indent}plt.savefig('plot_{_SAVEFIG_COUNTER:03d}.png', dpi=100, bbox_inches='tight')\n"
            f"{indent}plt.close()"
        )

    # Match plt.show() with any leading whitespace
    code = re.sub(r'^([ \t]*)plt\.show\(\)\s*$', _replacement, code, flags=re.MULTILINE)
    return code


def _numbered_code(code: str) -> str:
    """Return code with line numbers for the LLM to reference."""
    lines = code.split("\n")
    return "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))


def _extract_code_from_response(raw: str) -> str:
    """Strip markdown fences if the LLM wrapped its response."""
    if "```python" in raw:
        raw = raw.split("```python")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return raw.strip()


# ═══════════════════════════════════════════════════════════════════
# LLM interaction: generate and fix modes
# ═══════════════════════════════════════════════════════════════════

def _get_common_context(abs_data_path: str) -> dict:
    """Return context dict shared by both generate and fix modes."""
    frames = _discover_dataframes(abs_data_path)
    preamble = _build_preamble(frames)
    schema_ref = _build_schema_reference(frames)

    docs_context = ""
    try:
        docs_path = os.path.join(os.getcwd(), "specification", "documentation.md")
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                docs_context = f.read()
    except Exception as e:
        logger.warning(f"Could not load documentation: {e}")

    allowed_libs = ""
    try:
        req_path = os.path.join(os.getcwd(), "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path) as f:
                allowed_libs = f.read().strip()
    except Exception as e:
        logger.warning(f"Could not load requirements.txt: {e}")

    return {
        "frames": frames,
        "preamble": preamble,
        "schema_ref": schema_ref,
        "docs_context": docs_context,
        "allowed_libs": allowed_libs,
    }


def _generate_new_code(state: AgentState, ctx: dict) -> str:
    """Generate a brand-new analysis script (first call)."""
    logger.info("Code Generator: GENERATE mode (first call).")
    llm = get_llm(
        temperature=cfg("temperatures.code_generator", 0.0),
        max_tokens=cfg("code_generator.max_tokens", 8192),
    )

    hypotheses_str = "\n".join(state.get("hypotheses", []))
    investigation_tests = state.get("investigation_tests", [])
    code_grounding = summarize_enriched_claims(state.get("code_enriched_claims", []), limit=8)

    # ── System prompt ───────────────────────────────────────────────
    system_prompt = (
        "You are a Python data science expert specialized in quantitative finance.\n\n"
        "Your task: write ONLY the analysis and plotting code to validate the hypotheses.\n\n"
        "CRITICAL RULES:\n"
        "1. All DataFrames are ALREADY LOADED as variables. Do NOT call pd.read_csv().\n"
        "   Do NOT import pandas, matplotlib, or numpy — they are already imported.\n"
        "2. Use ONLY the pre-loaded variable names listed in the schema section below.\n"
        "   Do NOT invent variable names. If a variable is not in the schema, it does NOT exist.\n"
        "3. Use ONLY the EXACT column names shown in the schema. For example:\n"
        "   - Date column is 'date', NOT 'timestamp'\n"
        "   - Cash weight column is 'cash_weight', NOT 'cash_percent'\n"
        "   - Total assets column is 'total_asset', NOT 'total_value'\n"
        "4. Save EVERY plot with plt.savefig() and plt.close():\n"
        "   plt.savefig('descriptive_name.png', dpi=100, bbox_inches='tight')\n"
        "   plt.close()\n"
        "   NEVER use plt.show() — it does nothing. ALWAYS use plt.savefig().\n"
        "5. Print relevant statistics to stdout.\n"
        "6. Output ONLY pure Python code. No markdown fences, no explanation.\n"
        "7. Do NOT use broad try/except that silently swallows errors.\n"
        "8. Generate 5-8 impactful plots covering multiple hypotheses.\n"
        "9. EVERY plt.figure() MUST be followed by plt.savefig() and plt.close().\n"
        "10. When claim-to-code grounding is supplied, align the analysis with those modules, config keys, and flows.\n"
    )

    if ctx["docs_context"]:
        system_prompt += f"\nBackground documentation:\n{ctx['docs_context']}\n"

    if ctx["allowed_libs"]:
        system_prompt += (
            f"\nAllowed libraries (beyond pandas/matplotlib/numpy which are pre-imported):\n"
            f"{ctx['allowed_libs']}\n"
        )

    # ── User prompt ─────────────────────────────────────────────────
    var_names = ", ".join(f["var"] for f in ctx["frames"])
    user_prompt = (
        f"=== ALLOWED VARIABLE NAMES (ONLY these exist) ===\n"
        f"{var_names}\n"
        f"=== END ALLOWED VARIABLE NAMES ===\n\n"
        "=== PRE-LOADED DATAFRAMES (already available, just use them) ===\n"
        f"{ctx['schema_ref']}\n"
        "=== END PRE-LOADED DATAFRAMES ===\n\n"
    )

    if investigation_tests:
        user_prompt += "=== SPECIFIC TESTS TO IMPLEMENT ===\n"
        for i, test in enumerate(investigation_tests, 1):
            user_prompt += f"  {i}. {test}\n"
        user_prompt += "=== END TESTS ===\n\n"

    user_prompt += f"Hypotheses to validate:\n{hypotheses_str}\n\n"
    user_prompt += f"Claim-to-code grounding:\n{code_grounding}\n\n"

    tx_summary = state.get("transaction_summary", "")
    sn_summary = state.get("snapshot_summary", "")
    if tx_summary:
        user_prompt += f"Transaction Data Summary:\n{tx_summary}\n\n"
    if sn_summary:
        user_prompt += f"Snapshot Data Summary:\n{sn_summary}\n\n"

    user_prompt += (
        "REMEMBER: All DataFrames are pre-loaded. Do NOT import pandas/matplotlib/numpy. "
        "Do NOT call pd.read_csv(). Just write analysis + plotting code.\n"
        "Output the Python code now:"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = invoke_llm_with_retries(llm, messages)
    return _extract_code_from_response(response.content)


def _fix_existing_code(state: AgentState, ctx: dict) -> str:
    """Fix an existing broken script with minimal changes."""
    logger.info("Code Generator: FIX mode (retry — patching existing code).")
    llm = get_llm(
        temperature=cfg("temperatures.code_generator", 0.0),
        max_tokens=cfg("code_generator.max_tokens", 8192),
    )

    existing_code = state.get("generated_code", "")
    # Strip the preamble so the LLM only sees the analysis code
    user_code = _strip_preamble(existing_code)
    user_code = _fix_redundant_imports(user_code)
    numbered = _numbered_code(user_code)

    # Extract the execution error from recent messages
    error_traceback = ""
    for m in reversed(state.get("messages", [])):
        content = str(m.content)
        if "Execution Errors:" in content or "Exception traceback:" in content:
            error_traceback = content[:4000]
            break
    if not error_traceback:
        error_traceback = "\n".join(
            f"{m.type}: {str(m.content)[:2000]}" for m in state.get("messages", [])[-4:]
        )
    code_grounding = summarize_enriched_claims(state.get("code_enriched_claims", []), limit=8)

    # ── System prompt ───────────────────────────────────────────────
    system_prompt = (
        "You are a Python debugging expert. A data analysis script failed during execution.\n\n"
        "FIX the script with MINIMAL changes.\n\n"
        "RULES:\n"
        "1. Output the COMPLETE corrected script. It will replace the broken version entirely.\n"
        "2. Keep ALL working logic intact. Only change lines that cause errors.\n"
        "3. All DataFrames are ALREADY PRE-LOADED as variables. Do NOT call pd.read_csv().\n"
        "   Do NOT import pandas, matplotlib, or numpy — they are already imported.\n"
        "4. Use ONLY the EXACT column names from the schema below. Common mistakes:\n"
        "   - 'date' NOT 'timestamp'\n"
        "   - 'cash_weight' NOT 'cash_percent'\n"
        "   - 'total_asset' NOT 'total_value'\n"
        "5. Save plots as .png in the CURRENT WORKING DIRECTORY.\n"
        "6. Output ONLY pure Python code. No markdown, no explanation.\n"
    )

    # ── User prompt ─────────────────────────────────────────────────
    user_prompt = (
        "=== ERROR TRACEBACK ===\n"
        f"{error_traceback}\n"
        "=== END ERROR ===\n\n"
        "=== CURRENT SCRIPT (with line numbers) ===\n"
        f"{numbered}\n"
        "=== END SCRIPT ===\n\n"
        "=== PRE-LOADED DATAFRAMES (already available) ===\n"
        f"{ctx['schema_ref']}\n"
        "=== END PRE-LOADED DATAFRAMES ===\n\n"
        "=== CLAIM-TO-CODE GROUNDING ===\n"
        f"{code_grounding}\n"
        "=== END CLAIM-TO-CODE GROUNDING ===\n\n"
        "Identify the exact line(s) causing the error. "
        "Output the COMPLETE corrected script with minimal changes.\n"
        "Do NOT call pd.read_csv(). Do NOT import pandas/matplotlib/numpy.\n"
        "Output the corrected code now:"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = invoke_llm_with_retries(llm, messages)
    return _extract_code_from_response(response.content)


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def code_generator(state: AgentState):
    """
    Generates or fixes a Python script for hypothesis validation.

    Two modes:
    - GENERATE (first call): writes a new analysis script from scratch.
    - FIX (retry after error): takes the existing broken script + traceback
      and asks the LLM to make surgical corrections.

    In both modes, the preamble with pre-loaded DataFrames is auto-injected
    so the LLM never needs to handle file paths.
    """
    logger.info("Starting Code Generator node...")

    raw_data_path = state.get("raw_data_path", "")
    abs_data_path = os.path.abspath(raw_data_path)
    ctx = _get_common_context(abs_data_path)

    # ── Decide mode: generate vs fix ────────────────────────────────
    is_retry = state.get("code_fix_retries", 0) > 0 and state.get("generated_code", "")

    if is_retry:
        code = _fix_existing_code(state, ctx)
    else:
        code = _generate_new_code(state, ctx)

    # ── Post-process: strip any preamble the LLM generated & inject ours ──
    code = _strip_preamble(code)
    code = _fix_redundant_imports(code)
    code = _fix_plt_show(code)  # Replace plt.show() → plt.savefig()

    # Combine: our preamble + LLM's analysis code
    final_code = ctx["preamble"] + "\n" + code

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_generated_code(final_code)

    mode_label = "FIX" if is_retry else "GENERATE"
    msg = HumanMessage(
        content=f"[Code Generator — {mode_label}] Generated Python code for validation:\n\n" + final_code
    )

    return {"generated_code": final_code, "messages": [msg]}
