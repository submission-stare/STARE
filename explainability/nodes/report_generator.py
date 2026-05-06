"""
Report Generator Node — builds a structured three-part LaTeX/Markdown report.

Part 1: Hypotheses raised and their details.
Part 2: Tests performed, their objectives, and results with images (LaTeX \\ref).
Part 3: Consensus answers to "What?", "How?", "Why?" from the forum.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from utils.config import cfg
from utils.llm import get_llm
from utils.step_logger import get_step_logger
import json
import os
import re
import subprocess
import logging

logger = logging.getLogger(__name__)


def _escape_latex(text: str) -> str:
    return (
        text.replace("\\", "/")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _claim_id_for_hypothesis(idx: int) -> str:
    return f"hypothesis_{idx}"


def _claim_lookup(state: AgentState) -> dict[str, dict]:
    return {
        item.get("claim_id"): item
        for item in state.get("code_enriched_claims", [])
        if item.get("claim_id")
    }


def _evidence_lookup(state: AgentState) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for item in state.get("code_evidence_results", []):
        task_id = item.get("task_id", "")
        match = re.match(r"hypothesis_(\d+)_evidence$", task_id)
        if match:
            mapping[_claim_id_for_hypothesis(int(match.group(1)))] = item.get("result", {})
    return mapping


def _build_code_grounding_inline(claim: dict | None, evidence: dict | None) -> str:
    if not claim and not evidence:
        return ""

    lines = ["\\textbf{Code grounding:}"]
    if claim:
        code_paths = ", ".join(claim.get("code_paths", [])) or "(none)"
        config_keys = ", ".join(claim.get("config_keys", [])) or "(none)"
        lines.append(f"Relevant code paths: {_escape_latex(code_paths)}")
        lines.append(f"Relevant config keys: {_escape_latex(config_keys)}")
        lines.append(f"Exercised flow: {_escape_latex(claim.get('exercised_flow', '(not specified)'))}")
        lines.append(f"Why this code matters: {_escape_latex(claim.get('explanation', '(not specified)'))}")
    if evidence:
        supporting_paths = ", ".join(evidence.get("supporting_paths", [])) or "(none)"
        config_keys = ", ".join(evidence.get("config_keys", [])) or "(none)"
        snippets = " | ".join(evidence.get("evidence_snippets", [])) or "(none)"
        lines.append(f"Evidence summary: {_escape_latex(evidence.get('summary', '(not available)'))}")
        lines.append(f"Supporting paths: {_escape_latex(supporting_paths)}")
        lines.append(f"Evidence config keys: {_escape_latex(config_keys)}")
        lines.append(f"Evidence snippets: {_escape_latex(snippets)}")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Part 1: Hypotheses
# ═══════════════════════════════════════════════════════════════════

def _build_hypotheses_section(state: AgentState) -> str:
    """Build Part 1 — Hypotheses and their details."""
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return "\\section{Hypotheses}\n\nNo hypotheses were generated.\n"

    claims_by_id = _claim_lookup(state)
    evidence_by_id = _evidence_lookup(state)
    lines = ["\\section{Hypotheses}\n"]
    lines.append(
        "The following hypotheses were formulated by the multi-panelist forum "
        "to guide the investigation of agent trading behaviour.\n"
    )

    for idx, h in enumerate(hypotheses, start=1):
        # Each hypothesis is already formatted as markdown with ### headers
        # Convert to LaTeX subsections
        converted = _md_hypothesis_to_latex(h)
        lines.append(converted)
        grounding = _build_code_grounding_inline(
            claims_by_id.get(_claim_id_for_hypothesis(idx)),
            evidence_by_id.get(_claim_id_for_hypothesis(idx)),
        )
        if grounding:
            lines.append(grounding)
        lines.append("")

    return "\n".join(lines)


def _md_hypothesis_to_latex(h: str) -> str:
    """Convert a markdown hypothesis block to LaTeX."""
    # Replace ### Hypothesis N → \subsection{Hypothesis N}
    text = re.sub(r"^### (Hypothesis \d+)", r"\\subsection{\1}", h, flags=re.MULTILINE)
    # Replace **Field:** → \textbf{Field:}
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    # Replace markdown bullet lists → \begin{itemize}
    lines = text.split("\n")
    result: list[str] = []
    in_itemize = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_itemize:
                result.append("\\begin{itemize}")
                in_itemize = True
            result.append(f"  \\item {stripped[2:]}")
        else:
            if in_itemize:
                result.append("\\end{itemize}")
                in_itemize = False
            result.append(line)
    if in_itemize:
        result.append("\\end{itemize}")
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════
#  Part 2: Tests and Results with Images
# ═══════════════════════════════════════════════════════════════════

def _build_tests_section(
    state: AgentState,
    plot_paths: list[str],
    llm,
) -> str:
    """Build Part 2 — Tests, objectives, results, and image references.

    Uses an LLM call to produce the narrative tying tests to images,
    then injects the image figures with LaTeX \\label and \\ref.
    """
    investigation_tests = state.get("investigation_tests", [])
    hypotheses = state.get("hypotheses", [])
    code_claims = state.get("code_enriched_claims", [])
    code_evidence = state.get("code_evidence_results", [])

    # Build figure environment and ref list for the LLM
    figures: list[dict] = []
    for i, p in enumerate(plot_paths, 1):
        basename = os.path.basename(p)
        name_clean = os.path.splitext(basename)[0].replace("_", " ").title()
        label = f"fig:{os.path.splitext(basename)[0]}"
        try:
            rel_path = os.path.relpath(p, start=os.getcwd())
        except Exception:
            rel_path = p
        figures.append({
            "index": i,
            "label": label,
            "name": name_clean,
            "rel_path": rel_path,
            "basename": basename,
        })

    # Build the LaTeX figure blocks
    figure_blocks = []
    for fig in figures:
        figure_blocks.append(
            f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=0.85\\textwidth]{{{fig['rel_path']}}}\n"
            f"  \\caption{{{fig['name']}}}\n"
            f"  \\label{{{fig['label']}}}\n"
            f"\\end{{figure}}"
        )

    # Build available refs for the LLM
    ref_list = "\n".join(
        f"  - \\ref{{{fig['label']}}} — {fig['name']}"
        for fig in figures
    )

    # Ask the LLM to write the narrative
    tests_text = "\n".join(f"  {i}. {t}" for i, t in enumerate(investigation_tests, 1))
    hypotheses_text = "\n".join(hypotheses[:10])  # first 10 for context

    # Extract execution output
    exec_output = ""
    for m in reversed(state.get("messages", [])):
        content = str(m.content)
        if "Execution Output:" in content or "Code Execution Results:" in content:
            exec_output = content[:4000]
            break

    system_prompt = (
        "You are a technical report writer for quantitative finance research.\n"
        "Write the 'Tests and Results' section of a report in LaTeX format.\n\n"
        "RULES:\n"
        "1. For EACH test, explain its objective and present the results.\n"
        "2. Reference ALL figures using \\ref{fig:label} syntax — the labels are provided below.\n"
        "3. Every figure MUST be cited at least once in the text.\n"
        "4. Write in clear, technical prose. No code or raw data.\n"
        "5. Output ONLY the LaTeX content for this section (no \\section{} header, I'll add it).\n"
        "6. Use \\ref{} for cross-references, never use markdown image syntax.\n"
    )

    user_prompt = (
        f"=== TESTS PERFORMED ===\n{tests_text}\n=== END TESTS ===\n\n"
        f"=== HYPOTHESES BEING TESTED ===\n{hypotheses_text}\n=== END HYPOTHESES ===\n\n"
        f"=== CLAIM-TO-CODE GROUNDING ===\n{json.dumps(code_claims[:8], indent=2)}\n=== END CLAIM-TO-CODE GROUNDING ===\n\n"
        f"=== CODE EVIDENCE RESULTS ===\n{json.dumps(code_evidence[:5], indent=2)}\n=== END CODE EVIDENCE RESULTS ===\n\n"
        f"=== AVAILABLE FIGURE REFERENCES ===\n{ref_list}\n=== END REFS ===\n\n"
        f"=== EXECUTION OUTPUT (results data) ===\n{exec_output}\n=== END OUTPUT ===\n\n"
        "Write the tests & results narrative now, citing all figures with \\ref{}. "
        "When implementation references are available, connect the observed results to concrete modules, config keys, or execution flow:"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        narrative = response.content.strip()
        # Strip markdown fences if present
        if "```" in narrative:
            narrative = re.sub(r"```(?:latex|tex)?\n?", "", narrative)
            narrative = narrative.replace("```", "")
    except Exception as e:
        logger.warning(f"LLM failed to generate tests narrative: {e}")
        narrative = "Test results analysis could not be generated.\n"

    # Assemble the section
    lines = ["\\section{Tests and Results}\n"]
    lines.append(narrative)
    lines.append("")

    # Append all figure blocks
    if figure_blocks:
        lines.append("% --- Figures ---")
        for fb in figure_blocks:
            lines.append(fb)
            lines.append("")

    # Safety: ensure all figures are referenced
    for fig in figures:
        ref = f"\\ref{{{fig['label']}}}"
        if ref not in narrative:
            lines.append(
                f"\nSee also Figure~{ref} ({fig['name']}).\n"
            )

    implementation_notes = _build_tests_implementation_notes(state)
    if implementation_notes:
        lines.append("")
        lines.append(implementation_notes)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Part 3: Consensus Answers
# ═══════════════════════════════════════════════════════════════════

def _build_consensus_section(consensus_answers: dict[str, str]) -> str:
    """Build Part 3 — The three consensus answers."""
    if not consensus_answers:
        return "\\section{Agent Behavior Analysis}\n\nConsensus analysis not available.\n"

    titles = {
        "what": "What is the agent doing?",
        "how": "How is it doing it?",
        "why": "Why does it exhibit this behavior?",
    }

    lines = ["\\section{Agent Behavior Analysis}\n"]
    lines.append(
        "The following analysis was produced through a multi-panelist forum where "
        "experts independently proposed answers, peer-reviewed each other's responses, "
        "and converged on a consensus.\n"
    )

    for key in ["what", "how", "why"]:
        answer = consensus_answers.get(key, "(not available)")
        title = titles.get(key, key)
        lines.append(f"\\subsection{{{title}}}")
        lines.append(answer)
        lines.append("")

    return "\n".join(lines)


def _build_tests_implementation_notes(state: AgentState) -> str:
    claims_by_id = _claim_lookup(state)
    evidence_by_id = _evidence_lookup(state)
    notes: list[str] = []
    for idx, _hypothesis in enumerate(state.get("hypotheses", [])[:5], start=1):
        claim = claims_by_id.get(_claim_id_for_hypothesis(idx))
        evidence = evidence_by_id.get(_claim_id_for_hypothesis(idx))
        if not claim and not evidence:
            continue
        notes.append(f"\\textbf{{Hypothesis {idx} implementation refs:}}")
        if claim:
            notes.append(
                f"Paths: {_escape_latex(', '.join(claim.get('code_paths', [])) or '(none)')}. "
                f"Config keys: {_escape_latex(', '.join(claim.get('config_keys', [])) or '(none)')}."
            )
        if evidence:
            notes.append(
                f"Evidence summary: {_escape_latex(evidence.get('summary', '(not available)'))}"
            )
        notes.append("")
    if not notes:
        return ""
    return "\\subsection{Implementation References}\n" + "\n".join(notes).strip()


def _build_implementation_evidence_appendix(state: AgentState) -> str:
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return "\\section{Implementation Evidence Appendix}\n\nNo hypothesis-level code grounding was available.\n"

    claims_by_id = _claim_lookup(state)
    evidence_by_id = _evidence_lookup(state)
    lines = ["\\section{Implementation Evidence Appendix}\n"]
    lines.append(
        "This appendix lists the code-grounding context attached to each hypothesis so behavioral claims remain traceable to implementation details.\n"
    )

    for idx, _hypothesis in enumerate(hypotheses, start=1):
        claim_id = _claim_id_for_hypothesis(idx)
        claim = claims_by_id.get(claim_id)
        evidence = evidence_by_id.get(claim_id)
        lines.append(f"\\subsection{{Hypothesis {idx}}}")
        if not claim and not evidence:
            lines.append("No code grounding was captured for this hypothesis.")
            lines.append("")
            continue
        if claim:
            lines.append(f"Claim text: {_escape_latex(claim.get('claim_text', '(not available)'))}")
            lines.append(f"Relevant code paths: {_escape_latex(', '.join(claim.get('code_paths', [])) or '(none)')}")
            lines.append(f"Relevant config keys: {_escape_latex(', '.join(claim.get('config_keys', [])) or '(none)')}")
            lines.append(f"Exercised flow: {_escape_latex(claim.get('exercised_flow', '(not specified)'))}")
            lines.append(f"Why this code matters: {_escape_latex(claim.get('explanation', '(not specified)'))}")
        if evidence:
            lines.append(f"Evidence summary: {_escape_latex(evidence.get('summary', '(not available)'))}")
            lines.append(f"Supporting paths: {_escape_latex(', '.join(evidence.get('supporting_paths', [])) or '(none)')}")
            lines.append(f"Evidence config keys: {_escape_latex(', '.join(evidence.get('config_keys', [])) or '(none)')}")
            snippets = evidence.get("evidence_snippets", [])
            lines.append(f"Evidence snippets: {_escape_latex(' | '.join(snippets) if snippets else '(none)')}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  LaTeX document wrapper
# ═══════════════════════════════════════════════════════════════════

def _wrap_latex_document(body: str, title: str = "Trading Agent Post-Mortem Report") -> str:
    """Wrap the body content in a complete LaTeX document."""
    return (
        "\\documentclass[12pt,a4paper]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{enumitem}\n"
        "\\geometry{margin=2.5cm}\n"
        "\\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}\n"
        "\n"
        f"\\title{{{title}}}\n"
        "\\author{Alpha-Inspector Autonomous Agent}\n"
        "\\date{\\today}\n"
        "\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\tableofcontents\n"
        "\\newpage\n"
        "\n"
        f"{body}\n"
        "\n"
        "\\end{document}\n"
    )


# ═══════════════════════════════════════════════════════════════════
#  Main node
# ═══════════════════════════════════════════════════════════════════

def report_generator(state: AgentState):
    """
    Consolidates all phases into a structured three-part LaTeX report:
      1. Hypotheses
      2. Tests and Results (with figure references)
      3. Consensus Answers (What? How? Why?)
    """
    logger.info("Starting Report Generator node...")
    llm = get_llm(
        temperature=cfg("temperatures.report_generator", 0.2),
        max_tokens=cfg("report_generator.max_tokens", 16384),
    )

    hypotheses = state.get("hypotheses", [])
    plot_paths = state.get("plot_paths", [])
    consensus_answers = state.get("consensus_answers", {})

    # ── Build the three parts ───────────────────────────────────────
    part1 = _build_hypotheses_section(state)
    part2 = _build_tests_section(state, plot_paths, llm)
    part3 = _build_consensus_section(consensus_answers)
    part4 = _build_implementation_evidence_appendix(state)

    body = f"{part1}\n\\newpage\n{part2}\n\\newpage\n{part3}\n\\newpage\n{part4}"

    # ── Generate LaTeX document ─────────────────────────────────────
    latex_content = _wrap_latex_document(body)

    # Save .tex file
    tex_path = "report.tex"
    with open(tex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"LaTeX report saved to {tex_path}")

    # Also save a readable markdown version
    md_content = _latex_to_readable_md(body, plot_paths)
    with open("report.md", "w") as f:
        f.write(md_content)

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_report(latex_content)

    # ── Compile PDF via pdflatex or tectonic ────────────────────────
    pdf_path = "report.pdf"
    try:
        # Try tectonic first (self-contained, downloads packages)
        result = subprocess.run(
            ["tectonic", tex_path],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            logger.info(f"PDF report generated via tectonic: {pdf_path}")
        else:
            logger.warning(f"tectonic failed (rc={result.returncode}): {result.stderr[:500]}")
            # Fallback to pandoc
            _try_pandoc_pdf(md_content, pdf_path)
    except FileNotFoundError:
        logger.info("tectonic not found, trying pandoc...")
        _try_pandoc_pdf(md_content, pdf_path)
    except subprocess.TimeoutExpired:
        logger.warning("tectonic timed out after 180s, trying pandoc...")
        _try_pandoc_pdf(md_content, pdf_path)
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")

    return {
        "final_report": latex_content,
        "messages": [HumanMessage(
            content="Final report generated: report.tex, report.md, report.pdf"
        )],
    }


def _try_pandoc_pdf(md_content: str, pdf_path: str):
    """Fallback PDF generation via pandoc from markdown."""
    try:
        with open("_report_tmp.md", "w") as f:
            f.write(md_content)
        result = subprocess.run(
            ["pandoc", "_report_tmp.md", "-o", pdf_path, "--pdf-engine=tectonic"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info(f"PDF generated via pandoc: {pdf_path}")
        else:
            logger.warning(f"pandoc failed: {result.stderr[:500]}")
    except Exception as e:
        logger.warning(f"pandoc fallback failed: {e}")
    finally:
        if os.path.exists("_report_tmp.md"):
            os.remove("_report_tmp.md")


def _latex_to_readable_md(latex_body: str, plot_paths: list[str]) -> str:
    """Convert the LaTeX body to a readable Markdown version."""
    text = latex_body

    # sections → # headers
    text = re.sub(r"\\section\{(.+?)\}", r"# \1", text)
    text = re.sub(r"\\subsection\{(.+?)\}", r"## \1", text)
    text = re.sub(r"\\subsubsection\{(.+?)\}", r"### \1", text)

    # textbf → **bold**
    text = re.sub(r"\\textbf\{(.+?)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{(.+?)\}", r"*\1*", text)

    # Remove figure environments but keep image refs as markdown
    def _replace_figure(match):
        fig_text = match.group(0)
        img_match = re.search(r"\\includegraphics\[.*?\]\{(.+?)\}", fig_text)
        cap_match = re.search(r"\\caption\{(.+?)\}", fig_text)
        if img_match:
            path = img_match.group(1)
            caption = cap_match.group(1) if cap_match else ""
            return f"![{caption}]({path})\n"
        return ""

    text = re.sub(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        _replace_figure, text, flags=re.DOTALL,
    )

    # \ref{fig:xxx} → (Fig. N)
    text = re.sub(r"\\ref\{(fig:.+?)\}", r"(see \1)", text)

    # itemize/enumerate
    text = re.sub(r"\\begin\{itemize\}", "", text)
    text = re.sub(r"\\end\{itemize\}", "", text)
    text = re.sub(r"\\begin\{enumerate\}", "", text)
    text = re.sub(r"\\end\{enumerate\}", "", text)
    text = re.sub(r"\\item\s*", "- ", text)

    # Clean up remaining LaTeX commands
    text = re.sub(r"\\newpage", "\n---\n", text)
    text = re.sub(r"\\tableofcontents", "", text)
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"~", " ", text)

    # Remove any remaining \command{...}
    text = re.sub(r"\\[a-zA-Z]+\{(.+?)\}", r"\1", text)

    # Clean up excess blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return f"# Trading Agent Post-Mortem Report\n\n{text}"
