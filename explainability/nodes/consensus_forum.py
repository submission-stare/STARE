"""
Consensus Forum Node — multi-panelist discussion to answer the three
core questions about agent behavior:

  1. **What?** — What is the agent doing to achieve its trading goal?
  2. **How?** — What means/methods is it using?
  3. **Why?** — Why does it exhibit this behavior?

Uses the generic forum engine (``utils.forum_engine.run_forum``) so the
same multi-round protocol (propose → evaluate → consensus) is applied.
"""

from __future__ import annotations

import logging
from langchain_core.messages import HumanMessage

from state import AgentState
from utils.config import cfg
from utils.code_report import summarize_code_evidence_results, summarize_enriched_claims
from utils.forum_engine import run_forum, ForumResult
from utils.llm import get_forum_panelists, get_llm
from utils.step_logger import get_step_logger

logger = logging.getLogger(__name__)

# ── The three core questions ────────────────────────────────────────
QUESTIONS = [
    {
        "key": "what",
        "title": "What?",
        "topic": (
            "What is the agent doing to try to achieve its trading objective? "
            "Describe the observable actions, trading patterns, and decision-making "
            "strategies visible in the data."
        ),
    },
    {
        "key": "how",
        "title": "How?",
        "topic": (
            "How is the agent executing its strategy? What specific mechanisms, "
            "techniques, and methods does it use? Consider entry/exit timing, "
            "position sizing, portfolio allocation, risk management, and any other "
            "observable execution patterns."
        ),
    },
    {
        "key": "why",
        "title": "Why?",
        "topic": (
            "Why does the agent exhibit this behavior? What underlying logic, "
            "reward signals, learned policies, or market conditions explain the "
            "observed patterns? Connect the behavior to the agent's training "
            "objective, architecture, and the data evidence gathered."
        ),
    },
]


def _build_context(state: AgentState) -> str:
    """Assemble the context that panelists need to answer the questions."""
    parts: list[str] = []

    # Hypotheses
    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        parts.append("=== HYPOTHESES ===")
        for h in hypotheses:
            parts.append(h)
        parts.append("=== END HYPOTHESES ===\n")

    # Investigation tests
    tests = state.get("investigation_tests", [])
    if tests:
        parts.append("=== INVESTIGATION TESTS ===")
        for i, t in enumerate(tests, 1):
            parts.append(f"  {i}. {t}")
        parts.append("=== END INVESTIGATION TESTS ===\n")

    # Transaction summary
    tx = state.get("transaction_summary", "")
    if tx:
        parts.append(f"=== TRANSACTION DATA SUMMARY ===\n{tx}\n=== END TRANSACTION DATA ===\n")

    # Snapshot summary
    sn = state.get("snapshot_summary", "")
    if sn:
        parts.append(f"=== PORTFOLIO SNAPSHOT SUMMARY ===\n{sn}\n=== END SNAPSHOT ===\n")

    # Execution results (from messages)
    exec_parts: list[str] = []
    for m in state.get("messages", []):
        content = str(m.content)
        if "Code Execution Results:" in content or "Execution Output:" in content:
            # Keep only execution outputs, truncated for context
            exec_parts.append(content[:3000])
    if exec_parts:
        parts.append("=== CODE EXECUTION RESULTS ===")
        parts.append("\n---\n".join(exec_parts[-3:]))  # last 3 executions
        parts.append("=== END EXECUTION RESULTS ===\n")

    enriched_claims = summarize_enriched_claims(state.get("code_enriched_claims", []), limit=8)
    if enriched_claims != "(none)":
        parts.append("=== CLAIM-TO-CODE GROUNDING ===")
        parts.append(enriched_claims)
        parts.append("=== END CLAIM-TO-CODE GROUNDING ===\n")

    evidence_results = summarize_code_evidence_results(state.get("code_evidence_results", []), limit=5)
    if evidence_results != "(none)":
        parts.append("=== CODE EVIDENCE RESULTS ===")
        parts.append(evidence_results)
        parts.append("=== END CODE EVIDENCE RESULTS ===\n")

    # Plot paths
    plots = state.get("plot_paths", [])
    if plots:
        import os
        parts.append("=== GENERATED PLOTS ===")
        for p in plots:
            name = os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
            parts.append(f"  - {name} ({os.path.basename(p)})")
        parts.append("=== END PLOTS ===\n")

    return "\n".join(parts) if parts else "(no context available)"


def consensus_forum(state: AgentState):
    """
    Run the three-round forum for each core question and return
    the consensus answers.
    """
    logger.info("Starting Consensus Forum node...")

    panelists = get_forum_panelists(
        temperature=cfg("temperatures.consensus_forum", 0.3),
        max_tokens=cfg("consensus_forum.max_tokens", 8192),
    )
    logger.info(f"Consensus Forum: {len(panelists)} panelist(s): {[l for l, _ in panelists]}")

    synthesiser = get_llm(
        temperature=cfg("temperatures.consensus_forum", 0.3),
        max_tokens=cfg("consensus_forum.max_tokens", 8192),
    )

    context = _build_context(state)
    max_rounds = cfg("consensus_forum.max_rounds", 3)

    # System prompts tailored for agent behavior analysis
    propose_prompt = (
        "You are a senior quantitative analyst specialised in algorithmic trading.\n"
        "You are participating in a panel discussion to answer a specific question about\n"
        "a trading agent's behavior. You have access to:\n"
        "  - Hypotheses that were formulated about the agent\n"
        "  - Investigation test results and generated plots\n"
        "  - Transaction and portfolio data summaries\n"
        "  - Claim-to-code mappings and code evidence when available\n\n"
        "Base your answer strictly on the evidence provided. Be specific and cite\n"
        "data points. When code grounding is available, cite concrete modules, config keys, and execution flow.\n"
        "Avoid speculation without evidence."
    )

    evaluate_prompt = (
        "You are a senior quantitative analyst peer-reviewing your colleagues' answers.\n"
        "For each proposal:\n"
        "1. Identify the strongest evidence-backed arguments.\n"
        "2. Point out unsupported claims or logical gaps.\n"
        "3. Note any important evidence that was overlooked.\n"
        "4. Suggest how the final consensus should prioritise the insights.\n"
        "5. Check whether each proposal ties the observed behavior back to concrete implementation details when available.\n"
        "Be constructive and evidence-focused."
    )

    consensus_prompt = (
        "You are a senior research director writing the definitive consensus answer.\n"
        "Merge the best elements from all panelists, address the peer evaluations,\n"
        "and write a single comprehensive answer that is:\n"
        "  - Evidence-based (cite specific data points and hypothesis results)\n"
        "  - Code-grounded when possible (cite modules, config keys, or execution flow from the supplied context)\n"
        "  - Well-structured (use paragraphs and clear logical flow)\n"
        "  - Definitive (take clear positions, avoid hedging without evidence)\n"
        "Output ONLY the final answer — no meta-commentary."
    )

    answers: dict[str, str] = {}
    forum_results: dict[str, ForumResult] = {}

    for q in QUESTIONS:
        logger.info(f"Consensus Forum: discussing '{q['title']}'...")
        result = run_forum(
            topic=q["topic"],
            context=context,
            panelists=panelists,
            synthesiser=synthesiser,
            propose_system_prompt=propose_prompt,
            evaluate_system_prompt=evaluate_prompt,
            consensus_system_prompt=consensus_prompt,
            max_rounds=max_rounds,
        )
        answers[q["key"]] = result.final_answer
        forum_results[q["key"]] = result
        logger.info(
            f"  '{q['title']}': consensus reached in {result.rounds_used} rounds "
            f"({len(result.proposals)} proposals, {len(result.evaluations)} evaluations, "
            f"{len(result.final_answer)} chars)"
        )

    # ── Log ─────────────────────────────────────────────────────────
    step_logger = get_step_logger()
    if step_logger:
        _log_consensus(step_logger, forum_results)

    # Compose readable summary for the message
    summary_lines = ["Consensus Forum completed:\n"]
    for q in QUESTIONS:
        summary_lines.append(f"## {q['title']}\n")
        summary_lines.append(answers[q['key']])
        summary_lines.append("")

    summary = "\n".join(summary_lines)

    return {
        "consensus_answers": answers,
        "messages": [HumanMessage(content=summary)],
    }


def _log_consensus(step_logger, forum_results: dict[str, ForumResult]):
    """Write consensus forum log to the run directory."""
    import os
    path = os.path.join(step_logger.run_dir, "consensus_forum.md")
    with open(path, "w") as f:
        from datetime import datetime
        f.write("# Consensus Forum Results\n\n")
        f.write(f"_Logged at {datetime.now().isoformat()}_\n\n")

        for q in QUESTIONS:
            key = q["key"]
            result = forum_results[key]
            f.write(f"## {q['title']}\n\n")
            f.write(f"**Topic:** {q['topic']}\n\n")
            f.write(f"**Rounds used:** {result.rounds_used}\n\n")

            f.write("### Round 1 — Proposals\n\n")
            for p in result.proposals:
                f.write(f"#### Panelist: {p.label}\n\n{p.answer}\n\n---\n\n")

            if result.evaluations:
                f.write("### Round 2 — Evaluations\n\n")
                for e in result.evaluations:
                    f.write(f"#### Evaluator: {e.label}\n\n{e.evaluation}\n\n---\n\n")

            f.write("### Final Consensus\n\n")
            f.write(f"{result.final_answer}\n\n")
            f.write("---\n\n")

    step_logger.log_step("consensus_forum", {
        "questions": [q["key"] for q in QUESTIONS],
        "rounds_per_question": {
            q["key"]: forum_results[q["key"]].rounds_used for q in QUESTIONS
        },
    })
