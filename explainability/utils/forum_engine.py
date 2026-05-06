"""
Generic Forum Engine — multi-panelist discussion with consensus.

Provides a standardised three-round protocol that can be reused
wherever multiple LLM panelists need to converge on an answer:

  Round 1 – **Propose**: each panelist independently generates an answer.
  Round 2 – **Evaluate**: each panelist reviews ALL other panelists' answers.
  Round 3 – **Consensus**: panelists see evaluations and converge on a
             unified final answer (majority-wins, or a designated
             synthesiser merges the best elements).

Usage::

    from utils.forum_engine import run_forum
    result = run_forum(
        topic="What is the agent doing to achieve its trading goal?",
        context="... hypotheses, test results ...",
        panelists=get_forum_panelists(),          # list[(label, ChatModel)]
        synthesiser=get_llm(temperature=0.1),     # optional single LLM
        max_rounds=2,
    )
    print(result.final_answer)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PanelistProposal:
    """One panelist's initial answer."""
    label: str
    answer: str


@dataclass
class PanelistEvaluation:
    """One panelist's evaluation of all proposals."""
    label: str
    evaluation: str  # Free-form critique / improvements


@dataclass
class ForumResult:
    """Outcome of a complete forum run."""
    topic: str
    proposals: list[PanelistProposal]
    evaluations: list[PanelistEvaluation]
    final_answer: str
    rounds_used: int


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _round1_propose(
    label: str,
    llm,
    topic: str,
    context: str,
    system_prompt: str,
) -> PanelistProposal:
    """Round 1: panelist generates its initial answer."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"=== TOPIC ===\n{topic}\n=== END TOPIC ===\n\n"
            f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
            "Provide your detailed, well-structured answer now."
        )),
    ]
    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"[{label}] Round 1 failed: {e}")
        answer = f"(panelist {label} failed to respond: {e})"
    return PanelistProposal(label=label, answer=answer)


def _round2_evaluate(
    label: str,
    llm,
    topic: str,
    proposals: list[PanelistProposal],
    system_prompt: str,
) -> PanelistEvaluation:
    """Round 2: panelist evaluates all proposals (including its own)."""
    proposals_text = "\n\n".join(
        f"--- Panelist {p.label} ---\n{p.answer}\n--- End {p.label} ---"
        for p in proposals
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"=== TOPIC ===\n{topic}\n=== END TOPIC ===\n\n"
            f"=== ALL PROPOSALS ===\n{proposals_text}\n=== END PROPOSALS ===\n\n"
            "Evaluate each proposal above:\n"
            "1. Identify the strongest points from each panelist.\n"
            "2. Identify weaknesses, gaps, or contradictions.\n"
            "3. Suggest what a final consensus answer should include.\n"
            "Be constructive and specific."
        )),
    ]
    try:
        response = llm.invoke(messages)
        evaluation = response.content.strip()
    except Exception as e:
        logger.error(f"[{label}] Round 2 failed: {e}")
        evaluation = f"(panelist {label} failed to evaluate: {e})"
    return PanelistEvaluation(label=label, evaluation=evaluation)


def _round3_consensus(
    synthesiser,
    topic: str,
    proposals: list[PanelistProposal],
    evaluations: list[PanelistEvaluation],
    system_prompt: str,
) -> str:
    """Round 3: a synthesiser LLM merges all proposals + evaluations into a consensus."""
    proposals_text = "\n\n".join(
        f"--- Panelist {p.label} ---\n{p.answer}\n--- End {p.label} ---"
        for p in proposals
    )
    evaluations_text = "\n\n".join(
        f"--- Evaluation by {e.label} ---\n{e.evaluation}\n--- End {e.label} ---"
        for e in evaluations
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"=== TOPIC ===\n{topic}\n=== END TOPIC ===\n\n"
            f"=== PROPOSALS ===\n{proposals_text}\n=== END PROPOSALS ===\n\n"
            f"=== PEER EVALUATIONS ===\n{evaluations_text}\n=== END EVALUATIONS ===\n\n"
            "Synthesise a single, comprehensive consensus answer that:\n"
            "1. Incorporates the best elements from all proposals.\n"
            "2. Addresses weaknesses raised in the evaluations.\n"
            "3. Is well-structured, evidence-based, and definitive.\n"
            "Output ONLY the final consensus answer — no meta-commentary."
        )),
    ]
    try:
        response = synthesiser.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Consensus synthesis failed: {e}")
        # Fallback: use the longest proposal
        longest = max(proposals, key=lambda p: len(p.answer))
        return f"(consensus failed — using {longest.label}'s answer)\n\n{longest.answer}"


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def run_forum(
    topic: str,
    context: str,
    panelists: list[tuple[str, object]],
    synthesiser=None,
    propose_system_prompt: str = "",
    evaluate_system_prompt: str = "",
    consensus_system_prompt: str = "",
    max_rounds: int = 3,
) -> ForumResult:
    """
    Run the full three-round forum protocol.

    Parameters
    ----------
    topic : str
        The question / topic to be discussed.
    context : str
        Background context (hypotheses, test results, data summaries, etc.).
    panelists : list[(label, ChatModel)]
        LLM instances to participate. If only one panelist, the forum
        degrades gracefully to a single-agent answer.
    synthesiser : ChatModel, optional
        LLM used for the final consensus round. If None, uses the first panelist.
    propose_system_prompt : str
        System prompt for round 1 (proposal). A default is used if empty.
    evaluate_system_prompt : str
        System prompt for round 2 (evaluation). A default is used if empty.
    consensus_system_prompt : str
        System prompt for round 3 (consensus). A default is used if empty.
    max_rounds : int
        1 = propose only, 2 = propose + evaluate, 3 = full consensus.

    Returns
    -------
    ForumResult
        Contains proposals, evaluations, and the final consensus answer.
    """
    if not panelists:
        logger.warning("No panelists provided — returning empty forum result.")
        return ForumResult(
            topic=topic, proposals=[], evaluations=[],
            final_answer="(no panelists available)", rounds_used=0,
        )

    # ── Default system prompts ──────────────────────────────────────
    if not propose_system_prompt:
        propose_system_prompt = (
            "You are a senior quantitative analyst participating in a panel discussion.\n"
            "Read the context carefully and provide a thorough, evidence-based answer to the topic.\n"
            "Support your arguments with specific data points where possible."
        )

    if not evaluate_system_prompt:
        evaluate_system_prompt = (
            "You are a senior quantitative analyst peer-reviewing panel proposals.\n"
            "Evaluate each proposal constructively: identify strengths, weaknesses, and gaps.\n"
            "Suggest what elements should be kept, improved, or discarded."
        )

    if not consensus_system_prompt:
        consensus_system_prompt = (
            "You are a senior research director synthesising a panel discussion into a\n"
            "definitive consensus answer. Merge the strongest elements from all proposals,\n"
            "address weaknesses raised in the peer evaluations, and produce a single\n"
            "well-structured, evidence-based final answer."
        )

    if synthesiser is None:
        _, synthesiser = panelists[0]

    # ── ROUND 1: Propose ────────────────────────────────────────────
    logger.info(f"Forum Round 1: {len(panelists)} panelist(s) proposing on '{topic[:80]}...'")
    proposals: list[PanelistProposal] = []

    with ThreadPoolExecutor(max_workers=len(panelists)) as executor:
        futures = {
            executor.submit(
                _round1_propose, label, llm, topic, context, propose_system_prompt
            ): label
            for label, llm in panelists
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                proposals.append(future.result())
            except Exception as e:
                logger.error(f"[{label}] Round 1 exception: {e}")
                proposals.append(PanelistProposal(label=label, answer=f"(failed: {e})"))

    if max_rounds < 2 or len(panelists) == 1:
        # Single panelist or max_rounds=1 → use the single proposal as final answer
        final = proposals[0].answer if proposals else "(no proposals)"
        return ForumResult(
            topic=topic, proposals=proposals, evaluations=[],
            final_answer=final, rounds_used=1,
        )

    # ── ROUND 2: Evaluate ───────────────────────────────────────────
    logger.info(f"Forum Round 2: {len(panelists)} panelist(s) evaluating proposals.")
    evaluations: list[PanelistEvaluation] = []

    with ThreadPoolExecutor(max_workers=len(panelists)) as executor:
        futures = {
            executor.submit(
                _round2_evaluate, label, llm, topic, proposals, evaluate_system_prompt
            ): label
            for label, llm in panelists
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                evaluations.append(future.result())
            except Exception as e:
                logger.error(f"[{label}] Round 2 exception: {e}")
                evaluations.append(PanelistEvaluation(
                    label=label, evaluation=f"(failed: {e})"
                ))

    if max_rounds < 3:
        # No consensus round — use the proposals
        final = proposals[0].answer if proposals else "(no proposals)"
        return ForumResult(
            topic=topic, proposals=proposals, evaluations=evaluations,
            final_answer=final, rounds_used=2,
        )

    # ── ROUND 3: Consensus ──────────────────────────────────────────
    logger.info("Forum Round 3: synthesising consensus answer.")
    final_answer = _round3_consensus(
        synthesiser, topic, proposals, evaluations, consensus_system_prompt,
    )

    return ForumResult(
        topic=topic, proposals=proposals, evaluations=evaluations,
        final_answer=final_answer, rounds_used=3,
    )
