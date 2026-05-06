"""
Hypothesis Forum – multi-panelist hypothesis generation with deduplication,
peer review, and adversarial defence.

Replaces ``hypothesis_maker`` as a drop-in node in the LangGraph graph.

Phase 1 (Sprint 2):
  1. N panelists generate hypotheses **in parallel**.
  2. All hypotheses are merged into a single pool.
  3. Near-duplicate hypotheses are removed via text similarity.
  4. A configurable cap is applied.

Phase 3 (Sprint 3 — peer review):
  5. A reviewer LLM scores each hypothesis on relevance, testability,
     specificity, and novelty (1-10).  Hypotheses below ``min_score``
     or with verdict DROP are eliminated.

Phase 4 (Sprint 3 — adversarial defence, optional):
  6. A devil's-advocate LLM challenges every surviving hypothesis.
  7. A defender responds to HIGH-severity challenges only.
  8. Hypotheses that fail the defence are dropped; others may be revised.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from nodes.hypothesis_maker import (
    AdversarialChallenge,
    ChallengesOutput,
    DefenceOutput,
    DefenceVerdict,
    HypothesesOutput,
    HypothesisReview,
    PeerReviewOutput,
    StructuredHypothesis,
    _format_hypothesis,
)
from state import AgentState
from utils.config import cfg
from utils.code_report import build_code_context_digest
from utils.data_scope import scoped_walk_roots
from utils.llm import get_forum_panelists, get_llm, get_llm_by_provider, _parse_panelist_spec
from utils.step_logger import get_step_logger

logger = logging.getLogger(__name__)

# ── Config (from config/settings.yaml, overridable by env vars) ─────
_SIMILARITY_THRESHOLD: float = cfg("forum.dedup_threshold", 0.70)
_MAX_HYPOTHESES: int = cfg("forum.max_hypotheses", 20)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _build_prompts(state: AgentState) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) pair for hypothesis generation.

    Extracted here so every panelist receives the **exact same context**.
    """
    raw_data_path = state.get("raw_data_path", "")

    # ── Data sample ─────────────────────────────────────────────────
    try:
        if os.path.isdir(raw_data_path):
            walk_roots = scoped_walk_roots(raw_data_path)
            tree: list[str] = []
            summary_scanned = ""
            for walk_root in walk_roots:
                for root, dirs, files in os.walk(walk_root):
                    level = root.replace(raw_data_path, "").count(os.sep)
                    indent = " " * 4 * level
                    tree.append(f"{indent}{os.path.basename(root)}/")
                    subindent = " " * 4 * (level + 1)
                    for f in files:
                        tree.append(f"{subindent}{f}")
                        if f in ("advanced_stats_summary.csv", "statistical_summary.csv") and not summary_scanned:
                            try:
                                df = pd.read_csv(os.path.join(root, f))
                                summary_scanned = f"Summary File ({f}) Head:\n{df.head().to_string()}\n"
                            except Exception:
                                pass
            if not tree:
                raise ValueError(
                    "Configured data scope directories were not found under "
                    f"{raw_data_path}."
                )
            tree_str = "\n".join(tree[:100])
            if len(tree) > 100:
                tree_str += "\n... (truncated)"
            data_sample = f"Data Directory Structure:\n{tree_str}\n\n{summary_scanned}"
        elif raw_data_path.endswith(".csv"):
            df = pd.read_csv(raw_data_path)
            data_sample = f"Data Head:\n{df.head().to_string()}\n\nData Description:\n{df.describe().to_string()}"
        elif raw_data_path.endswith(".json"):
            df = pd.read_json(raw_data_path)
            data_sample = f"Data Head:\n{df.head().to_string()}\n\nData Description:\n{df.describe().to_string()}"
        else:
            raise ValueError("Unsupported file format or path.")
    except Exception as e:
        data_sample = f"Could not load data from {raw_data_path}: {e}"

    # ── Documentation context ───────────────────────────────────────
    docs_context = ""
    try:
        docs_path = os.path.join(os.getcwd(), "specification", "documentation.md")
        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                docs_context = f.read()
    except Exception as e:
        logger.warning(f"Could not load documentation: {e}")

    # ── Transaction / snapshot summaries ────────────────────────────
    tx_summary = state.get("transaction_summary", "")
    sn_summary = state.get("snapshot_summary", "")
    code_context_bundle = state.get("code_context_bundle", {})

    # ── System prompt ───────────────────────────────────────────────
    system_prompt = (
        "You are a senior quantitative analyst. Your task is to review the trading model's "
        "performance metrics AND transaction-level behaviour to propose up to 20 clear, testable hypotheses "
        "regarding agent behaviour or potential issues.\n"
        "Focus on actionable, data-driven hypotheses that explain *why* agents act the way they do "
        "(e.g., 'Agent X concentrates buys on ticker Y during periods of low cash weight, suggesting "
        "momentum-chasing behaviour').\n"
        "For each hypothesis, provide:\n"
        "  - hypothesis: A clear, concise statement.\n"
        "  - rationale: Why this hypothesis is plausible.\n"
        "  - key_factors: 2-4 key data factors that support it.\n"
        "  - possible_tests: Concrete tests to confirm or refute it.\n"
        "\nCRITICAL: Each hypothesis MUST be unique and substantially different from every other. "
        "Do NOT repeat, rephrase, or reformulate the same idea. If two hypotheses share the same core "
        "mechanism or conclusion, keep only the stronger one.\n"
    )
    if docs_context:
        system_prompt += f"\nPipeline documentation context:\n{docs_context}\n"
    if code_context_bundle:
        system_prompt += (
            "\nYou also have a compact code-context digest. Use it to connect behavioral hypotheses to concrete "
            "implementation paths and config consumers, but do not quote raw source or invent unsupplied code details.\n"
        )

    # ── User prompt ─────────────────────────────────────────────────
    user_prompt = f"Here is a sample of the raw performance metrics:\n\n{data_sample}\n\n"
    if tx_summary:
        user_prompt += f"Transaction-level Data:\n{tx_summary}\n\n"
    if sn_summary:
        user_prompt += f"Portfolio Snapshot Data:\n{sn_summary}\n\n"
    if code_context_bundle:
        user_prompt += f"{build_code_context_digest(code_context_bundle)}\n\n"

    existing_hypotheses = state.get("hypotheses", [])
    if existing_hypotheses:
        user_prompt += (
            "IMPORTANT — The following hypotheses have ALREADY been generated. "
            "Do NOT repeat or reformulate any of them. Generate only NEW, substantially different hypotheses.\n\n"
            "=== EXISTING HYPOTHESES (DO NOT REPEAT) ===\n"
            + "\n\n".join(existing_hypotheses)
            + "\n=== END EXISTING HYPOTHESES ===\n\n"
            f"Generate new hypotheses that are different from the {len(existing_hypotheses)} above. "
            "Produce as many novel ones as are relevant (up to 20 total including the existing ones)."
        )
    else:
        user_prompt += "Please generate between 1 and 20 hypotheses (as many as are relevant, up to 20)."

    return system_prompt, user_prompt


def _call_panelist(
    label: str,
    llm,
    system_prompt: str,
    user_prompt: str,
) -> list[StructuredHypothesis]:
    """
    Call a single panelist and return its list of StructuredHypothesis objects.
    Falls back to empty list on failure.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        structured_llm = llm.with_structured_output(HypothesesOutput)
        result: HypothesesOutput = structured_llm.invoke(messages)
        result.hypotheses = result.hypotheses[:_MAX_HYPOTHESES]
        logger.info(f"Panelist [{label}] produced {len(result.hypotheses)} hypotheses.")
        # Strip echoed "Hypothesis:" prefix the LLM may add to the hypothesis field
        for h in result.hypotheses:
            h.hypothesis = _strip_echo(h.hypothesis)
        return result.hypotheses
    except Exception as e:
        logger.warning(f"Panelist [{label}] structured output failed: {e}. Trying plain text fallback...")
        try:
            response = llm.invoke(messages)
            fallback = _parse_markdown_hypotheses(response.content)
            logger.info(f"Panelist [{label}] fallback produced {len(fallback)} hypotheses.")
            return fallback
        except Exception as e2:
            logger.error(f"Panelist [{label}] failed completely: {e2}")
            return []


def _parse_markdown_hypotheses(text: str) -> list[StructuredHypothesis]:
    """
    Parse a plain-text / markdown LLM response into StructuredHypothesis objects.

    Tries multiple strategies to handle diverse LLM output formats:

      1. Headers: ``### Hypothesis N``, ``### **N.``, ``**Hypothesis N:``, etc.
      2. Field-boundary split: repeated ``**Rationale**:`` markers.
      3. Numbered list: ``1. <title>``.
      4. Naive line split (last resort).
    """
    # ── Strategy 1: split by hypothesis headers ─────────────────────
    # Broad pattern: any ##+ header with a number, or bold "Hypothesis N"
    header_pattern = re.compile(
        r"^(?:"
        r"#{2,4}\s+\*{0,2}(?:Hypothesis\s+)?\d+"       # ### Hypothesis 1 / ### **1. / ### 1.
        r"|"
        r"\*{2}(?:Hypothesis\s+)?\d+[\.\):]\s*.+?\*{2}" # **Hypothesis 1: Title** / **1. Title**
        r")",
        re.MULTILINE | re.IGNORECASE,
    )
    splits = list(header_pattern.finditer(text))

    if len(splits) >= 2:
        blocks: list[str] = []
        for i, m in enumerate(splits):
            start = m.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            blocks.append(text[start:end].strip())

        results: list[StructuredHypothesis] = []
        for block in blocks:
            h = _parse_single_block(block)
            if h:
                results.append(h)
        if results:
            logger.info(f"Markdown parser: Strategy 1 (headers) → {len(results)} hypotheses.")
            return results

    # ── Strategy 2: field-boundary split ─────────────────────────────
    # If the model outputs blocks separated by repeated "**Rationale**:" markers
    # (without explicit hypothesis headers), reconstruct from field boundaries.
    rationale_hits = list(re.finditer(
        r"^[\s-]*\*{0,2}Rationale\*{0,2}\s*:",
        text,
        re.MULTILINE | re.IGNORECASE,
    ))

    if len(rationale_hits) >= 2:
        results = _split_by_field_boundaries(text, rationale_hits)
        if results:
            logger.info(f"Markdown parser: Strategy 2 (field boundaries) → {len(results)} hypotheses.")
            return results

    # ── Strategy 3: numbered list (1. text) ─────────────────────────
    numbered = re.findall(r"^\*{0,2}\d+[\.\)]\s*\*{0,2}\s*(.+)", text, re.MULTILINE)
    if numbered and len(numbered) >= 2:
        results = []
        for line in numbered:
            clean = _clean_md(line.strip())
            if len(clean) > 15:
                results.append(StructuredHypothesis(
                    hypothesis=clean,
                    rationale="(parsed from numbered list)",
                    key_factors=["See hypothesis statement"],
                    possible_tests=["Verify with data analysis"],
                ))
        if results:
            logger.info(f"Markdown parser: Strategy 3 (numbered) → {len(results)} hypotheses.")
            return results

    # ── Strategy 4 (last resort): naive line split ──────────────────
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    results = [
        StructuredHypothesis(
            hypothesis=_clean_md(line),
            rationale="(generated from plain-text fallback)",
            key_factors=["See hypothesis statement"],
            possible_tests=["Verify with data analysis"],
        )
        for line in lines
        if len(line) > 30
    ]
    logger.warning(f"Markdown parser: Strategy 4 (naive) → {len(results)} lines. Consider improving LLM output format.")
    return results


def _split_by_field_boundaries(
    text: str,
    rationale_hits: list[re.Match],
) -> list[StructuredHypothesis]:
    """
    Reconstruct hypothesis blocks from repeated ``**Rationale**:`` markers.

    For each Rationale occurrence, looks backward for a title line (the last
    non-field, non-empty line before the Rationale) and forward for
    Key Factors / Possible Tests.
    """
    lines = text.split("\n")
    results: list[StructuredHypothesis] = []

    for hi, hit in enumerate(rationale_hits):
        # Find which line the Rationale match is on
        block_start_offset = hit.start()
        char_count = 0
        rationale_line_idx = 0
        for li, line in enumerate(lines):
            char_count += len(line) + 1  # +1 for \n
            if char_count > block_start_offset:
                rationale_line_idx = li
                break

        # ── Title: scan backward for a non-field, non-empty line ────
        title = ""
        for back in range(rationale_line_idx - 1, max(rationale_line_idx - 5, -1), -1):
            candidate = lines[back].strip()
            if not candidate:
                continue
            # Skip if it looks like a field value or intro text
            if re.match(r"^[\s-]*\*{0,2}(Rationale|Key Factors|Possible Tests)\*{0,2}\s*:", candidate, re.IGNORECASE):
                continue
            if candidate.lower().startswith("here are"):
                continue
            title = _clean_md(re.sub(r"^#{1,4}\s*", "", candidate))
            title = re.sub(r"^\*{0,2}\d+[\.\):\s]*\*{0,2}\s*", "", title)  # strip numbering
            title = _clean_md(title)
            break

        # ── Block end: next Rationale or end of text ────────────────
        if hi + 1 < len(rationale_hits):
            next_offset = rationale_hits[hi + 1].start()
            char_count2 = 0
            end_line_idx = len(lines)
            for li, line in enumerate(lines):
                char_count2 += len(line) + 1
                if char_count2 > next_offset:
                    end_line_idx = li
                    break
        else:
            end_line_idx = len(lines)

        # ── Parse the block from rationale_line_idx to end_line_idx ─
        block_text = "\n".join(lines[rationale_line_idx:end_line_idx])
        rationale = _extract_field(block_text, "Rationale")
        key_factors_raw = _extract_field(block_text, "Key Factors")
        possible_tests_raw = _extract_field(block_text, "Possible Tests")

        key_factors = _extract_bullets(key_factors_raw) if key_factors_raw else []
        possible_tests = _extract_bullets(possible_tests_raw) if possible_tests_raw else []

        # Collect loose bullet lines as possible tests (some models list tests as plain dashes)
        if not possible_tests:
            loose = []
            in_tests_zone = False
            for line in lines[rationale_line_idx:end_line_idx]:
                if re.match(r"^[\s-]*\*{0,2}Possible Tests\*{0,2}", line, re.IGNORECASE):
                    in_tests_zone = True
                    continue
                if re.match(r"^[\s-]*\*{0,2}(Rationale|Key Factors)\*{0,2}", line, re.IGNORECASE):
                    in_tests_zone = False
                    continue
                if in_tests_zone and line.strip().startswith("-"):
                    loose.append(_clean_md(line.strip().lstrip("-").strip()))
            if loose:
                possible_tests = loose

        # Single-line key_factors/tests
        if not key_factors and key_factors_raw:
            key_factors = [_clean_md(key_factors_raw)]
        if not possible_tests and possible_tests_raw:
            possible_tests = [_clean_md(possible_tests_raw)]

        if not key_factors:
            key_factors = ["See hypothesis statement"]
        if not possible_tests:
            possible_tests = ["Verify with data analysis"]

        if title or rationale:
            results.append(StructuredHypothesis(
                hypothesis=_strip_echo(title) if title else "(title not found)",
                rationale=rationale or "(not provided)",
                key_factors=key_factors,
                possible_tests=possible_tests,
            ))

    return results


def _parse_single_block(block: str) -> StructuredHypothesis | None:
    """
    Extract hypothesis, rationale, key_factors, possible_tests from a
    markdown block that starts with a header line.
    """
    lines = block.split("\n")
    if not lines:
        return None

    # ── Extract hypothesis title from the header line ───────────────
    header = lines[0]
    # Strip markdown headers: ### **Hypothesis 1: Title** / ### 1. Title / **1. Title**
    title = re.sub(r"^#{1,4}\s*", "", header)                     # strip leading #
    title = re.sub(r"^\*{0,2}(?:Hypothesis\s+)?\d+[:\.\)]*\s*", "", title, flags=re.IGNORECASE)
    title = _clean_md(title)
    if not title or len(title) < 5:
        title = _clean_md(header)

    # ── Collect remaining text by field ─────────────────────────────
    body = "\n".join(lines[1:])

    # If the body has an explicit **Hypothesis:** field, prefer its content
    # over the (often terse) header-derived title.
    hyp_field = _extract_field(body, "Hypothesis")
    if hyp_field and len(hyp_field) > len(title):
        title = _strip_echo(_clean_md(hyp_field))

    rationale = _extract_field(body, "Rationale")
    key_factors_raw = _extract_field(body, "Key Factors")
    possible_tests_raw = _extract_field(body, "Possible Tests")

    # Parse bullet items for list fields
    key_factors = _extract_bullets(key_factors_raw) if key_factors_raw else ["See hypothesis statement"]
    possible_tests = _extract_bullets(possible_tests_raw) if possible_tests_raw else ["Verify with data analysis"]

    # If key_factors / possible_tests were a single line (no bullets), wrap them
    if not key_factors:
        key_factors = [key_factors_raw.strip()] if key_factors_raw.strip() else ["See hypothesis statement"]
    if not possible_tests:
        possible_tests = [possible_tests_raw.strip()] if possible_tests_raw.strip() else ["Verify with data analysis"]

    return StructuredHypothesis(
        hypothesis=title,
        rationale=rationale or "(not provided)",
        key_factors=key_factors,
        possible_tests=possible_tests,
    )


def _extract_field(body: str, field_name: str) -> str:
    """
    Extract the content after ``**Field Name:**`` or ``- **Field Name:**``
    up to the next field or end of block.
    """
    pattern = re.compile(
        rf"-?\s*\*{{0,2}}{field_name}\*{{0,2}}[:\s]*(.+?)(?=\n\s*-?\s*\*{{0,2}}(?:Rationale|Key Factors|Possible Tests|Hypothesis)\*{{0,2}}[:\s]|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(body)
    if m:
        return _clean_md(m.group(1).strip())
    return ""


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet points (``- item`` or ``* item``) from a text block."""
    items = re.findall(r"^[\s]*[-*•]\s+(.+)", text, re.MULTILINE)
    return [_clean_md(i.strip()) for i in items if i.strip()]


def _clean_md(text: str) -> str:
    """Strip residual markdown bold markers (``**``) from field text."""
    text = text.strip()
    # Remove leading/trailing ** markers
    while text.startswith("**"):
        text = text[2:]
    while text.endswith("**"):
        text = text[:-2]
    return text.strip()


def _strip_echo(text: str) -> str:
    """Remove a redundant leading ``Hypothesis:`` echo that some models add.

    Example: ``Hypothesis: A2C exhibits…`` → ``A2C exhibits…``
    """
    return re.sub(r"^\s*Hypothesis\s*:\s*", "", text, count=1, flags=re.IGNORECASE).strip()


def _deduplicate(
    hypotheses: list[StructuredHypothesis],
    threshold: float = _SIMILARITY_THRESHOLD,
) -> list[StructuredHypothesis]:
    """
    Remove near-duplicate hypotheses based on the ``hypothesis`` statement text.

    Uses ``difflib.SequenceMatcher`` ratio — two hypotheses with ratio >= threshold
    are considered duplicates; the first one encountered is kept.
    """
    if not hypotheses:
        return []

    unique: list[StructuredHypothesis] = []
    seen_texts: list[str] = []

    for h in hypotheses:
        text = h.hypothesis.lower().strip()
        is_dup = False
        for seen in seen_texts:
            ratio = SequenceMatcher(None, text, seen).ratio()
            if ratio >= threshold:
                is_dup = True
                logger.debug(f"Dedup: dropped '{h.hypothesis[:60]}...' (similarity={ratio:.2f})")
                break
        if not is_dup:
            unique.append(h)
            seen_texts.append(text)

    dropped = len(hypotheses) - len(unique)
    if dropped:
        logger.info(f"Deduplication: kept {len(unique)}, dropped {dropped} near-duplicates (threshold={threshold}).")
    return unique


# ═══════════════════════════════════════════════════════════════════
#  Phase 3 — Peer Review
# ═══════════════════════════════════════════════════════════════════

def _get_review_llm(temperature: float | None = None):
    """Return the LLM used for peer review (and adversarial defence)."""
    temp = temperature if temperature is not None else cfg("temperatures.peer_reviewer", 0.1)
    reviewer_spec: str = cfg("forum.peer_review.reviewer", "")
    if reviewer_spec:
        provider, model = _parse_panelist_spec(reviewer_spec)
        return get_llm_by_provider(provider, model, temperature=temp)
    return get_llm(temperature=temp)


def _peer_review(
    hypotheses: list[StructuredHypothesis],
    min_score: float,
) -> tuple[list[StructuredHypothesis], dict]:
    """
    A reviewer LLM scores each hypothesis on four dimensions (1-10).

    Returns ``(kept_hypotheses, review_stats)`` where *review_stats* contains
    per-hypothesis scores for logging.
    """
    if not hypotheses:
        return [], {}

    reviewer = _get_review_llm()

    hyp_text = "\n\n".join(
        f"[{i + 1}] {h.hypothesis}\n    Rationale: {h.rationale}"
        for i, h in enumerate(hypotheses)
    )

    system_prompt = (
        "You are a senior peer reviewer for quantitative trading research.\n"
        "For each hypothesis below, score it on four dimensions (1-10 each):\n"
        "  - relevance: How relevant to the trading model's observed behaviour?\n"
        "  - testability: How concretely can this be tested with available data?\n"
        "  - specificity: How specific, precise, and actionable is it?\n"
        "  - novelty: How novel and non-obvious is the insight?\n"
        "Then give a verdict: KEEP (worth investigating) or DROP (weak/vague/untestable).\n"
        "Be rigorous but fair. A hypothesis doesn't need perfect scores — it needs to be\n"
        "worth investigating.\n"
    )
    user_prompt = (
        f"Review the following {len(hypotheses)} hypotheses:\n\n"
        f"{hyp_text}\n\n"
        f"Score each one and provide your verdict."
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        structured = reviewer.with_structured_output(PeerReviewOutput)
        result: PeerReviewOutput = structured.invoke(messages)
    except Exception as e:
        logger.warning(f"Peer review structured output failed: {e}. Keeping all hypotheses.")
        return hypotheses, {"error": str(e)}

    kept: list[StructuredHypothesis] = []
    stats: dict[str, list] = {"kept": [], "dropped": []}

    for review in result.reviews:
        avg = (review.relevance + review.testability + review.specificity + review.novelty) / 4.0
        idx = review.index - 1  # 0-based
        if not (0 <= idx < len(hypotheses)):
            continue
        entry = {
            "index": review.index,
            "avg": round(avg, 2),
            "relevance": review.relevance,
            "testability": review.testability,
            "specificity": review.specificity,
            "novelty": review.novelty,
            "verdict": review.verdict,
            "reasoning": review.reasoning,
        }
        if review.verdict.upper() == "KEEP" and avg >= min_score:
            kept.append(hypotheses[idx])
            stats["kept"].append(entry)
            logger.debug(f"Review [{idx + 1}] KEEP (avg={avg:.1f})")
        else:
            stats["dropped"].append(entry)
            logger.info(
                f"Review [{idx + 1}] DROP (avg={avg:.1f}, verdict={review.verdict}): "
                f"{hypotheses[idx].hypothesis[:80]}... — {review.reasoning}"
            )

    logger.info(f"Peer review: kept {len(kept)}/{len(hypotheses)} (min_score={min_score}).")
    return kept, stats


# ═══════════════════════════════════════════════════════════════════
#  Phase 4 — Adversarial Defence
# ═══════════════════════════════════════════════════════════════════

def _get_challenger_llm():
    """Return the LLM used for adversarial challenges."""
    temp = cfg("temperatures.adversarial_challenger", 0.3)
    spec: str = cfg("forum.adversarial.challenger", "")
    if spec:
        provider, model = _parse_panelist_spec(spec)
        return get_llm_by_provider(provider, model, temperature=temp)
    return get_llm(temperature=temp)


def _adversarial_defence(
    hypotheses: list[StructuredHypothesis],
) -> tuple[list[StructuredHypothesis], dict]:
    """
    Two-step adversarial process:

    1. A *challenger* LLM attacks every hypothesis (counter-arguments + severity).
    2. A *defender* LLM responds **only** to HIGH-severity challenges.
       - ``survives=false`` → hypothesis eliminated.
       - ``revised_hypothesis`` → hypothesis text is refined.

    Returns ``(surviving_hypotheses, adversarial_stats)``.
    """
    if not hypotheses:
        return [], {}

    # ── Step 1: generate challenges ─────────────────────────────────
    challenger = _get_challenger_llm()

    hyp_text = "\n\n".join(
        f"[{i + 1}] {h.hypothesis}\n    Rationale: {h.rationale}"
        for i, h in enumerate(hypotheses)
    )

    challenge_sys = (
        "You are a devil's advocate for quantitative trading research.\n"
        "For each hypothesis, find the strongest possible counter-argument, confounding\n"
        "factor, or alternative explanation. Be constructive but rigorous.\n"
        "Rate severity as:\n"
        "  LOW   — minor quibble, does not undermine the core idea.\n"
        "  MEDIUM — worth noting, but the hypothesis may still hold.\n"
        "  HIGH  — fundamental flaw that could invalidate the hypothesis.\n"
    )
    challenge_usr = (
        f"Challenge the following {len(hypotheses)} hypotheses:\n\n{hyp_text}"
    )

    try:
        structured_ch = challenger.with_structured_output(ChallengesOutput)
        challenges: ChallengesOutput = structured_ch.invoke([
            SystemMessage(content=challenge_sys),
            HumanMessage(content=challenge_usr),
        ])
    except Exception as e:
        logger.warning(f"Adversarial challenge failed: {e}. Keeping all hypotheses.")
        return hypotheses, {"error_challenge": str(e)}

    high_challenges = [c for c in challenges.challenges if c.severity.upper() == "HIGH"]
    adv_stats: dict = {
        "total_challenges": len(challenges.challenges),
        "high_severity": len(high_challenges),
    }

    if not high_challenges:
        logger.info("Adversarial: no HIGH-severity challenges. All hypotheses survive.")
        adv_stats["outcome"] = "all_survived"
        return hypotheses, adv_stats

    # ── Step 2: defend against HIGH challenges ──────────────────────
    defender = _get_review_llm(temperature=cfg("temperatures.adversarial_defender", 0.2))

    defence_text = "\n\n".join(
        f"[{c.index}] Hypothesis: "
        f"{hypotheses[c.index - 1].hypothesis if 0 < c.index <= len(hypotheses) else '?'}\n"
        f"    Challenge ({c.severity}): {c.challenge}"
        for c in high_challenges
    )

    defence_sys = (
        "You are defending hypotheses against adversarial challenges.\n"
        "For each challenged hypothesis decide:\n"
        "  - Does it survive the challenge? (survives: true/false)\n"
        "  - If it can be improved, provide a revised_hypothesis. Otherwise null.\n"
        "  - If the challenge is fatal, explain why it should be dropped.\n"
        "Be honest — do NOT defend a hypothesis if the challenge is valid.\n"
    )

    try:
        structured_def = defender.with_structured_output(DefenceOutput)
        defence: DefenceOutput = structured_def.invoke([
            SystemMessage(content=defence_sys),
            HumanMessage(content=f"Defend against these challenges:\n\n{defence_text}"),
        ])
    except Exception as e:
        logger.warning(f"Adversarial defence failed: {e}. Keeping all hypotheses.")
        adv_stats["error_defence"] = str(e)
        return hypotheses, adv_stats

    # ── Apply verdicts ──────────────────────────────────────────────
    dropped_indices: set[int] = set()
    revisions: dict[int, str] = {}

    for v in defence.verdicts:
        idx = v.index - 1
        if not (0 <= idx < len(hypotheses)):
            continue
        if not v.survives:
            dropped_indices.add(idx)
            logger.info(
                f"Adversarial DROP [{idx + 1}]: "
                f"{hypotheses[idx].hypothesis[:80]}... — {v.reasoning}"
            )
        elif v.revised_hypothesis:
            revisions[idx] = v.revised_hypothesis
            logger.info(f"Adversarial REVISE [{idx + 1}]: {v.revised_hypothesis[:80]}...")

    result: list[StructuredHypothesis] = []
    for i, h in enumerate(hypotheses):
        if i in dropped_indices:
            continue
        if i in revisions:
            h = StructuredHypothesis(
                hypothesis=revisions[i],
                rationale=h.rationale,
                key_factors=h.key_factors,
                possible_tests=h.possible_tests,
            )
        result.append(h)

    adv_stats.update({
        "dropped": len(dropped_indices),
        "revised": len(revisions),
        "survived": len(result),
    })
    logger.info(
        f"Adversarial defence: {len(result)}/{len(hypotheses)} survived "
        f"({len(dropped_indices)} dropped, {len(revisions)} revised)."
    )
    return result, adv_stats


# ═══════════════════════════════════════════════════════════════════
#  Main node function (drop-in replacement for hypothesis_maker)
# ═══════════════════════════════════════════════════════════════════

def hypothesis_forum(state: AgentState):
    """
    Multi-panelist hypothesis generation with deduplication,
    peer review, and adversarial defence.

    1. Gets the list of panelists and generates hypotheses **in parallel**.
    2. Merges and deduplicates.
    3. Peer-reviews (if enabled) — drops low-quality hypotheses.
    4. Adversarial defence (if enabled) — stress-tests survivors.
    5. Returns the same output shape as ``hypothesis_maker``.
    """
    logger.info("Starting Hypothesis Forum node...")

    panelists = get_forum_panelists(
        temperature=cfg("temperatures.hypothesis_forum", 0.4),
        max_tokens=cfg("forum.max_tokens", 8192),
    )
    logger.info(f"Forum: {len(panelists)} panelist(s) registered: {[l for l, _ in panelists]}")

    system_prompt, user_prompt = _build_prompts(state)

    # ── Phase 1: parallel hypothesis generation ─────────────────────
    all_hypotheses: list[StructuredHypothesis] = []
    panelist_stats: dict[str, int] = {}
    panelist_hypotheses: dict[str, list[StructuredHypothesis]] = {}

    with ThreadPoolExecutor(max_workers=len(panelists)) as executor:
        futures = {
            executor.submit(_call_panelist, label, llm, system_prompt, user_prompt): label
            for label, llm in panelists
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                panelist_stats[label] = len(result)
                panelist_hypotheses[label] = result
                all_hypotheses.extend(result)
            except Exception as e:
                logger.error(f"Forum panelist [{label}] raised exception: {e}")
                panelist_stats[label] = 0
                panelist_hypotheses[label] = []

    logger.info(
        f"Forum Phase 1 complete: {len(all_hypotheses)} total hypotheses from {len(panelists)} panelist(s). "
        f"Breakdown: {panelist_stats}"
    )

    # ── Phase 2: deduplication ──────────────────────────────────────
    unique_hypotheses = _deduplicate(all_hypotheses, threshold=_SIMILARITY_THRESHOLD)

    # ── Phase 2b: apply configurable cap ────────────────────────────
    if len(unique_hypotheses) > _MAX_HYPOTHESES:
        logger.info(
            f"Capping hypotheses from {len(unique_hypotheses)} to {_MAX_HYPOTHESES} "
            f"(forum.max_hypotheses config)."
        )
        unique_hypotheses = unique_hypotheses[:_MAX_HYPOTHESES]

    # ── Phase 3: peer review (if enabled) ───────────────────────────
    review_stats: dict = {}
    pre_review = len(unique_hypotheses)
    if cfg("forum.peer_review.enabled", True):
        min_score = float(cfg("forum.peer_review.min_score", 6.0))
        unique_hypotheses, review_stats = _peer_review(unique_hypotheses, min_score)
        logger.info(f"Phase 3 (peer review): {pre_review} → {len(unique_hypotheses)}")

    # ── Phase 4: adversarial defence (if enabled) ───────────────────
    adversarial_stats: dict = {}
    pre_defence = len(unique_hypotheses)
    if cfg("forum.adversarial.enabled", False):
        unique_hypotheses, adversarial_stats = _adversarial_defence(unique_hypotheses)
        logger.info(f"Phase 4 (adversarial): {pre_defence} → {len(unique_hypotheses)}")

    # ── Format for state ────────────────────────────────────────────
    hypotheses_list = [
        _format_hypothesis(i + 1, h) for i, h in enumerate(unique_hypotheses)
    ]

    logger.info(f"Forum output: {len(hypotheses_list)} hypotheses (from {len(all_hypotheses)} raw, cap={_MAX_HYPOTHESES}).")

    # ── Log ─────────────────────────────────────────────────────────
    step_logger = get_step_logger()
    if step_logger:
        # Per-panelist hypothesis files
        for label, hyps in panelist_hypotheses.items():
            step_logger.log_panelist_hypotheses(label, [
                {
                    "hypothesis": h.hypothesis,
                    "rationale": h.rationale,
                    "key_factors": h.key_factors,
                    "possible_tests": h.possible_tests,
                }
                for h in hyps
            ])

        # Final merged hypotheses
        step_logger.log_hypotheses(hypotheses_list)

        # Peer-review log
        if review_stats:
            step_logger.log_peer_review(review_stats, pre_review, len(unique_hypotheses))

        # Adversarial-defence log
        if adversarial_stats:
            step_logger.log_adversarial(adversarial_stats, pre_defence, len(unique_hypotheses))

        step_logger.log_step("hypothesis_forum", {
            "panelists": list(panelist_stats.keys()),
            "per_panelist": panelist_stats,
            "raw_count": len(all_hypotheses),
            "unique_count": len(unique_hypotheses),
            "dedup_threshold": _SIMILARITY_THRESHOLD,
            "max_hypotheses_cap": _MAX_HYPOTHESES,
            "peer_review": review_stats,
            "adversarial": adversarial_stats,
        })

    summary_msg = HumanMessage(
        content=f"Hypothesis Forum generated {len(hypotheses_list)} unique hypotheses "
        f"(from {len(all_hypotheses)} raw across {len(panelists)} panelists):\n\n"
        + "\n\n".join(hypotheses_list)
    )
    return {"hypotheses": hypotheses_list, "messages": [summary_msg]}
