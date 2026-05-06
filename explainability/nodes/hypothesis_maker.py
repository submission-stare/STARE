from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import pandas as pd
import re
from state import AgentState
from utils.config import cfg
from utils.data_scope import scoped_walk_roots
from utils.llm import get_llm
from utils.step_logger import get_step_logger
import logging

logger = logging.getLogger(__name__)


# ── Structured hypothesis schema ────────────────────────────────────
class StructuredHypothesis(BaseModel):
    """A single structured hypothesis about agent trading behaviour."""
    hypothesis: str = Field(description="A clear, concise statement of the hypothesis.")
    rationale: str = Field(description="Why this hypothesis is plausible given the data.")
    key_factors: list[str] = Field(description="The 2-4 key data factors / indicators that support or relate to this hypothesis.")
    possible_tests: list[str] = Field(description="Concrete, data-driven tests or analyses that could confirm or refute this hypothesis.")


class HypothesesOutput(BaseModel):
    """Between 1 and 20 structured hypotheses."""
    hypotheses: list[StructuredHypothesis] = Field(
        description="Between 1 and 20 structured hypotheses about agent trading behaviour.",
    )


# ── Peer Review schemas (Sprint 3) ──────────────────────────────────
class HypothesisReview(BaseModel):
    """Peer-review scoring of a single hypothesis."""
    index: int = Field(description="1-based index of the hypothesis being reviewed.")
    relevance: int = Field(description="Score 1-10: How relevant to the trading model's observed behaviour.", ge=1, le=10)
    testability: int = Field(description="Score 1-10: How concretely this can be tested with available data.", ge=1, le=10)
    specificity: int = Field(description="Score 1-10: How specific, precise, and actionable the hypothesis is.", ge=1, le=10)
    novelty: int = Field(description="Score 1-10: How novel and non-obvious the insight is.", ge=1, le=10)
    verdict: str = Field(description="KEEP or DROP.")
    reasoning: str = Field(description="One-sentence justification for the verdict.")


class PeerReviewOutput(BaseModel):
    """Batch of peer reviews — one entry per hypothesis."""
    reviews: list[HypothesisReview] = Field(
        description="Exactly one review per hypothesis, in the same order as the input.",
    )


# ── Adversarial Defence schemas (Sprint 3) ──────────────────────────
class AdversarialChallenge(BaseModel):
    """Devil's-advocate challenge to a single hypothesis."""
    index: int = Field(description="1-based index of the hypothesis being challenged.")
    challenge: str = Field(description="A strong counter-argument, confounding factor, or alternative explanation.")
    severity: str = Field(description="LOW, MEDIUM, or HIGH — how fundamental is the flaw.")


class ChallengesOutput(BaseModel):
    """Batch of adversarial challenges."""
    challenges: list[AdversarialChallenge] = Field(
        description="One challenge per hypothesis.",
    )


class DefenceVerdict(BaseModel):
    """Defence response to a single adversarial challenge."""
    index: int = Field(description="1-based index of the hypothesis.")
    survives: bool = Field(description="True if the hypothesis withstands the challenge.")
    revised_hypothesis: str | None = Field(
        default=None,
        description="Refined hypothesis text if the challenge led to improvement. Null if unchanged.",
    )
    reasoning: str = Field(description="Brief explanation of why it survives or should be dropped.")


class DefenceOutput(BaseModel):
    """Batch of defence verdicts."""
    verdicts: list[DefenceVerdict] = Field(
        description="One verdict per challenged hypothesis.",
    )


def _format_hypothesis(idx: int, h: StructuredHypothesis) -> str:
    """Render a structured hypothesis as a readable Markdown-style block."""
    factors = "\n".join(f"   - {f}" for f in h.key_factors)
    tests = "\n".join(f"   - {t}" for t in h.possible_tests)
    return (
        f"### Hypothesis {idx}\n"
        f"**Hypothesis:** {h.hypothesis}\n"
        f"**Rationale:** {h.rationale}\n"
        f"**Key Factors:**\n{factors}\n"
        f"**Possible Tests:**\n{tests}"
    )


def hypothesis_maker(state: AgentState):
    """
    Analyzes the initial metrics and generates 3 clear, structured diagnostic hypotheses.
    """
    logger.info("Starting Hypothesis Maker node...")
    llm = get_llm(temperature=cfg("temperatures.hypothesis_maker", 0.4))
    raw_data_path = state.get("raw_data_path", "")
    
    # Read a sample of the data to provide context to the LLM
    try:
        import os
        if os.path.isdir(raw_data_path):
            # Try to build a directory tree and inspect a top level summary file
            tree = []
            summary_scanned = ""
            for walk_root in scoped_walk_roots(raw_data_path):
                for root, dirs, files in os.walk(walk_root):
                    level = root.replace(raw_data_path, '').count(os.sep)
                    indent = ' ' * 4 * (level)
                    tree.append(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        tree.append(f"{subindent}{f}")
                        # If we find a known summary file, let's include its content
                        if f in ["advanced_stats_summary.csv", "statistical_summary.csv"] and not summary_scanned:
                            try:
                                df = pd.read_csv(os.path.join(root, f))
                                summary_scanned = f"Summary File ({f}) Head:\n{df.head().to_string()}\n"
                            except Exception:
                                pass
            if not tree:
                raise ValueError(
                    f"Configured data scope directories were not found under {raw_data_path}."
                )
            
            tree_str = "\n".join(tree[:100]) # Cap tree to 100 lines to avoid blowing context
            if len(tree) > 100:
                tree_str += "\n... (truncated)"
            
            data_sample = f"Data Directory Structure:\n{tree_str}\n\n{summary_scanned}"
            
        elif raw_data_path.endswith('.csv'):
            df = pd.read_csv(raw_data_path)
            data_sample = f"Data Head:\n{df.head().to_string()}\n\nData Description:\n{df.describe().to_string()}"
        elif raw_data_path.endswith('.json'):
            df = pd.read_json(raw_data_path)
            data_sample = f"Data Head:\n{df.head().to_string()}\n\nData Description:\n{df.describe().to_string()}"
        else:
            raise ValueError("Unsupported file format or path.")
            
    except Exception as e:
        data_sample = f"Could not load data from {raw_data_path}: {str(e)}"

    # Read documentation to provide business context
    docs_context = ""
    try:
        docs_path = os.path.join(os.getcwd(), "specification", "documentation.md")
        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                docs_context = f.read()
    except Exception as e:
        logger.warning(f"Could not load documentation: {e}")

    # ── Include transaction-level summaries if available ──────────
    tx_summary = state.get("transaction_summary", "")
    sn_summary = state.get("snapshot_summary", "")

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

    # ── Inject existing hypotheses to prevent duplication ────────
    existing_hypotheses = state.get("hypotheses", [])
    
    user_prompt = f"Here is a sample of the raw performance metrics:\n\n{data_sample}\n\n"
    if tx_summary:
        user_prompt += f"Transaction-level Data:\n{tx_summary}\n\n"
    if sn_summary:
        user_prompt += f"Portfolio Snapshot Data:\n{sn_summary}\n\n"

    if existing_hypotheses:
        user_prompt += (
            "IMPORTANT — The following hypotheses have ALREADY been generated. "
            "Do NOT repeat or reformulate any of them. Generate only NEW, substantially different hypotheses.\n\n"
            "=== EXISTING HYPOTHESES (DO NOT REPEAT) ===\n"
            + "\n\n".join(existing_hypotheses)
            + "\n=== END EXISTING HYPOTHESES ===\n\n"
        )
        user_prompt += (
            f"Generate new hypotheses that are different from the {len(existing_hypotheses)} above. "
            "Produce as many novel ones as are relevant (up to 20 total including the existing ones)."
        )
    else:
        user_prompt += "Please generate between 1 and 20 hypotheses (as many as are relevant, up to 20)."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Use structured output to force the LLM to fill the hypothesis form
    structured_llm = llm.with_structured_output(HypothesesOutput)

    try:
        result: HypothesesOutput = structured_llm.invoke(messages)
        result.hypotheses = result.hypotheses[:20]
        # Strip echoed "Hypothesis:" prefix the LLM may add to the hypothesis field
        for h in result.hypotheses:
            h.hypothesis = re.sub(r"^\s*Hypothesis\s*:\s*", "", h.hypothesis, count=1, flags=re.IGNORECASE).strip()
        # Format each structured hypothesis as a readable text block
        hypotheses_list = [
            _format_hypothesis(i + 1, h) for i, h in enumerate(result.hypotheses)
        ]
    except Exception as e:
        # Fallback: if structured output fails, use plain text
        logger.warning(f"Structured output failed, falling back to plain text: {e}")
        response = llm.invoke(messages)
        hypotheses_list = [h.strip() for h in response.content.split('\n') if h.strip()]
    
    logger.info(f"Generated {len(hypotheses_list)} hypotheses.")
    logger.debug(f"Hypotheses content: {hypotheses_list}")
    
    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_hypotheses(hypotheses_list)
    
    summary_msg = HumanMessage(content="Generated Hypotheses:\n\n" + "\n\n".join(hypotheses_list))
    return {"hypotheses": hypotheses_list, "messages": [summary_msg]}
