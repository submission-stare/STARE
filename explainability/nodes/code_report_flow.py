from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import AgentState
from utils.code_report import (
    build_code_context_bundle,
    build_code_context_bundle_from_report_data,
    summarize_code_evidence_results,
    summarize_enriched_claims,
)
from utils.config import cfg
from utils.forum_engine import run_forum
from utils.llm import _parse_panelist_spec, get_llm, get_llm_by_provider
from utils.step_logger import get_step_logger

logger = logging.getLogger(__name__)

CODE_ROLES = [
    "Architecture Cartographer",
    "Environment and Risk Auditor",
    "Training and Optimization Auditor",
    "Metrics and Validation Auditor",
]
CODE_EXPLAINER_ROLE = "Code Explainer"
ROLE_DEFAULTS = {
    "Architecture Cartographer": {
        "hypothesis": "The supplied benchmark config and the reachable training and evaluation modules drive most behavior, while benchmark.main mainly coordinates the flow between those components.",
        "rationale": "The current code path matters because it decides where configuration is consumed and where training, environment construction, and evaluation happen, but behavior should be explained through those downstream modules rather than by overemphasizing generic orchestration code.",
        "key_factors": [
            "The supplied benchmark config is loaded once and then threaded into worker and pipeline code",
            "benchmark.main coordinates config loading, regime stitching, worker execution, and aggregation",
            "Worker and pipeline modules bind the environment, training, and evaluation behavior that the report is trying to explain",
        ],
        "possible_verification": [
            "Trace how the supplied config values move from the entry module into the worker and agent pipeline helpers",
            "Confirm which reachable modules actually consume behavior-relevant settings instead of assuming all meaning sits in benchmark.main",
        ],
        "suggested_changes": [
            {"category": "architecture", "change": "Store dependency-trace snapshots with each code report so explanations remain tied to the current import graph."},
            {"category": "code", "change": "Expose a compact runtime wiring summary that highlights config consumers and behavior-relevant modules instead of generic entrypoint facts."},
        ],
        "confidence": "high",
    },
    "Environment and Risk Auditor": {
        "hypothesis": "Risk behavior in this portfolio-trading benchmark is primarily governed by the supplied config values for environment construction, such as allow-short, hmax, costs, reward scaling, and initial allocation, while explicit liquidation or margin-call safeguards are not obvious in the reachable local code.",
        "rationale": "The benchmark uses the supplied config to build TradingEnvConfig and env_kwargs for a financial portfolio environment, so risk explanations should focus on those environment controls and on whether the reachable local path adds stronger safeguards.",
        "key_factors": [
            "TradingEnvConfig forwards several parameters from the supplied config",
            "Worker code builds env_kwargs for portfolio training and evaluation from those config-driven settings",
            "Risk-control keywords are sparse or absent in the reachable local benchmark path",
        ],
        "possible_verification": [
            "Map the env_kwargs construction path from the supplied config into the portfolio-trading environment constructor",
            "Search the reachable local code for trading and algorithmic hooks",
        ],
        "suggested_changes": [
            {"category": "algorithm", "change": "Add explicit leverage, liquidation, or equity-floor controls in the environment rather than relying on emergent policy behavior."},
            {"category": "hyperparameters", "change": "Re-evaluate allow_short, hmax, and transaction-cost defaults under the interleaved stress test."},
        ],
        "confidence": "medium",
    },
    "Training and Optimization Auditor": {
        "hypothesis": "Because these are RL algorithms trading financial portfolios, behavior should be evaluated through each algorithm family and its config-driven optimization setup, not just through generic benchmark wiring.",
        "rationale": "A2C and PPO are on-policy, while DDPG, SAC, and TD3 are off-policy, and the supplied benchmark config drives shared architecture and optimization knobs such as net_arch, lr_scale, batch_size, buffer_size, total_timesteps, plus PPO-specific overrides. Those choices can materially shape portfolio behavior.",
        "key_factors": [
            "build_policy_kwargs applies one architecture spec across RL algorithms unless explicitly overridden",
            "build_scaled_model_kwargs maps supplied config values into optimizer and replay-related settings",
            "The benchmark config contains PPO-specific overrides alongside global settings for a portfolio-trading RL setup",
        ],
        "possible_verification": [
            "Trace net_arch, lr_scale, batch_size, buffer_size, and total_timesteps from the supplied YAML into worker helpers and RL model builders",
            "Compare shared model kwargs with algorithm-specific overrides while interpreting them in light of A2C, PPO, DDPG, SAC, and TD3 behavior",
        ],
        "suggested_changes": [
            {"category": "code", "change": "Separate global defaults from algorithm-specific overrides in the config trace shown to users, especially for RL portfolio-training settings."},
            {"category": "hyperparameters", "change": "Tune per-algorithm learning and batch settings instead of relying on one global scaling layer for all agents."},
        ],
        "confidence": "high",
    },
    "Metrics and Validation Auditor": {
        "hypothesis": "Interpretation of report claims depends heavily on the metrics pipeline configured for this benchmark run, because Sharpe, final-value aggregation, and cross-run summaries are computed downstream from the worker in dedicated evaluation and analysis modules.",
        "rationale": "The supplied benchmark config determines the run shape, while the reachable evaluation path converts account values into metrics, so conclusions about success, failure, or anomalous Sharpe behavior should be grounded in that code path instead of in generic benchmark-level observations.",
        "key_factors": [
            "Worker imports MetricsEvaluator and PortfolioAnalyzer for evaluation",
            "The entry module imports aggregate results from the analysis module",
            "Report claims about Sharpe and final value map to the traced evaluation path for the supplied run configuration",
        ],
        "possible_verification": [
            "Trace the metric path from account_values.csv generation to MetricsEvaluator and aggregate_results",
            "Audit how negative-equity scenarios are represented in downstream metrics code for the supplied benchmark setup",
        ],
        "suggested_changes": [
            {"category": "architecture", "change": "Surface metric provenance explicitly in generated reports so conclusions can be tied to the exact evaluation path."},
            {"category": "code", "change": "Add guardrails or annotations for negative-equity metric interpretation in reporting outputs."},
        ],
        "confidence": "high",
    },
}


class SuggestedChange(BaseModel):
    category: Literal["architecture", "code", "algorithm", "hyperparameters"]
    change: str


class StructuredCodeHypothesis(BaseModel):
    role: str
    hypothesis: str
    rationale: str
    key_factors: list[str]
    possible_verification: list[str]
    code_evidence_refs: list[str]
    confidence: Literal["low", "medium", "high"]
    suggested_changes: list[SuggestedChange]


class CodeHypothesesOutput(BaseModel):
    hypotheses: list[StructuredCodeHypothesis]


class CodeHypothesisReview(BaseModel):
    index: int = Field(ge=1)
    relevance: int = Field(ge=1, le=10)
    evidence_grounding: int = Field(ge=1, le=10)
    actionability: int = Field(ge=1, le=10)
    verdict: Literal["KEEP", "DROP"]
    reasoning: str


class CodePeerReviewOutput(BaseModel):
    reviews: list[CodeHypothesisReview]


class CodeClaimContext(BaseModel):
    claim_id: str
    title: str
    kind: str
    claim_text: str
    code_paths: list[str]
    config_keys: list[str]
    explanation: str
    exercised_flow: str


class CodeClaimContextOutput(BaseModel):
    claims: list[CodeClaimContext]


class CodeHypothesisEvidence(BaseModel):
    task_id: str
    title: str
    objective: str
    summary: str
    supporting_paths: list[str]
    config_keys: list[str]
    evidence_snippets: list[str]
    confidence: Literal["low", "medium", "high"]


class CodeHypothesisEvidenceOutput(BaseModel):
    results: list[CodeHypothesisEvidence]


def _format_code_hypothesis(idx: int, hypothesis: StructuredCodeHypothesis) -> str:
    key_factors = "\n".join(f"   - {item}" for item in hypothesis.key_factors)
    verification = "\n".join(f"   - {item}" for item in hypothesis.possible_verification)
    evidence = "\n".join(f"   - {item}" for item in hypothesis.code_evidence_refs)
    changes = "\n".join(
        f"   - [{item.category}] {item.change}" for item in hypothesis.suggested_changes
    )
    return (
        f"### Hypothesis {idx}\n"
        f"**Role:** {hypothesis.role}\n"
        f"**Hypothesis:** {hypothesis.hypothesis}\n"
        f"**Rationale:** {hypothesis.rationale}\n"
        f"**Key Factors:**\n{key_factors}\n"
        f"**Possible Verification:**\n{verification}\n"
        f"**Code Evidence Refs:**\n{evidence}\n"
        f"**Confidence:** {hypothesis.confidence}\n"
        f"**Suggested Changes:**\n{changes}"
    )


def _safe_panelists() -> list[tuple[str, object, str]]:
    specs = cfg("code_forum.panelists", cfg("forum.panelists", [])) or []
    roles = cfg("code_forum.roles", CODE_ROLES) or CODE_ROLES
    panelists: list[tuple[str, object, str]] = []
    for idx, role in enumerate(roles):
        spec = specs[idx % len(specs)] if specs else ""
        if not spec:
            continue
        try:
            provider, model = _parse_panelist_spec(spec)
            llm = get_llm_by_provider(
                provider=provider,
                model=model,
                temperature=cfg("temperatures.code_hypothesis_forum", 0.2),
                max_tokens=cfg("code_forum.max_tokens", 8192),
            )
            panelists.append((role, llm, f"{provider}/{model}"))
        except Exception as exc:
            logger.warning("Skipping code forum panelist %s: %s", role, exc)
    return panelists


def _code_explainer_llm() -> tuple[object, str]:
    spec = cfg("code_forum.explainer.model", "")
    if not spec:
        specs = cfg("code_forum.panelists", cfg("forum.panelists", [])) or []
        spec = specs[0] if specs else ""
    if not spec:
        raise ValueError("No code explainer model configured in code_forum.explainer.model or code_forum.panelists.")
    provider, model = _parse_panelist_spec(spec)
    llm = get_llm_by_provider(
        provider=provider,
        model=model,
        temperature=cfg("code_forum.explainer.temperature", cfg("temperatures.code_hypothesis_forum", 0.1)),
        max_tokens=cfg("code_forum.explainer.max_tokens", cfg("code_forum.max_tokens", 8192)),
    )
    return llm, f"{provider}/{model}"


def _context_summary(bundle: dict) -> str:
    dependency_graph = bundle["dependency_graph"]
    config_traces = bundle["config_traces"]
    interesting_keys = {
        key: config_traces["traces"].get(key, [])[:2]
        for key in ("allow_short", "net_arch", "reward_scaling", "relative_return_alpha", "total_timesteps")
    }
    return (
        f"Report path: {bundle['report_path']}\n"
        f"Benchmark entry: {bundle['benchmark_entry']}\n"
        f"Benchmark config (source of truth; ignore sibling configs): {bundle['benchmark_config']}\n"
        f"Behavior-relevant reachable modules: {len(dependency_graph['modules'])}\n"
        f"Supplemental placeholder files: {len(bundle.get('supplemental_code_placeholders', []))}\n"
        f"Focus config traces: {json.dumps(interesting_keys, indent=2)}\n"
        f"Report claims: {len(bundle['report_data']['claims'])}\n"
    )


def _extract_markdown_field(block: str, field: str) -> str:
    match = re.search(
        rf"\*\*{re.escape(field)}:\*\*\s*(.*?)(?=\n\*\*[^\n]+:\*\*|\Z)",
        block,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_markdown_list(block: str, field: str) -> list[str]:
    content = _extract_markdown_field(block, field)
    if not content:
        return []
    items = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    if items:
        return items
    return [re.sub(r"\s+", " ", content).strip()] if content else []


def _build_live_report_claims(state: AgentState) -> list[dict]:
    claims: list[dict] = []
    for idx, hypothesis in enumerate(state.get("hypotheses", []), start=1):
        statement = _extract_markdown_field(hypothesis, "Hypothesis")
        rationale = _extract_markdown_field(hypothesis, "Rationale")
        claims.append(
            {
                "id": f"hypothesis_{idx}",
                "kind": "hypothesis",
                "title": f"Hypothesis {idx}",
                "text": statement or rationale or hypothesis,
            }
        )

    tests = state.get("investigation_tests", [])
    if tests:
        claims.append(
            {
                "id": "tests_and_results",
                "kind": "tests_results",
                "title": "Tests and Results",
                "text": "\n".join(f"{idx}. {test}" for idx, test in enumerate(tests, start=1)),
            }
        )

    for key, value in state.get("consensus_answers", {}).items():
        if value:
            claims.append(
                {
                    "id": f"consensus_{key}",
                    "kind": "consensus",
                    "title": key,
                    "text": value,
                }
            )
    return claims


def _build_live_report_data(state: AgentState) -> dict:
    claims = _build_live_report_claims(state)
    hypotheses = []
    for idx, hypothesis in enumerate(state.get("hypotheses", []), start=1):
        hypotheses.append(
            {
                "id": f"hypothesis_{idx}",
                "title": f"Hypothesis {idx}",
                "statement": _extract_markdown_field(hypothesis, "Hypothesis"),
                "body": hypothesis,
            }
        )
    consensus_sections = {
        key: value
        for key, value in state.get("consensus_answers", {}).items()
        if value
    }
    return {
        "title": "Integrated Trading Agent Report",
        "hypotheses": hypotheses,
        "tests_and_results": "\n".join(state.get("investigation_tests", [])),
        "behavior_analysis": "\n\n".join(consensus_sections.values()),
        "consensus_sections": consensus_sections,
        "claims": claims,
        "raw_text": "",
    }


def _bundle_with_report_data(bundle: dict, report_data: dict) -> dict:
    updated = dict(bundle)
    updated["report_data"] = report_data
    return updated


def _claim_summary(claim: dict) -> str:
    text = re.sub(r"\s+", " ", claim.get("text", "")).strip()
    if len(text) > 500:
        text = text[:497] + "..."
    return (
        f"- {claim.get('id')} [{claim.get('kind')}]: {claim.get('title')}\n"
        f"  {text}"
    )


def _module_source_excerpt(path: str, max_chars: int = 2200) -> str:
    if not path:
        return "(source unavailable)"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read().strip()
    except OSError:
        return "(source unavailable)"
    if len(source) <= max_chars:
        return source
    head = source[: max_chars // 2].rstrip()
    tail = source[-max_chars // 3 :].lstrip()
    return f"{head}\n# ... truncated ...\n{tail}"


def _format_prompt_source_block(path_label: str, source: str, heading: str | None = None) -> str:
    lines = []
    if heading:
        lines.append(heading)
    lines.extend(
        [
            f"BEGIN SOURCE FILE: {path_label}",
            f"Path: {path_label}",
            "Source:",
            "```python",
            source,
            "```",
            f"END SOURCE FILE: {path_label}",
        ]
    )
    return "\n".join(lines)


def _build_code_explainer_context(bundle: dict) -> str:
    graph = bundle["dependency_graph"]
    placeholder_sources = graph.get("placeholder_sources", {})
    rel = lambda path: os.path.relpath(path, os.getcwd())

    modules = []
    for module in graph["modules"]:
        outgoing = [
            edge for edge in graph["edges"]
            if edge["from"] == module["module"]
        ][:8]
        outgoing_text = ", ".join(
            f"{edge['to']} ({edge['kind']})"
            for edge in outgoing
        ) or "(none)"
        symbols = ", ".join(list(module["symbol_defs"].keys())[:12]) or "(none)"
        module_path = rel(module["path"])
        source = _module_source_excerpt(module["path"])
        placeholder_label = module.get("source_path") or "(no)"
        placeholder_heading = f"Placeholder source: {placeholder_label}"
        if module["path"] in placeholder_sources:
            source = placeholder_sources[module["path"]].strip()
            placeholder_heading = f"Placeholder source copied from: {module.get('source_path')}"
        modules.append(
            _format_prompt_source_block(
                module_path,
                source,
                heading="\n".join(
                    [
                        f"### Module: {module['module']}",
                        f"Path: {module_path}",
                        f"Docstring: {module['docstring'] or '(none)'}",
                        f"Top-level symbols: {symbols}",
                        f"Outgoing edges: {outgoing_text}",
                        placeholder_heading,
                    ]
                ),
            )
        )

    claims = "\n\n".join(_claim_summary(claim) for claim in bundle["report_data"]["claims"])
    config_traces = json.dumps(bundle["config_traces"]["traces"], indent=2)
    symbol_refs = json.dumps(graph["symbol_refs"][:80], indent=2)
    modules_text = "\n\n".join(modules)
    return (
        "You are given the parsed report claims and the reachable Python code tree.\n\n"
        "=== REPORT CLAIMS ===\n"
        f"{claims}\n\n"
        "=== DEPENDENCY GRAPH OVERVIEW ===\n"
        f"Entry module: {graph['entry_module']}\n"
        f"Entry path: {rel(graph['entry_path'])}\n"
        f"Reachable modules: {len(graph['modules'])}\n"
        f"External dependencies: {', '.join(graph['external_dependencies']) or '(none)'}\n\n"
        "=== SUPPLEMENTAL PLACEHOLDER FILES ===\n"
        f"{json.dumps(bundle.get('supplemental_code_placeholders', []), indent=2)}\n\n"
        "=== RESOLVED SYMBOL REFERENCES ===\n"
        f"{symbol_refs}\n\n"
        "=== CONFIG TRACES ===\n"
        f"{config_traces}\n\n"
        "=== REACHABLE MODULES ===\n"
        f"{modules_text}"
    )


def _format_code_claim_context(idx: int, item: CodeClaimContext) -> str:
    code_paths = "\n".join(f"   - {path}" for path in item.code_paths) or "   - (none)"
    config_keys = "\n".join(f"   - {key}" for key in item.config_keys) or "   - (none)"
    return (
        f"### Claim {idx}: {item.title}\n"
        f"**Claim ID:** {item.claim_id}\n"
        f"**Kind:** {item.kind}\n"
        f"**Claim Text:** {item.claim_text}\n"
        f"**Relevant Code Paths:**\n{code_paths}\n"
        f"**Relevant Config Keys:**\n{config_keys}\n"
        f"**Exercised Flow:** {item.exercised_flow}\n"
        f"**Why this code matters:** {item.explanation}"
    )


def _fallback_code_claim_contexts(bundle: dict) -> list[CodeClaimContext]:
    graph = bundle["dependency_graph"]
    rel_paths = [
        os.path.relpath(module["path"], os.getcwd())
        for module in graph["modules"][:10]
    ]
    config_keys = bundle["config_traces"]["keys"][:12]
    return [
        CodeClaimContext(
            claim_id=claim["id"],
            title=claim["title"],
            kind=claim["kind"],
            claim_text=claim["text"],
            code_paths=rel_paths,
            config_keys=config_keys,
            explanation=(
                "LLM-based claim-to-code enrichment was unavailable, so this claim is paired with the "
                "reachable benchmark code tree and benchmark config trace as broad context."
            ),
            exercised_flow=(
                "entry module -> reachable worker / agent pipeline modules -> analysis modules "
                "reachable from the traced import graph"
            ),
        )
        for claim in bundle["report_data"]["claims"]
    ]


def _fallback_hypotheses(bundle: dict) -> list[StructuredCodeHypothesis]:
    entry = os.path.relpath(bundle["benchmark_entry"], os.getcwd())
    config = os.path.relpath(bundle["benchmark_config"], os.getcwd())
    module_paths = [
        os.path.relpath(m["path"], os.getcwd())
        for m in bundle["dependency_graph"]["modules"][:6]
    ]
    hypotheses: list[StructuredCodeHypothesis] = []
    for role in cfg("code_forum.roles", CODE_ROLES) or CODE_ROLES:
        template = ROLE_DEFAULTS[role]
        evidence_refs = [entry, config] + module_paths[:2]
        hypotheses.append(
            StructuredCodeHypothesis(
                role=role,
                hypothesis=template["hypothesis"],
                rationale=template["rationale"],
                key_factors=template["key_factors"],
                possible_verification=template["possible_verification"],
                code_evidence_refs=evidence_refs,
                confidence=template["confidence"],
                suggested_changes=[SuggestedChange(**item) for item in template["suggested_changes"]],
            )
        )
    return hypotheses


def _resolve_repo_path(bundle: dict, path: str) -> str | None:
    if not path:
        return None
    placeholder_sources = {
        **bundle.get("code_dependency_graph", {}).get("placeholder_sources", {}),
        **bundle.get("dependency_graph", {}).get("placeholder_sources", {}),
    }
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.extend(
            [
                os.path.join(os.getcwd(), path),
                os.path.join(os.path.dirname(bundle["code_scope_root"]), path),
                os.path.join(bundle["code_scope_root"], path),
            ]
        )
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in placeholder_sources:
            return abs_candidate
        if os.path.exists(candidate):
            return abs_candidate
    return None


def _select_claims_for_hypothesis(hypothesis: dict, enriched_claims: list[dict], max_claims: int = 3) -> list[dict]:
    scored: list[tuple[int, dict]] = []
    hypothesis_text = hypothesis.get("hypothesis", "").lower()
    evidence_refs = set(hypothesis.get("code_evidence_refs", []))
    for claim in enriched_claims:
        score = 0
        claim_text = claim.get("claim_text", "").lower()
        if any(path in evidence_refs for path in claim.get("code_paths", [])):
            score += 4
        if any(key in hypothesis_text for key in claim.get("config_keys", [])):
            score += 2
        if any(word and word in claim_text for word in hypothesis_text.split()[:8]):
            score += 1
        scored.append((score, claim))
    ranked = [claim for score, claim in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:max_claims] or enriched_claims[:max_claims]


def _build_hypothesis_evidence_tasks(hypotheses: list[dict], max_tasks: int) -> list[dict]:
    tasks = []
    for idx, hypothesis in enumerate(hypotheses[:max_tasks], start=1):
        tasks.append(
            {
                "id": f"hypothesis_{idx}_evidence",
                "title": f"Hypothesis {idx} Evidence",
                "task_type": "hypothesis_evidence",
                "objective": hypothesis.get("hypothesis", ""),
                "hypothesis_index": idx,
                "role": hypothesis.get("role", ""),
            }
        )
    return tasks


def _build_evidence_prompt_context(bundle: dict, hypothesis: dict, claims: list[dict]) -> str:
    evidence_paths = []
    for path in hypothesis.get("code_evidence_refs", []):
        if path not in evidence_paths:
            evidence_paths.append(path)
    for claim in claims:
        for path in claim.get("code_paths", []):
            if path not in evidence_paths:
                evidence_paths.append(path)
    evidence_paths = evidence_paths[:5]

    config_keys = []
    for claim in claims:
        for key in claim.get("config_keys", []):
            if key not in config_keys:
                config_keys.append(key)
    config_traces = {
        key: bundle["config_traces"]["traces"].get(key, [])[:2]
        for key in config_keys[:6]
    }

    modules = []
    placeholder_sources = bundle.get("dependency_graph", {}).get("placeholder_sources", {})
    for path in evidence_paths:
        resolved = _resolve_repo_path(bundle, path)
        source = placeholder_sources.get(resolved) if resolved else None
        excerpt = (
            source.strip()
            if source is not None
            else (_module_source_excerpt(resolved, max_chars=1800) if resolved else "(source unavailable)")
        )
        modules.append(
            _format_prompt_source_block(
                path,
                excerpt,
            )
        )

    claim_text = "\n\n".join(_format_code_claim_context(idx, CodeClaimContext(**claim)) for idx, claim in enumerate(claims, start=1))
    modules_text = "\n\n".join(modules) or "(none)"
    return (
        f"Hypothesis: {hypothesis.get('hypothesis', '')}\n"
        f"Role: {hypothesis.get('role', '')}\n"
        f"Rationale: {hypothesis.get('rationale', '')}\n"
        f"Possible verification: {json.dumps(hypothesis.get('possible_verification', []), indent=2)}\n\n"
        "Relevant claims:\n"
        f"{claim_text or '(none)'}\n\n"
        "Relevant config traces:\n"
        f"{json.dumps(config_traces, indent=2)}\n\n"
        "Relevant code excerpts:\n"
        f"{modules_text}"
    )


def _fallback_hypothesis_evidence(bundle: dict, hypotheses: list[dict], enriched_claims: list[dict], max_tasks: int) -> tuple[list[dict], list[dict]]:
    tasks = _build_hypothesis_evidence_tasks(hypotheses, max_tasks=max_tasks)
    results = []
    for task in tasks:
        hypothesis = hypotheses[task["hypothesis_index"] - 1]
        claims = _select_claims_for_hypothesis(hypothesis, enriched_claims)
        supporting_paths = []
        for path in hypothesis.get("code_evidence_refs", []):
            if path not in supporting_paths:
                supporting_paths.append(path)
        for claim in claims:
            for path in claim.get("code_paths", []):
                if path not in supporting_paths:
                    supporting_paths.append(path)
        config_keys = []
        for claim in claims:
            for key in claim.get("config_keys", []):
                if key not in config_keys:
                    config_keys.append(key)
        snippets = []
        for claim in claims:
            snippets.append(f"{claim.get('title', 'Claim')}: {claim.get('explanation', '')}")
        if not snippets:
            snippets.append("Fallback evidence used the hypothesis evidence refs and reachable config traces.")
        results.append(
            {
                "task_id": task["id"],
                "title": task["title"],
                "task_type": task["task_type"],
                "result": {
                    "summary": (
                        "Prompt-based evidence collection was unavailable, so this entry falls back to the "
                        "hypothesis evidence refs, claim-to-code mappings, and config traces gathered earlier."
                    ),
                    "supporting_paths": supporting_paths[:6],
                    "config_keys": config_keys[:6],
                    "evidence_snippets": snippets[:4],
                    "confidence": "medium",
                },
            }
        )
    return tasks, results


def _collect_hypothesis_evidence(bundle: dict, enriched_claims: list[dict], hypotheses: list[dict], max_tasks: int) -> tuple[list[dict], list[dict]]:
    tasks = _build_hypothesis_evidence_tasks(hypotheses, max_tasks=max_tasks)
    if not tasks:
        return [], []
    try:
        llm, label = _code_explainer_llm()
    except Exception as exc:
        logger.warning("Code evidence reviewer unavailable: %s", exc)
        return _fallback_hypothesis_evidence(bundle, hypotheses, enriched_claims, max_tasks)

    prompt_items = []
    for task in tasks:
        hypothesis = hypotheses[task["hypothesis_index"] - 1]
        claims = _select_claims_for_hypothesis(hypothesis, enriched_claims)
        prompt_items.append(
            "\n\n".join(
                [
                    f"Task ID: {task['id']}",
                    f"Title: {task['title']}",
                    f"Objective: {task['objective']}",
                    _build_evidence_prompt_context(bundle, hypothesis, claims),
                ]
            )
        )

    system_prompt = (
        "You are a code-grounded evidence reviewer.\n"
        "Inspect only the supplied code excerpts, claim mappings, and config traces.\n"
        "For each hypothesis, identify the strongest code evidence that supports or constrains it.\n"
        "Do not rely on outside knowledge or files that were not supplied.\n"
        "Keep evidence_snippets short, concrete, and tied to the supplied paths."
    )
    user_prompt = (
        "Return one result per task below.\n"
        "Use the given task_id exactly.\n"
        "supporting_paths must reference only paths present in the supplied context.\n"
        "config_keys must come from the supplied claim mappings or config traces.\n"
        "If the evidence is weak or indirect, say so in the summary.\n\n"
        + "\n\n====\n\n".join(prompt_items)
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    try:
        result: CodeHypothesisEvidenceOutput = llm.with_structured_output(CodeHypothesisEvidenceOutput).invoke(messages)
    except Exception as exc:
        logger.warning("Code evidence reviewer %s failed: %s", label, exc)
        return _fallback_hypothesis_evidence(bundle, hypotheses, enriched_claims, max_tasks)

    by_id = {item.task_id: item for item in result.results}
    evidence_results = []
    for task in tasks:
        item = by_id.get(task["id"])
        if item is None:
            _, fallback_results = _fallback_hypothesis_evidence(
                bundle,
                [hypotheses[task["hypothesis_index"] - 1]],
                enriched_claims,
                max_tasks=1,
            )
            evidence_results.extend(fallback_results)
            continue
        evidence_results.append(
            {
                "task_id": task["id"],
                "title": task["title"],
                "task_type": task["task_type"],
                "result": {
                    "summary": item.summary,
                    "supporting_paths": item.supporting_paths,
                    "config_keys": item.config_keys,
                    "evidence_snippets": item.evidence_snippets,
                    "confidence": item.confidence,
                },
            }
        )
    return tasks, evidence_results


def _claim_context_summary(enriched_claims: list[dict]) -> str:
    summary = summarize_enriched_claims(enriched_claims, limit=8)
    return "No claim-to-code context available." if summary == "(none)" else summary


def _build_integrated_hypotheses_data(state: AgentState) -> list[dict]:
    claims_by_id = {
        claim.get("claim_id"): claim
        for claim in state.get("code_enriched_claims", [])
        if claim.get("claim_id")
    }
    hypotheses: list[dict] = []
    for idx, hypothesis_block in enumerate(state.get("hypotheses", []), start=1):
        claim = claims_by_id.get(f"hypothesis_{idx}", {})
        hypotheses.append(
            {
                "role": "Integrated report",
                "hypothesis": _extract_markdown_field(hypothesis_block, "Hypothesis") or f"Hypothesis {idx}",
                "rationale": _extract_markdown_field(hypothesis_block, "Rationale") or "Grounded in observed portfolio behavior and execution outputs.",
                "possible_verification": _extract_markdown_list(hypothesis_block, "Possible Tests") or ["Validate against the generated plots and execution outputs."],
                "code_evidence_refs": claim.get("code_paths", [])[:6],
                "confidence": "medium",
            }
        )
    return hypotheses


def _call_code_explainer(bundle: dict) -> list[CodeClaimContext]:
    try:
        llm, label = _code_explainer_llm()
    except Exception as exc:
        logger.warning("Code explainer unavailable: %s", exc)
        return _fallback_code_claim_contexts(bundle)

    system_prompt = (
        "You are the Code Explainer for a code-grounded benchmark analysis forum.\n"
        "Your job is to inspect the entire reachable code tree and connect each report claim to the code paths "
        "that would be exercised if that claim is describing the observed behavior.\n"
        "Your code explanation can mention what the algorithm implementation is doing, the role of configurations and parameters,"
        "and how the code maps to the behavior observed in the claim.\n"
        "Build a rich claim-to-code map that names the real files, config keys, algorithms, and execution flow "
        "that seem most relevant to each claim."
    )
    user_prompt = (
        "For every claim in the parsed report, identify the relevant code paths from the provided reachable code tree.\n"
        "Use the module graph, symbol references, config traces, and code excerpts.\n"
        "Return one claim entry per report claim.\n\n"
        f"{_build_code_explainer_context(bundle)}"
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    try:
        result: CodeClaimContextOutput = llm.with_structured_output(CodeClaimContextOutput).invoke(messages)
    except Exception as exc:
        logger.warning("Code explainer %s structured output failed: %s", label, exc)
        return _fallback_code_claim_contexts(bundle)

    by_id = {item.claim_id: item for item in result.claims}
    enriched: list[CodeClaimContext] = []
    for claim in bundle["report_data"]["claims"]:
        if claim["id"] in by_id:
            enriched.append(by_id[claim["id"]])
        else:
            enriched.append(
                CodeClaimContext(
                    claim_id=claim["id"],
                    title=claim["title"],
                    kind=claim["kind"],
                    claim_text=claim["text"],
                    code_paths=[],
                    config_keys=[],
                    explanation="The Code Explainer did not return a claim-specific mapping for this claim.",
                    exercised_flow="Not specified.",
                )
            )
    return enriched


def _call_code_panelist(role: str, llm, bundle: dict, enriched_claims: list[dict]) -> list[StructuredCodeHypothesis]:
    system_prompt = (
        "You are participating in a code-grounded benchmark analysis forum.\n"
        f"Your assigned role is: {role}.\n"
        "These are RL algorithms running financial portfolio trading code.\n"
        "Use the supplied benchmark config as the only config source of truth for this analysis.\n"
        "Ignore sibling config files in the same config directory.\n"
        "Do not over-index on generic facts that are true across the whole benchmark path just because they appear in benchmark.main.\n"
        "Prioritize behavior-relevant differences: portfolio environment controls, RL algorithm family, algorithm-specific configuration, optimization settings, and evaluation logic.\n"
        "Generate 1 to 2 hypotheses that explain how the code could produce the behaviors discussed in report.md.\n"
        "Use the Code Explainer claim-to-code mappings as the primary bridge between report claims and implementation details.\n"
        "Every hypothesis must be grounded in the provided code context, cite local file paths in code_evidence_refs,\n"
        "and include suggested changes categorized as architecture, code, algorithm, or hyperparameters."
    )
    user_prompt = (
        "Use this context to generate code-grounded hypotheses:\n\n"
        f"{_context_summary(bundle)}\n"
        "Code-enriched claims:\n"
        f"{_claim_context_summary(enriched_claims)}\n\n"
        "Focus on the reachable benchmark code under the directory tree only.\n"
        "Treat the supplied benchmark config as the config used in the report and ignore other config files in that directory."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    structured_llm = llm.with_structured_output(CodeHypothesesOutput)
    result: CodeHypothesesOutput = structured_llm.invoke(messages)
    hypotheses = result.hypotheses[:2]
    for hypothesis in hypotheses:
        hypothesis.role = role
    return hypotheses


def _dedupe_code_hypotheses(hypotheses: list[StructuredCodeHypothesis]) -> list[StructuredCodeHypothesis]:
    deduped: list[StructuredCodeHypothesis] = []
    threshold = cfg("code_forum.dedup_threshold", 0.72)
    for item in hypotheses:
        if any(
            SequenceMatcher(None, item.hypothesis.lower(), existing.hypothesis.lower()).ratio() >= threshold
            for existing in deduped
        ):
            continue
        deduped.append(item)
    return deduped


def _peer_review(hypotheses: list[StructuredCodeHypothesis], bundle: dict) -> list[StructuredCodeHypothesis]:
    if not hypotheses or not cfg("code_forum.peer_review.enabled", True):
        return hypotheses
    try:
        llm = get_llm(
            temperature=cfg("temperatures.peer_reviewer", 0.1),
            max_tokens=cfg("code_forum.max_tokens", 8192),
        )
    except Exception as exc:
        logger.warning("Code peer reviewer unavailable: %s", exc)
        return hypotheses

    prompt = (
        "Review the following code-grounded hypotheses.\n"
        "Score each on relevance, evidence grounding, and actionability.\n"
        "Drop only hypotheses that are redundant, weakly grounded, or not actionable.\n"
    )
    entries = []
    for idx, item in enumerate(hypotheses, start=1):
        entries.append(_format_code_hypothesis(idx, item))
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="\n\n".join(entries) + "\n\n" + _context_summary(bundle)),
    ]
    try:
        result: CodePeerReviewOutput = llm.with_structured_output(CodePeerReviewOutput).invoke(messages)
    except Exception as exc:
        logger.warning("Code peer review failed: %s", exc)
        return hypotheses

    kept_indices = {
        review.index
        for review in result.reviews
        if review.verdict == "KEEP" and (review.relevance + review.evidence_grounding + review.actionability) / 3 >= 6
    }
    reviewed = [item for idx, item in enumerate(hypotheses, start=1) if idx in kept_indices]
    return reviewed or hypotheses


def _build_recommendations(hypotheses: list[StructuredCodeHypothesis], bundle: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "architecture": [],
        "code": [],
        "algorithm": [],
        "hyperparameters": [],
    }
    for hypothesis in hypotheses:
        for change in hypothesis.suggested_changes:
            if change.change not in grouped[change.category]:
                grouped[change.category].append(change.change)

    if not grouped["architecture"]:
        grouped["architecture"].append("Persist dependency and config-trace evidence alongside each generated code report.")
    if not grouped["code"]:
        grouped["code"].append("Expose code-path provenance for major report claims directly in the generated markdown output.")
    if not grouped["algorithm"]:
        grouped["algorithm"].append("Introduce explicit environment-level leverage and liquidation safeguards for stress-test runs.")
    if not grouped["hyperparameters"]:
        grouped["hyperparameters"].append("Tune agent-specific learning, batch, and buffer settings instead of relying only on shared global defaults.")
    return grouped


def code_context_builder(state: AgentState):
    logger.info("Starting Code Context Builder node...")
    if state.get("analysis_mode") == "report":
        bundle = build_code_context_bundle_from_report_data(
            report_path=state.get("report_path", ""),
            benchmark_entry=state["benchmark_entry"],
            benchmark_config=state["benchmark_config"],
            code_scope_root=state["code_scope_root"],
            report_data=_build_live_report_data(state),
            data_path=state.get("raw_data_path", ""),
        )
    else:
        bundle = build_code_context_bundle(
            report_path=state["report_path"],
            benchmark_entry=state["benchmark_entry"],
            benchmark_config=state["benchmark_config"],
            code_scope_root=state["code_scope_root"],
            data_path=state.get("raw_data_path", ""),
        )
    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_step(
            "code_context_builder",
            {
                "report_path": bundle["report_path"],
                "module_count": len(bundle["dependency_graph"]["modules"]),
                "claim_count": len(bundle["report_data"]["claims"]),
                "supplemental_placeholders": len(bundle.get("supplemental_code_placeholders", [])),
            },
        )
    return {
        "report_claims": bundle["report_data"]["claims"],
        "code_dependency_graph": bundle["dependency_graph"],
        "code_context_bundle": bundle,
        "next_node": "code_claim_explainer",
        "messages": [
            HumanMessage(
                content=(
                    "Code context bundle prepared.\n"
                    f"Reachable modules: {len(bundle['dependency_graph']['modules'])}\n"
                    f"Claims extracted: {len(bundle['report_data']['claims'])}\n"
                    f"Supplemental placeholders: {len(bundle.get('supplemental_code_placeholders', []))}"
                )
            )
        ],
    }


def code_claim_explainer(state: AgentState):
    logger.info("Starting Code Claim Explainer node...")
    bundle = state["code_context_bundle"]
    if state.get("analysis_mode") == "report":
        report_data = _build_live_report_data(state)
        bundle = _bundle_with_report_data(bundle, report_data)
    enriched = _call_code_explainer(bundle)
    formatted = [
        _format_code_claim_context(idx, item)
        for idx, item in enumerate(enriched, start=1)
    ]

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_step(
            "code_claim_explainer",
            {
                "claims": len(enriched),
                "modules": len(bundle["dependency_graph"]["modules"]),
            },
        )

    return {
        "report_claims": bundle["report_data"]["claims"],
        "code_context_bundle": bundle,
        "code_enriched_claims": [item.model_dump() for item in enriched],
        "next_node": "code_hypothesis_forum",
        "messages": [
            HumanMessage(
                content="Code explainer mapped report claims to reachable code:\n\n" + "\n\n".join(formatted)
            )
        ],
    }


def code_hypothesis_forum(state: AgentState):
    logger.info("Starting Code Hypothesis Forum node...")
    bundle = state["code_context_bundle"]
    enriched_claims = state.get("code_enriched_claims", [])
    panelists = _safe_panelists()
    hypotheses: list[StructuredCodeHypothesis] = []

    if panelists:
        with ThreadPoolExecutor(max_workers=len(panelists)) as executor:
            futures = {
                executor.submit(_call_code_panelist, role, llm, bundle, enriched_claims): (role, label)
                for role, llm, label in panelists
            }
            for future in as_completed(futures):
                role, label = futures[future]
                try:
                    result = future.result()
                    hypotheses.extend(result)
                    step_logger = get_step_logger()
                    if step_logger:
                        step_logger.log_step(
                            "code_hypothesis_panelist",
                            {"role": role, "panelist": label, "count": len(result)},
                        )
                except Exception as exc:
                    logger.warning("Code panelist %s failed: %s", role, exc)

    if not hypotheses:
        hypotheses = _fallback_hypotheses(bundle)

    hypotheses = _peer_review(_dedupe_code_hypotheses(hypotheses), bundle)
    recommendations = _build_recommendations(hypotheses, bundle)
    formatted = [_format_code_hypothesis(idx, item) for idx, item in enumerate(hypotheses, start=1)]

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_step(
            "code_hypothesis_forum",
            {
                "hypotheses_count": len(formatted),
                "roles": [item.role for item in hypotheses],
            },
        )

    return {
        "code_hypotheses": formatted,
        "code_hypotheses_data": [item.model_dump() for item in hypotheses],
        "code_recommendations": recommendations,
        "next_node": "code_hypothesis_investigator",
        "messages": [HumanMessage(content="Generated code-grounded hypotheses:\n\n" + "\n\n".join(formatted))],
    }


def code_hypothesis_investigator(state: AgentState):
    logger.info("Starting Code Hypothesis Investigator node...")
    if state.get("analysis_mode") == "report":
        hypotheses = _build_integrated_hypotheses_data(state)
    else:
        hypotheses = state.get("code_hypotheses_data", [])
    tasks, evidence_results = _collect_hypothesis_evidence(
        state["code_context_bundle"],
        state.get("code_enriched_claims", []),
        hypotheses,
        max_tasks=cfg("code_investigator.max_evidence_tasks", 5),
    )
    if state.get("analysis_mode") == "report":
        next_node = "consensus_forum"
        reasoning = "Prompt-based code evidence collected from the live report hypotheses and supplied code context."
    else:
        next_node = "code_consensus_forum"
        reasoning = "Prompt-based code evidence collected from the supplied code context."

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_step(
            "code_hypothesis_investigator",
            {"next_node": next_node, "tasks": len(tasks), "evidence_results": len(evidence_results)},
        )

    return {
        "code_investigation_tasks": tasks,
        "code_evidence_results": evidence_results,
        "next_node": next_node,
        "messages": [HumanMessage(content=f"Code investigator decision: {next_node}. {reasoning}")],
    }


def _code_forum_context(state: AgentState) -> str:
    parts = [
        f"Report path: {state['report_path']}",
        f"Benchmark entry: {state['benchmark_entry']}",
        f"Benchmark config: {state['benchmark_config']}",
        "",
        "=== CODE-ENRICHED CLAIMS ===",
        _claim_context_summary(state.get("code_enriched_claims", [])),
        "",
        "=== CODE HYPOTHESES ===",
        "\n\n".join(state.get("code_hypotheses", [])),
        "",
        "=== REPORT CLAIMS ===",
        json.dumps(state.get("report_claims", [])[:10], indent=2),
        "",
        "=== EVIDENCE RESULTS ===",
        json.dumps(state.get("code_evidence_results", []), indent=2),
    ]
    return "\n".join(parts)


def _fallback_consensus_answers(state: AgentState) -> dict[str, str]:
    return {
        "what": (
            "The benchmark code trains configured agents on real-market data, then evaluates those trained policies "
            "on a stitched interleaved synthetic regime while recording account values, trading logs, and aggregate metrics."
        ),
        "how": (
            "It does this by loading YAML config into the benchmark entry module, routing runtime control through the worker, "
            "building env kwargs with the trading environment config, sharing architecture and optimizer helpers through the agent pipeline, "
            "and aggregating evaluation outputs through the analysis and metrics stack."
        ),
        "why": (
            "The observed behavior emerges from the combination of environment controls, shared training knobs, "
            "algorithm-specific overrides, and downstream metric interpretation. The code path makes these effects "
            "visible because config values are threaded directly into environment construction, agent training, and evaluation."
        ),
    }


def code_consensus_forum(state: AgentState):
    logger.info("Starting Code Consensus Forum node...")
    panelists = _safe_panelists()
    context = _code_forum_context(state)

    if not panelists:
        answers = _fallback_consensus_answers(state)
    else:
        forum_panelists = [(f"{role}::{label}", llm) for role, llm, label in panelists]
        synthesiser = None
        try:
            synthesiser = get_llm(
                temperature=cfg("temperatures.code_consensus_forum", 0.2),
                max_tokens=cfg("code_report_generator.max_tokens", 8192),
            )
        except Exception as exc:
            logger.warning("Code consensus synthesiser unavailable: %s", exc)
        prompts = {
            "propose": (
                "You are a senior software and quantitative systems reviewer.\n"
                "Answer the question strictly from the code evidence provided. Use concrete module and config references."
            ),
            "evaluate": (
                "Peer-review the proposals for code grounding, internal consistency, and actionability."
            ),
            "consensus": (
                "Write one definitive, code-grounded consensus answer. Cite concrete modules, functions, and config flows."
            ),
        }
        questions = [
            ("what", "What is the code making the agents do?"),
            ("how", "How is the code doing it?"),
            ("why", "Why is the code producing the observed behavior?"),
        ]
        answers = {}
        for key, topic in questions:
            result = run_forum(
                topic=topic,
                context=context,
                panelists=forum_panelists,
                synthesiser=synthesiser,
                propose_system_prompt=prompts["propose"],
                evaluate_system_prompt=prompts["evaluate"],
                consensus_system_prompt=prompts["consensus"],
                max_rounds=cfg("code_consensus_forum.max_rounds", 3),
            )
            answers[key] = result.final_answer

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_step(
            "code_consensus_forum",
            {"keys": list(answers.keys())},
        )

    return {
        "code_consensus_answers": answers,
        "next_node": "code_report_generator",
        "messages": [HumanMessage(content="Code consensus forum completed.")],
    }


def _recommendations_section(recommendations: dict[str, list[str]]) -> str:
    lines = ["# Recommendations", ""]
    sections = [
        ("Architecture changes", "architecture"),
        ("Code changes", "code"),
        ("Algorithm changes", "algorithm"),
        ("Hyperparameter changes", "hyperparameters"),
    ]
    for title, key in sections:
        lines.append(f"## {title}")
        for item in recommendations.get(key, []):
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _tests_results_section(state: AgentState) -> str:
    lines = ["# Tests and Results", ""]
    tasks = state.get("code_investigation_tasks", [])
    results_by_id = {item["task_id"]: item["result"] for item in state.get("code_evidence_results", [])}
    if not tasks:
        lines.append("No code investigation tasks were selected.")
        return "\n".join(lines)
    for task in tasks:
        lines.append(f"## {task['title']}")
        lines.append(task["objective"])
        result = results_by_id.get(task["id"])
        if result is None:
            lines.append("")
            lines.append("Result: evidence was not captured for this task.")
        else:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(result, indent=2, sort_keys=True))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip()


def _behavior_analysis_section(answers: dict[str, str]) -> str:
    sections = [
        ("What is the agent doing?", answers.get("what", "(not available)")),
        ("How is it doing it?", answers.get("how", "(not available)")),
        ("Why does it exhibit this behavior?", answers.get("why", "(not available)")),
    ]
    lines = ["# Agent Behavior Analysis", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip()


def code_report_generator(state: AgentState):
    logger.info("Starting Code Report Generator node...")
    recommendations = state.get("code_recommendations", {})
    body_parts = [
        "# Trading Agent Code Report",
        "",
        "# Hypotheses",
        "",
        "The following code-grounded hypotheses were formulated by the multi-panelist forum to explain how the reachable benchmark code could produce the behaviors discussed in report.md.",
        "",
        "\n\n".join(state.get("code_hypotheses", [])) or "No code-grounded hypotheses were generated.",
        "",
        _tests_results_section(state),
        "",
        _behavior_analysis_section(state.get("code_consensus_answers", {})),
        "",
        _recommendations_section(recommendations),
        "",
    ]
    report = "\n".join(part for part in body_parts if part is not None).strip() + "\n"
    with open("code_report.md", "w", encoding="utf-8") as handle:
        handle.write(report)

    step_logger = get_step_logger()
    if step_logger:
        step_logger.log_step(
            "code_report_generator",
            {"report_length": len(report)},
        )
        step_logger.log_code_report(report)

    return {
        "final_code_report": report,
        "messages": [HumanMessage(content="Final code report generated: code_report.md")],
    }
