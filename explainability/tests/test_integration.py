import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from langchain_core.messages import HumanMessage
    from main import _build_initial_state, _normalize_analysis_mode, create_agent_graph
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    HumanMessage = None
    _build_initial_state = None
    _normalize_analysis_mode = None
    create_agent_graph = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "run.py"))
CONFIG_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "config.yaml"))
SCOPE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))


def _merge_state(current_state: dict, node_update: dict) -> None:
    for key, value in node_update.items():
        if key in {"messages", "plot_paths"}:
            current_state[key] = current_state.get(key, []) + value
        else:
            current_state[key] = value


def _router_stub(state):
    if not state.get("hypotheses"):
        next_node = "hypothesis_maker"
    elif not state.get("plot_paths"):
        next_node = "hypothesis_investigator"
    else:
        next_node = "code_hypothesis_investigator"
    return {
        "next_node": next_node,
        "messages": [HumanMessage(content=f"Stub routed to {next_node}")],
    }


def _hypothesis_stub(_state):
    hypothesis = (
        "### Hypothesis 1\n"
        "**Hypothesis:** The agent concentrates capital into a narrow set of names during drawdown windows.\n"
        "**Rationale:** Transaction bursts and shrinking cash buffers point to aggressive concentration.\n"
        "**Key Factors:**\n"
        "   - Buy activity clusters around a small ticker set\n"
        "   - Cash weight falls sharply into the stressed window\n"
        "**Possible Tests:**\n"
        "   - Compare peak single-name weights and cash depletion around the drawdown period"
    )
    return {
        "hypotheses": [hypothesis],
        "messages": [HumanMessage(content="Generated stub hypotheses.")],
    }


def _claim_explainer_stub(_state):
    claims = [
        {
            "id": "hypothesis_1",
            "kind": "hypothesis",
            "title": "Hypothesis 1",
            "text": "The agent concentrates capital into a narrow set of names during drawdown windows.",
        }
    ]
    enriched = [
        {
            "claim_id": "hypothesis_1",
            "title": "Hypothesis 1",
            "kind": "hypothesis",
            "claim_text": claims[0]["text"],
            "code_paths": [
                "experiments/neurips2026_turb/run.py",
                "evaluation/experiment_setup.py",
            ],
            "config_keys": ["agents", "tickers"],
            "exercised_flow": "experiments.neurips2026_turb.run -> evaluation.experiment_setup -> evaluation.runner",
            "explanation": "The experiment entry point threads config into experiment_setup and runner that shape agent training and evaluation.",
        }
    ]
    return {
        "report_claims": claims,
        "code_enriched_claims": enriched,
        "messages": [HumanMessage(content="Mapped stub claims to code.")],
    }


def _investigator_stub(state):
    if state.get("plot_paths"):
        return {
            "next_node": "data_analyst",
            "investigation_tests": [
                "Compare peak single-name weights and cash depletion around the drawdown period"
            ],
            "messages": [HumanMessage(content="Investigation complete, routing back to data_analyst.")],
        }
    return {
        "next_node": "code_generator",
        "investigation_tests": [
            "Compare peak single-name weights and cash depletion around the drawdown period"
        ],
        "messages": [HumanMessage(content="Selected stub investigation tests.")],
    }


def _code_generator_stub(_state):
    return {
        "generated_code": "print('stub analysis')",
        "messages": [HumanMessage(content="Generated stub code.")],
    }


def _code_executor_stub(_state):
    return {
        "plot_paths": ["generated_code_results/plot_001.png"],
        "next_node": "hypothesis_investigator",
        "messages": [
            HumanMessage(
                content=(
                    "Code Execution Results:\n"
                    "Execution Output:\n"
                    "Peak weight concentration reached 42% while cash weight fell below 5%.\n\n"
                    "Generated Plots: plot_001.png\n"
                )
            )
        ],
    }


def _code_evidence_stub(_state):
    return {
        "code_investigation_tasks": [
            {
                "id": "hypothesis_1_evidence",
                "title": "Hypothesis 1 Evidence",
                "task_type": "hypothesis_evidence",
                "objective": "The agent concentrates capital into a narrow set of names during drawdown windows.",
            }
        ],
        "code_evidence_results": [
            {
                "task_id": "hypothesis_1_evidence",
                "title": "Hypothesis 1 Evidence",
                "task_type": "hypothesis_evidence",
                "result": {
                    "summary": "Worker and agent-builder paths concentrate config-driven position sizing into the training and evaluation loop.",
                    "supporting_paths": [
                        "experiments/neurips2026_turb/run.py",
                        "evaluation/experiment_setup.py",
                    ],
                    "config_keys": ["agents", "tickers"],
                    "evidence_snippets": [
                        "benchmark.main threads benchmark config into worker execution",
                        "worker builds environment and agent kwargs from config-driven settings",
                    ],
                    "confidence": "medium",
                },
            }
        ],
        "next_node": "consensus_forum",
        "messages": [HumanMessage(content="Collected stub code evidence.")],
    }


def _consensus_stub(_state):
    answers = {
        "what": "The agent concentrates capital into a small set of names and does so most aggressively during stressed windows.",
        "how": "It does this through config-driven environment and evaluation paths in experiments/neurips2026_turb/run.py and evaluation/experiment_setup.py.",
        "why": "The observed concentration is consistent with the benchmark config flowing through the worker and agent builders, which shape position sizing and exposure.",
    }
    return {
        "consensus_answers": answers,
        "messages": [HumanMessage(content="Generated stub consensus answers.")],
    }


def _code_context_builder_stub(state):
    bundle = {
        "report_path": state.get("report_path", ""),
        "benchmark_entry": state["benchmark_entry"],
        "benchmark_config": state["benchmark_config"],
        "code_scope_root": state["code_scope_root"],
        "report_data": {"title": "", "hypotheses": [], "tests_and_results": "", "behavior_analysis": "", "consensus_sections": {}, "claims": [], "raw_text": ""},
        "dependency_graph": {"entry_module": "experiments.neurips2026_turb.run", "entry_path": state["benchmark_entry"], "modules": [], "edges": [], "symbol_refs": [], "external_dependencies": [], "unresolved_imports": [], "placeholder_sources": {}, "supplemental_placeholders": []},
        "config_traces": {"config_path": state["benchmark_config"], "keys": ["agents", "tickers"], "traces": {"agents": [], "tickers": []}},
        "supplemental_code_placeholders": [],
        "unresolved_items": [],
    }
    return {
        "report_claims": [],
        "code_dependency_graph": bundle["dependency_graph"],
        "code_context_bundle": bundle,
        "next_node": "code_claim_explainer",
        "messages": [HumanMessage(content="Stub code context bundle prepared.")],
    }


    @unittest.skipIf(HumanMessage is None or create_agent_graph is None, "LangChain runtime is not available in the system Python")
    def test_integrated_report_graph_generates_grounded_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                initial_state = _build_initial_state(
                    analysis_mode="report",
                    data_path="",
                    report_path=os.path.join(tmpdir, "report.md"),
                    benchmark_entry=ENTRY_PATH,
                    benchmark_config=CONFIG_PATH,
                    code_scope_root=SCOPE_ROOT,
                )
                initial_state["messages"] = [HumanMessage(content="Starting stub integrated report test")]

                with patch("main.code_context_builder", side_effect=_code_context_builder_stub), \
                     patch("main.data_analyst", side_effect=_router_stub), \
                     patch("main.hypothesis_forum", side_effect=_hypothesis_stub), \
                     patch("main.code_claim_explainer", side_effect=_claim_explainer_stub), \
                     patch("main.hypothesis_investigator", side_effect=_investigator_stub), \
                     patch("main.code_generator", side_effect=_code_generator_stub), \
                     patch("main.code_executor", side_effect=_code_executor_stub), \
                     patch("main.code_hypothesis_investigator", side_effect=_code_evidence_stub), \
                     patch("main.consensus_forum", side_effect=_consensus_stub):
                    graph = create_agent_graph(analysis_mode="report")
                    final_state = initial_state
                    for state in graph.stream(initial_state, {"recursion_limit": 20}):
                        node_name = list(state.keys())[0]
                        _merge_state(final_state, state[node_name])

                report_path = os.path.join(tmpdir, "report.md")
                self.assertTrue(os.path.exists(report_path))
                with open(report_path, "r", encoding="utf-8") as handle:
                    content = handle.read()

                self.assertIn("Code grounding", content)
                self.assertIn("Implementation References", content)
                self.assertIn("Implementation Evidence Appendix", content)
                self.assertIn("experiments/neurips2026_turb/run.py", content)
                self.assertIn("agents", content)
                self.assertTrue(final_state["final_report"])
            finally:
                os.chdir(prev_cwd)

    @unittest.skipIf(_normalize_analysis_mode is None or create_agent_graph is None, "LangChain runtime is not available in the system Python")
    def test_both_mode_normalizes_to_integrated_report(self):
        self.assertEqual(_normalize_analysis_mode("both"), "report")
        self.assertIsNotNone(create_agent_graph(analysis_mode="both"))


if __name__ == "__main__":
    unittest.main()
