import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from langchain_core.messages import HumanMessage
    from main import create_agent_graph
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    HumanMessage = None
    create_agent_graph = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(REPO_ROOT, "report.md")
ENTRY_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "run.py"))
CONFIG_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "config.yaml"))
SCOPE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))


class TestCodeReportIntegration(unittest.TestCase):
    @unittest.skipIf(HumanMessage is None or create_agent_graph is None, "LangChain runtime is not available in the system Python")
    @patch("nodes.code_report_flow._safe_panelists", return_value=[])
    def test_code_report_graph_generates_markdown_report(self, _mock_panelists):
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                initial_state = {
                    "messages": [HumanMessage(content="Starting code-report integration test")],
                    "raw_data_path": "",
                    "hypotheses": [],
                    "generated_code": "",
                    "plot_paths": [],
                    "final_report": "",
                    "next_node": "",
                    "investigation_tests": [],
                    "code_fix_retries": 0,
                    "transaction_summary": "",
                    "snapshot_summary": "",
                    "consensus_answers": {},
                    "analysis_mode": "code-report",
                    "report_path": REPORT_PATH,
                    "benchmark_entry": ENTRY_PATH,
                    "benchmark_config": CONFIG_PATH,
                    "code_scope_root": SCOPE_ROOT,
                    "report_claims": [],
                    "code_dependency_graph": {},
                    "code_context_bundle": {},
                    "code_enriched_claims": [],
                    "code_hypotheses": [],
                    "code_hypotheses_data": [],
                    "code_investigation_tasks": [],
                    "code_evidence_results": [],
                    "code_consensus_answers": {},
                    "code_recommendations": {},
                    "final_code_report": "",
                }

                graph = create_agent_graph(analysis_mode="code-report")
                final_state = initial_state
                for state in graph.stream(initial_state, {"recursion_limit": 20}):
                    node_name = list(state.keys())[0]
                    node_update = state[node_name]
                    for key, value in node_update.items():
                        if key in {"messages", "plot_paths"}:
                            final_state[key] = final_state.get(key, []) + value
                        else:
                            final_state[key] = value

                code_report_path = os.path.join(tmpdir, "code_report.md")
                self.assertTrue(os.path.exists(code_report_path))
                self.assertTrue(final_state["final_code_report"])

                with open(code_report_path, "r", encoding="utf-8") as handle:
                    content = handle.read()

                self.assertIn("# Trading Agent Code Report", content)
                self.assertIn("# Hypotheses", content)
                self.assertIn("# Tests and Results", content)
                self.assertIn("# Agent Behavior Analysis", content)
                self.assertIn("# Recommendations", content)
                self.assertIn("Architecture changes", content)
                self.assertIn("Code changes", content)
                self.assertIn("Algorithm changes", content)
                self.assertIn("Hyperparameter changes", content)
                self.assertIn("experiments/neurips2026_turb/run.py", content)
            finally:
                os.chdir(prev_cwd)


if __name__ == "__main__":
    unittest.main()
