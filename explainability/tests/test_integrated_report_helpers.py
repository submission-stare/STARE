import os
import unittest

from utils.code_report import build_code_context_bundle_from_report_data

try:
    from nodes.code_report_flow import _build_integrated_hypotheses_data, _build_live_report_claims
    from nodes.hypothesis_forum import _build_prompts
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    _build_integrated_hypotheses_data = None
    _build_live_report_claims = None
    _build_prompts = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "run.py"))
CONFIG_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "config.yaml"))
SCOPE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))


class TestIntegratedReportHelpers(unittest.TestCase):
    @unittest.skipIf(_build_live_report_claims is None, "LangChain runtime is not available in the system Python")
    def test_live_report_claims_include_hypotheses_and_tests(self):
        state = {
            "hypotheses": [
                "### Hypothesis 1\n"
                "**Hypothesis:** The agent over-concentrates in one sector.\n"
                "**Rationale:** Concentration spikes alongside drawdowns.\n"
                "**Possible Tests:**\n"
                "   - Measure peak weight concentration by sector"
            ],
            "investigation_tests": ["Measure peak weight concentration by sector"],
            "consensus_answers": {},
        }

        claims = _build_live_report_claims(state)
        self.assertEqual(claims[0]["id"], "hypothesis_1")
        self.assertEqual(claims[0]["kind"], "hypothesis")
        self.assertIn("over-concentrates", claims[0]["text"])
        self.assertEqual(claims[1]["id"], "tests_and_results")

    @unittest.skipIf(_build_integrated_hypotheses_data is None, "LangChain runtime is not available in the system Python")
    def test_integrated_hypotheses_data_reuses_claim_grounding(self):
        state = {
            "hypotheses": [
                "### Hypothesis 1\n"
                "**Hypothesis:** The agent over-concentrates in one sector.\n"
                "**Rationale:** Concentration spikes alongside drawdowns.\n"
                "**Possible Tests:**\n"
                "   - Measure peak weight concentration by sector"
            ],
            "code_enriched_claims": [
                {
                    "claim_id": "hypothesis_1",
                    "title": "Hypothesis 1",
                    "kind": "hypothesis",
                    "claim_text": "The agent over-concentrates in one sector.",
                    "code_paths": ["experiments/neurips2026_turb/run.py"],
                    "config_keys": ["agents"],
                    "exercised_flow": "benchmark.main -> benchmark.worker",
                    "explanation": "The worker threads env config into the training loop.",
                }
            ],
        }

        hypotheses = _build_integrated_hypotheses_data(state)
        self.assertEqual(hypotheses[0]["hypothesis"], "The agent over-concentrates in one sector.")
        self.assertEqual(hypotheses[0]["code_evidence_refs"], ["experiments/neurips2026_turb/run.py"])
        self.assertIn("Measure peak weight concentration by sector", hypotheses[0]["possible_verification"][0])

    @unittest.skipIf(_build_prompts is None, "LangChain runtime is not available in the system Python")
    def test_hypothesis_prompt_receives_compact_code_digest_only(self):
        bundle = build_code_context_bundle_from_report_data(
            report_path="",
            benchmark_entry=ENTRY_PATH,
            benchmark_config=CONFIG_PATH,
            code_scope_root=SCOPE_ROOT,
            report_data={"claims": []},
        )
        state = {
            "raw_data_path": REPO_ROOT,
            "hypotheses": [],
            "transaction_summary": "Transactions show concentrated buying.",
            "snapshot_summary": "Snapshots show shrinking cash buffers.",
            "code_context_bundle": bundle,
        }

        system_prompt, user_prompt = _build_prompts(state)
        self.assertIn("compact code-context digest", system_prompt)
        self.assertIn("=== CODE CONTEXT DIGEST ===", user_prompt)
        self.assertNotIn("BEGIN SOURCE FILE", user_prompt)
        self.assertNotIn("```python", user_prompt)


if __name__ == "__main__":
    unittest.main()
