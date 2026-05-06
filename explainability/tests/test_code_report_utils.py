import os
import tempfile
import unittest

from utils.code_report import (
    build_code_context_bundle_from_report_data,
    build_code_context_bundle,
    build_code_context_digest,
    discover_supplemental_code_placeholders,
    parse_report_markdown,
    trace_python_dependencies,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(REPO_ROOT, "report.md")
ENTRY_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "run.py"))
CONFIG_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "experiments", "neurips2026_turb", "config.yaml"))
SCOPE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))


class TestCodeReportUtils(unittest.TestCase):
    @unittest.skipIf(not os.path.exists(REPORT_PATH), "report.md not yet generated")
    def test_report_parser_extracts_hypotheses_and_consensus(self):
        parsed = parse_report_markdown(REPORT_PATH)
        self.assertGreaterEqual(len(parsed["hypotheses"]), 10)
        self.assertIn("Hypothesis 1", parsed["hypotheses"][0]["title"])
        self.assertIn("Hypothesis 1", parsed["tests_and_results"])
        self.assertTrue(parsed["consensus_sections"]["what"])
        self.assertTrue(parsed["consensus_sections"]["how"])
        self.assertTrue(parsed["consensus_sections"]["why"])

    def test_dependency_tracer_reaches_benchmark_modules_without_hardcoding(self):
        graph = trace_python_dependencies(ENTRY_PATH, SCOPE_ROOT)
        modules = {item["module"] for item in graph["modules"]}
        self.assertIn("experiments.neurips2026_turb.run", modules)
        self.assertIn("evaluation.experiment_setup", modules)
        self.assertIn("evaluation.runner", modules)
        self.assertIn("evaluation.reporting", modules)
        self.assertIn("evaluation.pipeline", modules)

    def test_dependency_tracer_handles_local_imports_and_reexports(self):
        graph = trace_python_dependencies(ENTRY_PATH, SCOPE_ROOT)
        symbol_refs = graph["symbol_refs"]
        self.assertTrue(
            any(
                ref["imported_from"] == "evaluation.experiment_setup"
                and ref["name"] == "build_envs"
                and ref["resolved_module"] == "evaluation.experiment_setup"
                for ref in symbol_refs
            )
        )
        self.assertTrue(
            any(
                ref["scope"] == "local"
                and ref["resolved_module"] == "agents.llms.llm_strategist"
                for ref in symbol_refs
            )
        )

    def test_config_traces_map_keys_into_runtime_consumers(self):
        bundle = build_code_context_bundle_from_report_data(
            report_path="", benchmark_entry=ENTRY_PATH, benchmark_config=CONFIG_PATH,
            code_scope_root=SCOPE_ROOT, report_data={"claims": []},
        )
        tickers_hits = bundle["config_traces"]["traces"]["tickers"]
        agents_hits = bundle["config_traces"]["traces"]["agents"]
        self.assertTrue(any(hit["module"] == "evaluation.experiment_setup" for hit in tickers_hits))
        self.assertTrue(any(hit["module"] == "evaluation.pipeline" for hit in tickers_hits))
        self.assertTrue(any(hit["module"] in {"evaluation.experiment_setup", "evaluation.pipeline"} for hit in agents_hits))

    def test_report_independent_bundle_builder_preserves_live_claims(self):
        report_data = {
            "title": "Integrated Report",
            "claims": [
                {
                    "id": "hypothesis_1",
                    "kind": "hypothesis",
                    "title": "Hypothesis 1",
                    "text": "The agent concentrates in a narrow set of tickers.",
                }
            ],
        }
        bundle = build_code_context_bundle_from_report_data(
            report_path="",
            benchmark_entry=ENTRY_PATH,
            benchmark_config=CONFIG_PATH,
            code_scope_root=SCOPE_ROOT,
            report_data=report_data,
        )

        self.assertEqual(bundle["report_path"], "")
        self.assertEqual(bundle["report_data"]["claims"][0]["id"], "hypothesis_1")
        self.assertIn("evaluation.experiment_setup", {item["module"] for item in bundle["dependency_graph"]["modules"]})

    def test_result_run_snapshots_are_added_as_code_placeholders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = os.path.join(tmpdir, "20260421115701_less_info_website")
            os.makedirs(os.path.join(result_root, "benchmark-data"))
            with open(os.path.join(result_root, "run_st_strategist.py"), "w", encoding="utf-8") as handle:
                handle.write("def run_st_strategist():\n    return 'ok'\n")
            with open(os.path.join(result_root, "config.yaml"), "w", encoding="utf-8") as handle:
                handle.write("timesteps_per_model: 20000\n")

            placeholders = discover_supplemental_code_placeholders(
                data_path=os.path.join(result_root, "benchmark-data"),
                code_scope_root=SCOPE_ROOT,
            )
            virtual_paths = {item["relative_path"] for item in placeholders}
            self.assertIn(
                os.path.join(
                    "..",
                    "experiments",
                    "liu_et_al_2020",
                    "20260421115701_less_info_website",
                    "run_st_strategist.py",
                ),
                virtual_paths,
            )

            bundle = build_code_context_bundle_from_report_data(
                report_path="",
                benchmark_entry=ENTRY_PATH,
                benchmark_config=CONFIG_PATH,
                code_scope_root=SCOPE_ROOT,
                report_data={"claims": []},
                data_path=result_root,
            )
            placeholder_modules = [
                module
                for module in bundle["dependency_graph"]["modules"]
                if module.get("is_placeholder")
            ]
            self.assertTrue(
                any(
                    module["module"].endswith(
                        "experiments.liu_et_al_2020.20260421115701_less_info_website.run_st_strategist"
                    )
                    for module in placeholder_modules
                )
            )
            self.assertEqual(len(bundle["supplemental_code_placeholders"]), 2)

    def test_code_context_digest_is_compact_and_source_free(self):
        bundle = build_code_context_bundle_from_report_data(
            report_path="",
            benchmark_entry=ENTRY_PATH,
            benchmark_config=CONFIG_PATH,
            code_scope_root=SCOPE_ROOT,
            report_data={"claims": []},
        )
        digest = build_code_context_digest(bundle)
        self.assertIn("=== CODE CONTEXT DIGEST ===", digest)
        self.assertIn("Focused config traces:", digest)
        self.assertNotIn("BEGIN SOURCE FILE", digest)
        self.assertNotIn("```python", digest)


if __name__ == "__main__":
    unittest.main()
