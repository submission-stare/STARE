import os
import tempfile
import unittest

import pandas as pd

from utils.data_scope import resolve_data_scope
from utils.data_sources import (
    DataSourceValidationError,
    discover_result_run,
    expand_data_paths,
    stage_multi_run_dataset,
    validate_result_runs,
)
from utils.s3_data_loader import generate_check_input_data, load_all

try:
    from main import _build_initial_state, _prepare_report_data_path
    from nodes.code_generator import _discover_dataframes
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    _build_initial_state = None
    _prepare_report_data_path = None
    _discover_dataframes = None


def _write_csv(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _make_result_run(
    tmpdir: str,
    run_name: str,
    agent: str,
    *,
    transactions: bool = True,
    snapshots: bool = True,
    account_values: bool = True,
) -> str:
    result_root = os.path.join(tmpdir, run_name)
    agent_dir = os.path.join(
        result_root,
        "benchmark-data",
        "general",
        "agents_trading",
        "trading_analysis",
        agent,
    )
    if transactions:
        _write_csv(
            os.path.join(agent_dir, "transactions.csv"),
            "date,ticker,price,action_type,shares_traded\n"
            "2019-01-02,AAPL,100.0,BUY,1\n",
        )
    if snapshots:
        _write_csv(
            os.path.join(agent_dir, "snapshots.csv"),
            "date,cash_weight,total_asset\n"
            "2019-01-02,1.0,1000000\n"
            "2019-01-03,0.9,1001000\n",
        )
    if account_values:
        _write_csv(
            os.path.join(result_root, "benchmark-data", "general", "account_values.csv"),
            f"date,{agent},benchmark\n"
            "2019-01-02,1000000,1000000\n"
            "2019-01-03,1001000,1000200\n",
        )
    with open(os.path.join(result_root, "compute_log.json"), "w", encoding="utf-8") as handle:
        handle.write(f'{{"agent_timings": {{"{agent}": {{"training_time_s": 1}}}}}}')
    return result_root


class TestDataSources(unittest.TestCase):
    def test_expand_data_paths_flattens_repeated_args_and_globs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = os.path.join(tmpdir, "20260430_a2c")
            second = os.path.join(tmpdir, "20260430_ppo")
            os.makedirs(first)
            os.makedirs(second)

            paths = expand_data_paths([[first], [os.path.join(tmpdir, "20260430_*")]])

            self.assertEqual(paths, [first, first, second])

    def test_result_root_with_general_layout_resolves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = _make_result_run(tmpdir, "20260430144239_turb_a2c", "A2C")

            run = discover_result_run(result_root)
            scope = resolve_data_scope(result_root)

            self.assertEqual(run.benchmark_root, os.path.join(result_root, "benchmark-data"))
            self.assertEqual([agent.label for agent in run.agents], ["A2C"])
            self.assertTrue(scope["trading_dir"].endswith(os.path.join("benchmark-data", "general")))

    def test_benchmark_data_root_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = _make_result_run(tmpdir, "single_run", "PPO")
            benchmark_root = os.path.join(result_root, "benchmark-data")

            run = discover_result_run(benchmark_root)

            self.assertEqual(run.result_root, result_root)
            self.assertEqual(run.agents[0].label, "PPO")

    def test_multiple_runs_stage_into_one_unified_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a2c = discover_result_run(_make_result_run(tmpdir, "20260430_a2c", "A2C"))
            ppo = discover_result_run(_make_result_run(tmpdir, "20260430_ppo", "PPO"))

            staged = stage_multi_run_dataset([a2c, ppo], os.path.join(tmpdir, "staged"))

            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        staged,
                        "general",
                        "agents_trading",
                        "trading_analysis",
                        "A2C",
                        "transactions.csv",
                    )
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        staged,
                        "general",
                        "agents_trading",
                        "trading_analysis",
                        "PPO",
                        "snapshots.csv",
                    )
                )
            )
            account_values = pd.read_csv(os.path.join(staged, "general", "account_values.csv"))
            self.assertEqual(account_values.columns.tolist(), ["date", "A2C", "PPO"])

    def test_duplicate_agent_labels_are_disambiguated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = discover_result_run(_make_result_run(tmpdir, "first_run", "A2C"))
            second = discover_result_run(_make_result_run(tmpdir, "second_run", "A2C"))

            staged = stage_multi_run_dataset([first, second], os.path.join(tmpdir, "staged"))
            agents_dir = os.path.join(staged, "general", "agents_trading", "trading_analysis")

            self.assertEqual(sorted(os.listdir(agents_dir)), ["A2C", "second_run_A2C"])

    def test_validation_fails_for_missing_benchmark_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_run = os.path.join(tmpdir, "empty")
            os.makedirs(empty_run)

            with self.assertRaises(DataSourceValidationError) as ctx:
                validate_result_runs([discover_result_run(empty_run)])

            self.assertIn("benchmark-data directory was not found", str(ctx.exception))

    def test_validation_fails_for_missing_agent_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_tx = discover_result_run(
                _make_result_run(tmpdir, "missing_transactions", "TD3", transactions=False)
            )
            missing_sn = discover_result_run(
                _make_result_run(tmpdir, "missing_snapshots", "DDPG", snapshots=False)
            )

            with self.assertRaises(DataSourceValidationError) as ctx:
                validate_result_runs([missing_tx, missing_sn])

            self.assertIn("missing transactions.csv", str(ctx.exception))
            self.assertIn("missing snapshots.csv", str(ctx.exception))

    def test_validation_fails_for_empty_selection(self):
        with self.assertRaises(DataSourceValidationError) as ctx:
            validate_result_runs([])

        self.assertIn("no data paths were provided or matched", str(ctx.exception))

    @unittest.skipIf(_prepare_report_data_path is None, "LangChain runtime is not available")
    def test_explicit_unmatched_glob_does_not_fall_back_to_inference(self):
        class DummyLogger:
            run_dir = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(DataSourceValidationError) as ctx:
                _prepare_report_data_path([os.path.join(tmpdir, "missing_*")], DummyLogger())

        self.assertIn("no data paths were provided or matched", str(ctx.exception))

    @unittest.skipIf(_build_initial_state is None or _discover_dataframes is None, "LangChain runtime is not available")
    def test_staged_data_loads_through_existing_pipeline_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                a2c = discover_result_run(_make_result_run(tmpdir, "20260430_a2c", "A2C"))
                ppo = discover_result_run(_make_result_run(tmpdir, "20260430_ppo", "PPO"))
                staged = stage_multi_run_dataset([a2c, ppo], os.path.join(tmpdir, "staged"))

                state = _build_initial_state(
                    analysis_mode="report",
                    data_path=staged,
                    report_path=os.path.join(tmpdir, "report.md"),
                    benchmark_entry="../experiments/neurips2026_turb/run.py",
                    benchmark_config="../experiments/neurips2026_turb/config.yaml",
                    code_scope_root="..",
                )
                self.assertEqual(state["raw_data_path"], staged)

                trading_analysis_dir = resolve_data_scope(staged)["trading_analysis_dir"]
                transactions, snapshots = load_all(base_dir=trading_analysis_dir)
                generate_check_input_data(transactions, snapshots, output_dir=os.path.join(tmpdir, "check_input_data"))

                self.assertEqual(set(transactions["agent"]), {"A2C", "PPO"})
                self.assertEqual(set(snapshots["agent"]), {"A2C", "PPO"})

                frames = _discover_dataframes(staged)
                frame_vars = {frame["var"] for frame in frames}
                self.assertIn("df_transactions", frame_vars)
                self.assertIn("df_snapshots", frame_vars)
                self.assertIn("df_account_values", frame_vars)
            finally:
                os.chdir(prev_cwd)


if __name__ == "__main__":
    unittest.main()
