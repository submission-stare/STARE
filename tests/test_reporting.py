import pytest
import os
import shutil
import pandas as pd
from evaluation.reporting import _get_or_create_run_dir, save_experiment_results, _summarize_compat_transactions

def test_get_or_create_run_dir(tmpdir):
    # This will test that a directory in 'results' starts to appear
    # We should run it inside a mock environment if we want true isolation,
    # but practically we can just check if os.path.exists works
    dir_path = _get_or_create_run_dir()
    assert os.path.exists(dir_path)
    assert "results" in dir_path
    
def test_save_experiment_results():
    results = {
        "A2C": {"AR": 0.15, "SR": 1.2, "Sortino": 1.6, "PSR": 0.95, "DSR": 0.90, "MaxDrawdown": 0.10}
    }
    save_experiment_results(results, "dummy_results.txt", "Test")
    
    run_dir = _get_or_create_run_dir()
    expected_path = os.path.join(run_dir, "dummy_results.txt")
    assert os.path.exists(expected_path)
    
    with open(expected_path, "r") as f:
        content = f.read()
        assert "A2C" in content
        assert "15.00%" in content
        assert "1.60" in content
        assert "10.00%" in content

def test_summarize_compat_transactions():
    df = pd.DataFrame({
        "action_type": ["BUY", "SELL", "HOLD"],
        "gross_value": [100.0, 50.0, 0.0],
        "transaction_cost": [1.0, 0.5, 0.0]
    })
    res = _summarize_compat_transactions("agent1", df)
    assert res["total_buys"] == 1
    assert res["total_sells"] == 1
    assert res["total_holds"] == 1
    assert res["total_value_buy"] == 100.0
    assert res["total_value_sell"] == 50.0
    assert res["total_cost"] == 1.5

