import pytest
import os
import pandas as pd
import numpy as np
from evaluation.plotting import plot_performance

def test_plot_performance(tmpdir):
    dates = pd.date_range("2020-01-01", periods=100)
    # create some random walk data
    vals = np.cumsum(np.random.normal(0.001, 0.01, 100)) + 1.0
    account_values = pd.Series(vals, index=dates)
    
    metrics = {
        "AR": 0.1,
        "SR": 1.5,
        "PSR": 0.98,
        "DSR": 0.95,
        "MaxDrawdown": 0.15
    }
    
    plot_performance("TestModel", account_values, metrics, str(tmpdir), None)
    
    expected_file = os.path.join(str(tmpdir), "TestModel_performance.png")
    assert os.path.exists(expected_file)
    assert os.path.getsize(expected_file) > 1000  # Should be a real image
