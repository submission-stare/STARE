import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.rigorous_stats import (
    calculate_expected_max_sr,
    calculate_psr,
    calculate_dsr,
    compute_benchmark_sr_daily,
)

def test_calculate_expected_max_sr():
    trials_sr = [0.5, 1.0, 1.5]
    expected_max = calculate_expected_max_sr(trials_sr)
    assert isinstance(expected_max, float)
    
    # Expected maximum SR of [0.5, 1.0, 1.5] given Euler-Mascheroni should safely be above the mean
    assert expected_max >= np.mean(trials_sr)
    
    # Test empty list guard gracefully
    assert calculate_expected_max_sr([]) == 0.0

def test_calculate_psr():
    # If the SR is significantly higher than the benchmark SR, PSR should be > 50%
    psr_val_high = calculate_psr(sr=1.5, t=252, skewness=0.0, kurtosis=3.0, sr_benchmark=1.0)
    assert 0.5 < psr_val_high <= 1.0

    # If the SR is lower than the benchmark SR, PSR should be < 50%
    psr_val_low = calculate_psr(sr=0.5, t=252, skewness=0.0, kurtosis=3.0, sr_benchmark=1.0)
    assert 0.0 <= psr_val_low < 0.5

def test_calculate_dsr():
    historical_trials = [0.1, 0.5, 1.2]
    # DSR calculates PSR utilizing the expected maximum SR from the historical pool as its benchmark
    dsr_val = calculate_dsr(sr=1.5, t=252, skewness=0.0, kurtosis=3.0, trials_sr=historical_trials)
    assert isinstance(dsr_val, float)
    assert 0.0 <= dsr_val <= 1.0
