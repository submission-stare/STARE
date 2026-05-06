"""Tests for data.ticker_presets module and integration with experiment configs."""

import pytest
from data.ticker_presets import (
    DJIA_2019,
    DJIA_LIU_ET_AL_2020,
    DJIA_POST_2020_RESHUFFLE,
    PRESETS,
    resolve_tickers,
    resolve_benchmark_tickers,
)


# ---------------------------------------------------------------------------
# Preset data integrity
# ---------------------------------------------------------------------------

def test_djia_2019_has_29_tickers():
    # 29 not 30: DOW (Dow Inc.) only has price history from 2019-03-20 (IPO);
    # the slot's predecessor DWDP (DowDuPont) was de-listed.  See preset comment.
    assert len(DJIA_2019) == 29
    assert len(set(DJIA_2019)) == 29  # no duplicates


def test_djia_2019_contains_pre_reshuffle_tickers():
    assert "XOM" in DJIA_2019
    assert "PFE" in DJIA_2019
    assert "RTX" in DJIA_2019
    # DOW omitted: Dow Inc. IPO was 2019-03-20; predecessor DWDP de-listed.
    assert "DOW" not in DJIA_2019
    assert "WBA" in DJIA_2019


def test_djia_2019_does_not_contain_post_reshuffle_additions():
    assert "CRM" not in DJIA_2019
    assert "AMGN" not in DJIA_2019
    assert "HON" not in DJIA_2019


def test_djia_post_reshuffle_has_28_tickers():
    assert len(DJIA_POST_2020_RESHUFFLE) == 28
    assert len(set(DJIA_POST_2020_RESHUFFLE)) == 28


def test_djia_post_reshuffle_contains_added_tickers():
    assert "CRM" in DJIA_POST_2020_RESHUFFLE
    assert "AMGN" in DJIA_POST_2020_RESHUFFLE
    assert "HON" in DJIA_POST_2020_RESHUFFLE


def test_djia_post_reshuffle_does_not_contain_removed_tickers():
    assert "XOM" not in DJIA_POST_2020_RESHUFFLE
    assert "PFE" not in DJIA_POST_2020_RESHUFFLE
    assert "RTX" not in DJIA_POST_2020_RESHUFFLE
    assert "DOW" not in DJIA_POST_2020_RESHUFFLE
    assert "WBA" not in DJIA_POST_2020_RESHUFFLE


def test_liu_et_al_preset_matches_post_reshuffle():
    assert DJIA_LIU_ET_AL_2020 == DJIA_POST_2020_RESHUFFLE


def test_presets_dict_has_all_known_presets():
    assert "djia_2019" in PRESETS
    assert "djia_post_2020_reshuffle" in PRESETS
    assert "djia_liu_et_al_2020" in PRESETS


# ---------------------------------------------------------------------------
# resolve_tickers
# ---------------------------------------------------------------------------

def test_resolve_tickers_from_preset():
    config = {"ticker_preset": "djia_2019"}
    tickers = resolve_tickers(config)
    assert tickers == DJIA_2019
    # must be a copy, not the same object
    assert tickers is not DJIA_2019


def test_resolve_tickers_preset_case_insensitive():
    config = {"ticker_preset": "DJIA_2019"}
    tickers = resolve_tickers(config)
    assert tickers == DJIA_2019


def test_resolve_tickers_from_explicit_list():
    config = {"tickers": ["AAPL", "MSFT"]}
    assert resolve_tickers(config) == ["AAPL", "MSFT"]


def test_resolve_tickers_preset_takes_precedence_over_list():
    config = {"ticker_preset": "djia_2019", "tickers": ["AAPL"]}
    tickers = resolve_tickers(config)
    assert len(tickers) == 29  # preset wins


def test_resolve_tickers_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown ticker_preset"):
        resolve_tickers({"ticker_preset": "not_a_real_preset"})


def test_resolve_tickers_empty_config_raises():
    with pytest.raises(ValueError, match="must specify"):
        resolve_tickers({})


def test_resolve_tickers_empty_list_raises():
    with pytest.raises(ValueError, match="must specify"):
        resolve_tickers({"tickers": []})


# ---------------------------------------------------------------------------
# Survivorship bias: specific ticker differences
# ---------------------------------------------------------------------------

def test_survivorship_bias_delta():
    """The difference between the two sets should be exactly the reshuffle."""
    added = set(DJIA_POST_2020_RESHUFFLE) - set(DJIA_2019)
    removed = set(DJIA_2019) - set(DJIA_POST_2020_RESHUFFLE)
    assert added == {"CRM", "AMGN", "HON"}
    # DOW omitted from djia_2019 (no usable price history before 2019-03-20)
    assert removed == {"XOM", "PFE", "RTX", "WBA"}


# ---------------------------------------------------------------------------
# resolve_benchmark_tickers
# ---------------------------------------------------------------------------

def test_resolve_benchmark_tickers_from_dedicated_preset():
    config = {"ticker_preset": "djia_liu_et_al_2020", "benchmark_ticker_preset": "djia_2019"}
    bm_tickers = resolve_benchmark_tickers(config)
    assert bm_tickers == DJIA_2019
    assert len(bm_tickers) == 29


def test_resolve_benchmark_tickers_from_explicit_list():
    config = {"ticker_preset": "djia_liu_et_al_2020", "benchmark_tickers": ["AAPL", "MSFT", "GOOG"]}
    bm_tickers = resolve_benchmark_tickers(config)
    assert bm_tickers == ["AAPL", "MSFT", "GOOG"]


def test_resolve_benchmark_tickers_falls_back_to_agent_tickers():
    config = {"ticker_preset": "djia_liu_et_al_2020"}
    bm_tickers = resolve_benchmark_tickers(config)
    agent_tickers = resolve_tickers(config)
    assert bm_tickers == agent_tickers


def test_resolve_benchmark_preset_takes_precedence_over_list():
    config = {
        "ticker_preset": "djia_liu_et_al_2020",
        "benchmark_ticker_preset": "djia_2019",
        "benchmark_tickers": ["AAPL"],
    }
    bm_tickers = resolve_benchmark_tickers(config)
    assert len(bm_tickers) == 29


def test_resolve_benchmark_unknown_preset_raises():
    config = {"ticker_preset": "djia_liu_et_al_2020", "benchmark_ticker_preset": "fake"}
    with pytest.raises(ValueError, match="Unknown ticker_preset"):
        resolve_benchmark_tickers(config)
