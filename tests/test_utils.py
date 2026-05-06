import pytest
from evaluation.utils import _resolve_rl_mode, _resolve_technical_indicator_columns

def test_resolve_rl_mode_valid():
    assert _resolve_rl_mode({"rl_mode": "portfolio"}) == "portfolio"
    assert _resolve_rl_mode({"rl_mode": "trading\n "}) == "trading"
    
def test_resolve_rl_mode_default():
    assert _resolve_rl_mode({}) == "portfolio"

def test_resolve_rl_mode_invalid():
    with pytest.raises(ValueError):
        _resolve_rl_mode({"rl_mode": "invalid_mode"})


def test_resolve_technical_indicator_columns_default_and_custom():
    default_columns = _resolve_technical_indicator_columns({})
    assert isinstance(default_columns, list)
    assert "macd" in default_columns

    assert _resolve_technical_indicator_columns({"technical_indicator_columns": ["MACD", "RSI"]}) == ["macd", "rsi_30"]


def test_resolve_technical_indicator_columns_invalid():
    with pytest.raises(ValueError):
        _resolve_technical_indicator_columns({"technical_indicator_columns": ["not_real"]})
