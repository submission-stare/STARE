import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from urllib import error as urlerror

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llms.llm_agent import LLMAgent, build_openrouter_agent_class


class _DummyActionSpace:
    def __init__(self, shape):
        self.shape = shape


class _DummyEnv:
    def __init__(self, action_dim: int, rl_mode: str):
        self.action_space = _DummyActionSpace((action_dim,))
        self.rl_mode = rl_mode


def test_missing_api_key_raises_runtime_error():
    env = _DummyEnv(action_dim=5, rl_mode="portfolio")
    LLMAgent._ENV_LOADED = True
    os.environ.pop("OPENROUTER_API_KEY", None)
    os.environ.pop("openrouter", None)
    agent = LLMAgent(env=env, api_key="")
    with pytest.raises(RuntimeError, match="missing API key"):
        agent.predict(np.array([1.0, 2.0]))


def test_quantstats_required(monkeypatch):
    monkeypatch.setattr("agents.llms.llm_agent.qs", None)
    with pytest.raises(ImportError, match="quantstats is required"):
        LLMAgent(env=_DummyEnv(action_dim=2, rl_mode="trading"), api_key="dummy")


def test_trading_action_is_clipped_and_sized(monkeypatch):
    env = _DummyEnv(action_dim=3, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy")

    monkeypatch.setattr(agent, "_call_openrouter", lambda obs: np.array([10.0, -2.0, 0.5, 99.0]))

    action, _ = agent.predict(np.array([1.0, 2.0]))
    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)


def test_trading_scalar_action_is_vectorized(monkeypatch):
    env = _DummyEnv(action_dim=4, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy")

    monkeypatch.setattr(agent, "_call_openrouter", lambda obs: 0.25)

    action, _ = agent.predict(np.array([1.0, 2.0]))
    assert action.shape == (4,)
    assert np.isclose(action[0], 0.25)
    assert np.all(action[1:] == 0.0)


def test_factory_sets_model_name():
    env = _DummyEnv(action_dim=2, rl_mode="trading")
    cls = build_openrouter_agent_class("anthropic/claude-3.5-sonnet")
    agent = cls(env=env, api_key="dummy")

    assert agent.model_name == "anthropic/claude-3.5-sonnet"


def test_loads_api_key_from_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    monkeypatch.setattr(
        "agents.llms.llm_agent.os.path.abspath",
        lambda _: str(tmp_path / "agents" / "llms" / "llm_agent.py"),
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Reset one-time loader flag for this test.
    LLMAgent._ENV_LOADED = False
    agent = LLMAgent(env=_DummyEnv(action_dim=2, rl_mode="trading"), api_key=None)

    assert agent.api_key == "test-key"


def test_request_retries_then_raises(monkeypatch):
    env = _DummyEnv(action_dim=2, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy", max_request_retries=3)

    def _always_fail(*args, **kwargs):
        raise urlerror.URLError("network down")

    monkeypatch.setattr("agents.llms.llm_agent.request.urlopen", _always_fail)

    with pytest.raises(RuntimeError, match="failed after 3 retries"):
        agent.predict(np.array([1.0, 2.0]))

    assert agent._api_calls == 3


def test_non_json_response_numeric_extraction(monkeypatch):
    env = _DummyEnv(action_dim=4, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy", max_request_retries=1)

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            body = {
                "choices": [
                    {"message": {"content": "Action suggestion: [0.2, -0.1, 0.0, 0.9]"}}
                ]
            }
            return str(body).replace("'", '"').encode("utf-8")

    monkeypatch.setattr("agents.llms.llm_agent.request.urlopen", lambda *args, **kwargs: _FakeResponse())

    action, _ = agent.predict(np.array([1.0, 2.0]))
    assert action.shape == (4,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)


def test_none_content_retries_then_succeeds(monkeypatch):
    env = _DummyEnv(action_dim=3, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy", max_request_retries=3)

    responses = [
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": '{"action": [0.2, -0.1, 0.4]}'}}]},
    ]

    class _FakeResponse:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._body).encode("utf-8")

    def _urlopen(*args, **kwargs):
        body = responses.pop(0)
        return _FakeResponse(body)

    monkeypatch.setattr("agents.llms.llm_agent.request.urlopen", _urlopen)

    action, _ = agent.predict(np.array([1.0, 2.0]))
    assert action.shape == (3,)
    assert agent._api_calls == 2


def test_env_overrides_verbose_and_log_frequency(monkeypatch):
    monkeypatch.setenv("LLM_AGENT_VERBOSE", "2")
    monkeypatch.setenv("LLM_AGENT_LOG_EVERY_N_STEPS", "7")

    env = _DummyEnv(action_dim=3, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy", verbose=0, log_every_n_steps=50)

    assert agent.verbose == 2
    assert agent.log_every_n_steps == 7


def test_verbose_level_3_logs_prompts_and_response(monkeypatch):
    env = _DummyEnv(action_dim=3, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy", verbose=3, max_request_retries=1)

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            body = {
                "choices": [
                    {"message": {"content": '{"action": [0.1, -0.2, 0.3]}'}}
                ]
            }
            return json.dumps(body).encode("utf-8")

    logs = []
    monkeypatch.setattr("agents.llms.llm_agent.request.urlopen", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(agent, "_log", lambda msg: logs.append(msg))

    action, _ = agent.predict(np.array([1.0, 2.0]))

    assert action.shape == (3,)
    assert any("system prompt:" in item for item in logs)
    assert any("user prompt:" in item for item in logs)
    assert any("raw response full:" in item for item in logs)


def test_writes_jsonl_request_and_response_logs(monkeypatch, tmp_path):
    env = _DummyEnv(action_dim=3, rl_mode="trading")
    agent = LLMAgent(env=env, api_key="dummy", max_request_retries=1)

    log_path = tmp_path / "llm_logs.jsonl"
    monkeypatch.setattr(agent, "_get_llm_log_path", lambda: str(log_path))

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            body = {
                "choices": [
                    {"message": {"content": '{"action": [0.4, 0.1, -0.2]}'}}
                ]
            }
            return json.dumps(body).encode("utf-8")

    monkeypatch.setattr("agents.llms.llm_agent.request.urlopen", lambda *args, **kwargs: _FakeResponse())

    action, _ = agent.predict(np.array([1.0, 2.0]))
    assert action.shape == (3,)
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    records = [json.loads(line) for line in lines]

    assert any(r.get("event") == "request" for r in records)
    assert any(r.get("event") == "response" for r in records)
    assert all("timestamp" in r for r in records)


def test_format_state_contains_quantstats_context(monkeypatch):
    env = _DummyEnv(action_dim=3, rl_mode="trading")
    env.C = np.array(
        [
            [100.0, 200.0],
            [101.0, 202.0],
            [102.0, 203.0],
            [103.0, 205.0],
            [104.0, 206.0],
        ]
    )
    env.current_step = 4

    agent = LLMAgent(env=env, api_key="dummy")

    monkeypatch.setattr(agent, "_build_quantstats_summary", lambda: {"qs_sharpe": 1.23, "qs_sortino": 2.34})

    txt = agent.format_state_to_text(np.array([1.0, 2.0, 3.0]))
    assert "quantstats=" in txt
    assert "qs_sharpe=1.23" in txt
    assert "qs_sortino=2.34" in txt


def test_quantstats_summary_with_env_dates():
    env = _DummyEnv(action_dim=3, rl_mode="trading")
    env.C = np.array(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 203.0],
            [104.0, 205.0],
            [103.0, 206.0],
            [105.0, 207.0],
            [106.0, 208.0],
            [107.0, 209.0],
            [108.0, 210.0],
            [109.0, 211.0],
            [110.0, 212.0],
            [111.0, 213.0],
            [112.0, 214.0],
            [113.0, 215.0],
            [114.0, 216.0],
            [115.0, 217.0],
            [116.0, 218.0],
            [117.0, 219.0],
            [118.0, 220.0],
            [119.0, 221.0],
            [120.0, 222.0],
            [121.0, 223.0],
            [122.0, 224.0],
            [123.0, 225.0],
            [124.0, 226.0],
        ]
    )
    env.dates = pd.date_range("2020-01-01", periods=env.C.shape[0], freq="D")
    env.current_step = env.C.shape[0] - 1

    agent = LLMAgent(env=env, api_key="dummy", quantstats_lookback=20)
    summary = agent._build_quantstats_summary()

    assert isinstance(summary, dict)
    assert "qs_sharpe" in summary
    assert "qs_sortino" in summary
