"""Tests for LLM agent integration in the unified run pipeline.

Validates that LLM aliases listed in config['agents'] are resolved
from config['llm_openrouter_models'], built via
build_openrouter_strategist_class, and evaluated with llm_agent_params.
"""

import os
from unittest import mock

import numpy as np
import pytest

from agents.llms.llm_strategist import build_openrouter_strategist_class
from evaluation.experiment_setup import resolve_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "agents": ["ppo", "LLM_GPT"],
    "ppo_params": {"learning_rate": 1e-4},
    "timesteps_per_model": 100,
    "llm_timesteps": 1,
    "llm_openrouter_models": {
        "LLM_GPT": "openai/gpt-5.4",
        "LLM_Sonnet": "anthropic/claude-sonnet-4.6",
    },
    "llm_agent_params": {
        "api_key_env_var": "OPENROUTER_API_KEY",
        "verbose": 0,
        "temperature": 0.1,
    },
}


# ---------------------------------------------------------------------------
# resolve_agent
# ---------------------------------------------------------------------------


class TestResolveAgent:
    """Unit tests for evaluation.experiment_setup.resolve_agent."""

    def test_resolves_drl_agent(self):
        cls, kwargs, timesteps = resolve_agent("ppo", MINIMAL_CONFIG, n_assets=5)
        from stable_baselines3 import PPO
        assert cls is PPO
        assert timesteps == 100
        assert "learning_rate" in kwargs

    def test_resolves_llm_agent(self):
        cls, kwargs, timesteps = resolve_agent("LLM_GPT", MINIMAL_CONFIG, n_assets=5)
        assert timesteps == 1
        assert "temperature" in kwargs
        assert kwargs["temperature"] == 0.1
        # The class should be a dynamically built LLMStrategist subclass
        assert "OpenRouterStrategist" in cls.__name__

    def test_llm_agent_receives_model_name(self):
        cls, kwargs, timesteps = resolve_agent("LLM_GPT", MINIMAL_CONFIG, n_assets=5)
        # Instantiate with a mock env to verify model_name binding
        with mock.patch("agents.llms.llm_agent.qs", new=mock.MagicMock()):
            dummy_env = mock.MagicMock()
            dummy_env.action_space.shape = (5,)
            dummy_env.rl_mode = "trading"
            instance = cls(env=dummy_env, api_key="test-key", **kwargs)
            assert instance.model_name == "openai/gpt-5.4"

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            resolve_agent("unknown_agent", MINIMAL_CONFIG, n_assets=5)

    def test_llm_alias_not_in_models_raises(self):
        config = {**MINIMAL_CONFIG, "llm_openrouter_models": {"LLM_Other": "x/y"}}
        with pytest.raises(ValueError, match="Unknown agent"):
            resolve_agent("LLM_GPT", config, n_assets=5)

    def test_llm_defaults_timesteps_to_1(self):
        config = {**MINIMAL_CONFIG}
        del config["llm_timesteps"]
        _, _, timesteps = resolve_agent("LLM_GPT", config, n_assets=5)
        assert timesteps == 1

    def test_llm_defaults_empty_params(self):
        config = {**MINIMAL_CONFIG}
        del config["llm_agent_params"]
        _, kwargs, _ = resolve_agent("LLM_GPT", config, n_assets=5)
        assert isinstance(kwargs, dict)
