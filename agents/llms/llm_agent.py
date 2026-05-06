import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict
from urllib import error, request

import numpy as np
import pandas as pd

try:
    import quantstats as qs
except ImportError:
    qs = None


class LLMAgent:
    """
    LLM-for-RL agent compatible with the project's evaluate_agent pipeline.

    This class can be passed to evaluation.runner.evaluate_agent similarly to SB3
    classes because it implements a no-op learn() and predict().
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    _ENV_LOADED = False

    def __init__(
        self,
        policy: str = "MlpPolicy",
        env: Any = None,
        verbose: int = 0,
        learning_rate: float = 3e-4,
        api_key: str | None = None,
        api_key_env_var: str = "OPENROUTER_API_KEY",
        model_name: str = "openai/gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 180,
        timeout_seconds: int = 30,
        max_api_calls: int = 10000,
        max_request_retries: int = 10,
        log_every_n_steps: int = 50,
        quantstats_lookback: int = 126,
        **_: Any,
    ):
        self._load_project_env_once()

        if qs is None:
            raise ImportError(
                "quantstats is required for LLMAgent context. Install with: pip install quantstats"
            )

        env_verbose = os.getenv("LLM_AGENT_VERBOSE")
        if env_verbose is not None and str(env_verbose).strip() != "":
            try:
                verbose = int(env_verbose)
            except ValueError:
                pass

        env_log_every = os.getenv("LLM_AGENT_LOG_EVERY_N_STEPS")
        if env_log_every is not None and str(env_log_every).strip() != "":
            try:
                log_every_n_steps = int(env_log_every)
            except ValueError:
                pass

        self.policy = policy
        self.env = env
        self.verbose = int(verbose)
        self.learning_rate = float(learning_rate)
        self.api_key = api_key or os.getenv(api_key_env_var) or os.getenv("openrouter") or ""
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.max_api_calls = int(max_api_calls)
        self.max_request_retries = int(max(1, max_request_retries))
        self.log_every_n_steps = int(max(1, log_every_n_steps))
        self.quantstats_lookback = int(max(20, quantstats_lookback))
        self._api_calls = 0
        self._predict_calls = 0
        self._fallback_count = 0
        self._warned_missing_key = False
        self._warned_api_limit = False
        self._warned_http_error = False
        self._warned_parse_error = False
        self._warned_quantstats_missing = False
        self._llm_log_path: str | None = None

        self.action_dim = self._infer_action_dim(env)
        self.is_trading_mode = bool(getattr(env, "rl_mode", "portfolio") == "trading")

        if self.verbose:
            mode = "trading" if self.is_trading_mode else "portfolio"
            print(
                f"[LLM:{self.model_name}] initialized | mode={mode} action_dim={self.action_dim} "
                f"timeout={self.timeout_seconds}s max_api_calls={self.max_api_calls} retries={self.max_request_retries}"
            )

    @classmethod
    def _load_project_env_once(cls) -> None:
        if cls._ENV_LOADED:
            return

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(project_root, ".env")
        if not os.path.exists(env_path):
            cls._ENV_LOADED = True
            return

        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
            cls._ENV_LOADED = True
            return
        except ImportError:
            pass

        # Fallback parser when python-dotenv is unavailable.
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        cls._ENV_LOADED = True

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LLM:{self.model_name}] {message}")

    def _get_llm_log_path(self) -> str:
        if self._llm_log_path is not None:
            return self._llm_log_path

        run_dir = None
        try:
            from evaluation.runner import _get_or_create_run_dir

            run_dir = _get_or_create_run_dir()
        except Exception:
            run_dir = os.path.join(os.getcwd(), "results", datetime.now().strftime("%Y%m%d%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)

        self._llm_log_path = os.path.join(run_dir, "llm_logs.jsonl")
        return self._llm_log_path

    def _append_llm_log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "model": self.model_name,
            "predict_call": int(self._predict_calls),
            "api_call": int(self._api_calls),
            **payload,
        }
        log_path = self._get_llm_log_path()
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _infer_action_dim(env: Any) -> int:
        if env is None:
            return 1
        shape = getattr(getattr(env, "action_space", None), "shape", None)
        if shape and len(shape) > 0:
            return int(shape[0])
        tickers = getattr(env, "tickers", [])
        if getattr(env, "rl_mode", "portfolio") == "trading":
            return max(1, len(tickers))
        return max(1, len(tickers) + 1)

    def _fallback_action(self) -> np.ndarray:
        if self.is_trading_mode:
            return np.zeros(self.action_dim, dtype=float)
        out = np.zeros(self.action_dim, dtype=float)
        out[0] = 1.0
        return out

    def learn(self, total_timesteps: int):
        # Keep compatibility with evaluate_agent().
        if self.verbose:
            print(f"[LLM:{self.model_name}] no-op learn() for {total_timesteps} timesteps")
        return self

    def format_state_to_text(self, observation) -> str:
        """
        Converts numerical state to compact textual summary.
        """
        obs = np.asarray(observation, dtype=float).flatten()
        head = np.round(obs[:24], 6).tolist()

        qs_summary = self._build_quantstats_summary()
        qs_text = "qs_unavailable"
        if qs_summary:
            qs_text = ", ".join(f"{k}={v}" for k, v in qs_summary.items())

        return (
            f"rl_mode={getattr(self.env, 'rl_mode', 'portfolio')}; "
            f"action_dim={self.action_dim}; "
            f"state_head={head}; "
            f"state_len={len(obs)}; "
            f"quantstats={qs_text}"
        )

    def _build_quantstats_summary(self) -> Dict[str, float]:
        if qs is None:
            if (not self._warned_quantstats_missing) and self.verbose:
                self._warned_quantstats_missing = True
                self._log("quantstats not installed; continuing without quantstats context")
            return {}

        if self.env is None or not hasattr(self.env, "C"):
            return {}

        try:
            prices = np.asarray(getattr(self.env, "C"), dtype=float)
            if prices.ndim != 2 or prices.shape[0] < 3:
                return {}

            current_step = int(getattr(self.env, "current_step", prices.shape[0] - 1))
            end_idx = max(1, min(current_step, prices.shape[0] - 1))
            start_idx = max(0, end_idx - self.quantstats_lookback)
            window = prices[start_idx : end_idx + 1, :]

            window = np.where(window > 0.0, window, np.nan)
            eq_idx = pd.Series(np.nanmean(window, axis=1), dtype=float).ffill().bfill()
            dates = getattr(self.env, "dates", None)
            if dates is not None:
                try:
                    date_window = pd.to_datetime(dates[start_idx : end_idx + 1])
                    if len(date_window) == len(eq_idx):
                        eq_idx.index = date_window
                except Exception:
                    pass
            returns = eq_idx.pct_change().dropna()
            if returns.empty:
                return {}

            sharpe = float(qs.stats.sharpe(returns))
            sortino = float(qs.stats.sortino(returns))
            max_dd = float(qs.stats.max_drawdown(returns))
            volatility = float(qs.stats.volatility(returns))
            win_rate = float((returns > 0).mean())

            def _safe(v: float) -> float:
                return float(np.round(v if np.isfinite(v) else 0.0, 6))

            return {
                "qs_sharpe": _safe(sharpe),
                "qs_sortino": _safe(sortino),
                "qs_max_drawdown": _safe(max_dd),
                "qs_volatility": _safe(volatility),
                "qs_win_rate": _safe(win_rate),
            }
        except Exception as exc:
            if self.verbose >= 2:
                self._log(f"quantstats context error: {exc}")
            return {}

    def _system_prompt(self) -> str:
        if self.is_trading_mode:
            return (
                "You are an execution-focused quantitative trading assistant. "
                "Output ONLY JSON with key 'action' containing real values in [-1, 1]. "
                "Positive means buy pressure, negative means sell pressure, near zero means hold. "
                "Prefer controlled risk and avoid extreme all-in behavior."
            )
        return (
            "You are a portfolio allocation assistant. "
            "Output ONLY JSON with key 'action' containing non-negative portfolio weights that sum to 1. "
            "Index 0 is cash weight. Keep risk-aware diversification."
        )

    def _user_prompt(self, observation) -> str:
        state_txt = self.format_state_to_text(observation)
        return (
            "Given the state snapshot below, propose one-step allocation/trading action.\n"
            "Return strict JSON only, e.g. {\"action\": [..], \"rationale\": \"..\"}.\n"
            f"'action' must be a flat numeric array with exactly {self.action_dim} values.\n"
            f"{state_txt}"
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("LLM response did not contain valid JSON")

    @staticmethod
    def _action_to_array(action_value: Any) -> np.ndarray:
        if isinstance(action_value, dict):
            for key in ("action", "weights", "signals", "vector", "values"):
                if key in action_value:
                    return LLMAgent._action_to_array(action_value[key])
            return np.asarray([], dtype=float)

        if isinstance(action_value, str):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", action_value)
            if not nums:
                return np.asarray([], dtype=float)
            return np.asarray([float(n) for n in nums], dtype=float)

        if np.isscalar(action_value):
            return np.asarray([float(action_value)], dtype=float)

        arr = np.asarray(action_value, dtype=float)
        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=float)
        return arr.reshape(-1)

    def _call_openrouter(self, observation) -> np.ndarray:
        if not self.api_key:
            msg = "missing API key (OPENROUTER_API_KEY/openrouter)"
            if (not self._warned_missing_key) and self.verbose:
                self._warned_missing_key = True
                self._log(msg)
            raise RuntimeError(msg)

        if self._api_calls >= self.max_api_calls:
            msg = "max_api_calls reached"
            if (not self._warned_api_limit) and self.verbose:
                self._warned_api_limit = True
                self._log(msg)
            raise RuntimeError(msg)

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(observation)},
            ],
            "response_format": {"type": "json_object"},
        }

        if self.verbose >= 3:
            self._log(f"system prompt: {payload['messages'][0]['content']}")
            self._log(f"user prompt: {payload['messages'][1]['content']}")

        req = request.Request(
            self.OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_request_retries + 1):
            try:
                self._append_llm_log(
                    "request",
                    {
                        "attempt": attempt,
                        "url": self.OPENROUTER_URL,
                        "request_payload": payload,
                    },
                )
                self._api_calls += 1
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("response content is None")
                if not isinstance(content, str):
                    raise ValueError(f"response content must be string, got {type(content).__name__}")
                self._append_llm_log(
                    "response",
                    {
                        "attempt": attempt,
                        "response_body": body,
                        "response_content": content,
                    },
                )
                if self.verbose >= 3:
                    self._log(f"raw response full: {content}")
                if self.verbose >= 2:
                    preview = content[:220].replace("\n", " ")
                    self._log(f"raw response preview: {preview}")
                try:
                    parsed = self._extract_json(content)
                    arr = self._action_to_array(parsed.get("action", []))
                except ValueError:
                    # Some providers/models ignore JSON mode; try best-effort extraction.
                    arr = self._action_to_array(content)
                    if self.verbose and arr.size > 0:
                        self._log("response was not strict JSON; used numeric extraction from raw text")
                if arr.size == 0:
                    raise ValueError("response JSON had no 'action' values")
                if self.verbose and attempt > 1:
                    self._log(f"request succeeded after retry {attempt}/{self.max_request_retries}")
                return arr
            except error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                last_error = RuntimeError(f"HTTPError {exc.code}: {exc.reason}. body={body[:300]}")
                self._append_llm_log(
                    "error",
                    {
                        "attempt": attempt,
                        "error_type": "HTTPError",
                        "error_message": str(last_error),
                    },
                )
            except error.URLError as exc:
                last_error = RuntimeError(f"URLError during request: {exc}")
                self._append_llm_log(
                    "error",
                    {
                        "attempt": attempt,
                        "error_type": "URLError",
                        "error_message": str(last_error),
                    },
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"failed to parse model response: {exc}")
                self._append_llm_log(
                    "error",
                    {
                        "attempt": attempt,
                        "error_type": "ParseError",
                        "error_message": str(last_error),
                    },
                )

            if self.verbose:
                self._log(
                    f"request attempt {attempt}/{self.max_request_retries} failed: {last_error}"
                )

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_request_retries} retries"
        ) from last_error

    def _normalize_action(self, raw: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.atleast_1d(np.asarray(raw, dtype=float)).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size < self.action_dim:
            arr = np.pad(arr, (0, self.action_dim - arr.size), mode="constant")
        elif arr.size > self.action_dim:
            arr = arr[: self.action_dim]

        if self.is_trading_mode:
            return np.clip(arr, -1.0, 1.0)

        arr = np.maximum(arr, 0.0)
        total = float(arr.sum())
        if total <= 1e-12:
            return self._fallback_action()
        return arr / total

    def predict(self, observation, deterministic=True):
        """
        Maps observation -> OpenRouter response -> env-compatible action vector.
        """
        self._predict_calls += 1
        raw_action = self._call_openrouter(observation)
        action = self._normalize_action(raw_action)
        fallback_action = self._fallback_action()
        if np.allclose(action, fallback_action):
            self._fallback_count += 1

        if self.verbose and (self._predict_calls % self.log_every_n_steps == 0):
            self._log(
                f"predict_calls={self._predict_calls} api_calls={self._api_calls} "
                f"fallbacks={self._fallback_count}"
            )
        return action, None


def build_openrouter_agent_class(model_name: str):
    """Factory to create evaluate_agent-compatible classes for fixed OpenRouter models."""

    class _FixedModelLLMAgent(LLMAgent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("model_name", model_name)
            super().__init__(*args, **kwargs)

    _FixedModelLLMAgent.__name__ = f"OpenRouter_{model_name.replace('/', '_').replace('-', '_')}"
    return _FixedModelLLMAgent
