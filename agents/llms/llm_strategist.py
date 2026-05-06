import json
import re
from typing import Any, Dict, Iterable
from urllib import error, request

import numpy as np
import pandas as pd

from agents.llms.llm_agent import LLMAgent


class LLMStrategist(LLMAgent):
    """
    Two-stage LLM trading agent:
    1) Select a trading strategy once at the beginning.
    2) Produce daily buy/sell/hold actions conditioned on latest trade results.
    """

    def __init__(
        self,
        strategy_candidates: Iterable[str] | None = None,
        strategy_max_tokens: int = 100,
        decision_max_tokens: int | None = None,
        include_trade_feedback: bool = True,
        blind_mode: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.blind_mode = bool(blind_mode)
        self.strategy_candidates = list(
            strategy_candidates
            if strategy_candidates is not None
            else [
                "trend_following",
                "mean_reversion",
                "breakout",
                "volatility_scalping",
                "risk_off",
            ]
        )
        self.strategy_max_tokens = int(max(16, strategy_max_tokens))
        self.decision_max_tokens = int(max(self.max_tokens, 256) if decision_max_tokens is None else max(64, decision_max_tokens))
        self.include_trade_feedback = bool(include_trade_feedback)
        self.selected_strategy: str | None = None
        self.selected_strategy_payload: Dict[str, Any] = {}
        self.last_decision_rationale: str = ""

    def _current_day_text(self) -> str:
        if self.env is None:
            return "unknown"
        try:
            step = int(getattr(self.env, "current_step", 0))
            dates = getattr(self.env, "dates", None)
            if dates is None or len(dates) == 0:
                return str(step)
            day = pd.Timestamp(dates[min(step, len(dates) - 1)])
            return str(day.date())
        except Exception:
            return "unknown"

    def _summarize_last_trade_feedback(self) -> str:
        if (self.env is None) or (not self.include_trade_feedback):
            return "trade_feedback=disabled"

        tx_fn = getattr(self.env, "get_episode_transactions", None)
        if tx_fn is None:
            return "trade_feedback=unavailable"

        try:
            tx = tx_fn()
        except Exception:
            return "trade_feedback=unavailable"

        if tx is None or not isinstance(tx, pd.DataFrame) or tx.empty:
            return "trade_feedback=no_transactions_yet"

        if "step" in tx.columns:
            last_step = int(tx["step"].max())
            tx = tx[tx["step"] == last_step]

        buys = int((tx.get("action_type") == "BUY").sum()) if "action_type" in tx.columns else 0
        sells = int((tx.get("action_type") == "SELL").sum()) if "action_type" in tx.columns else 0
        holds = int((tx.get("action_type") == "HOLD").sum()) if "action_type" in tx.columns else 0
        gross_buy = float(tx[tx.get("action_type") == "BUY"].get("gross_value", pd.Series(dtype=float)).sum()) if "action_type" in tx.columns else 0.0
        gross_sell = float(tx[tx.get("action_type") == "SELL"].get("gross_value", pd.Series(dtype=float)).sum()) if "action_type" in tx.columns else 0.0
        portfolio_value = float(getattr(self.env, "portfolio_value", np.nan))
        return (
            f"last_trade_result=buys:{buys},sells:{sells},holds:{holds},"
            f"gross_buy:{round(gross_buy, 4)},gross_sell:{round(gross_sell, 4)},"
            f"portfolio_value:{round(portfolio_value, 4)}"
        )

    def _strategy_system_prompt(self) -> str:
        return (
            "You are a quantitative strategist selecting ONE trading style for the episode. "
            "Return strict JSON only with one key: strategy."
        )

    def _strategy_user_prompt(self, observation: Any) -> str:
        state_txt = self.format_state_to_text(observation)
        candidates = ", ".join(self.strategy_candidates)
        return (
            "Choose one strategy for this market episode and keep it consistent.\n"
            f"Allowed strategies: [{candidates}]\n"
            "Output ONLY this shape: {\"strategy\":\"one_allowed_value\"}. Do not include rationale.\n"
            f"day={self._current_day_text()}; {state_txt}"
        )

    def _extract_strategy_fallback(self, text: str) -> str | None:
        match = re.search(r'"strategy"\s*:\s*"([^"\\]+)"', text)
        if match:
            return str(match.group(1)).strip()
        return None

    def _extract_action_fallback(self, text: str) -> np.ndarray:
        # Prefer parsing a bracketed list after "action" when available.
        action_match = re.search(r'"action"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if action_match:
            inside = action_match.group(1)
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", inside)
            if nums:
                return np.asarray([float(n) for n in nums], dtype=float)

        # Fallback for partially truncated content without a closing bracket.
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if nums:
            vals = np.asarray([float(n) for n in nums], dtype=float)
            if vals.size >= self.action_dim:
                return vals[: self.action_dim]
            return vals
        return np.asarray([], dtype=float)

    def _extract_rationale_fallback(self, text: str) -> str:
        for key in ("rationale_short", "rationale", "reason"):
            match = re.search(rf'"{key}"\s*:\s*"([^"\\]*)', text)
            if match:
                return str(match.group(1)).strip()
        return ""

    def _ensure_long_only_bootstrap_trade(self, action: np.ndarray, raw: np.ndarray) -> np.ndarray:
        if not self.is_trading_mode:
            return action
        if np.any(action > 0.0):
            return action
        if self.env is None:
            return action

        holdings = np.asarray(getattr(self.env, "holdings", []), dtype=float)
        if holdings.size == 0 or np.any(holdings > 0):
            return action

        # Long-only env starts with zero inventory. If model emits only negative
        # signals, nothing is executed; bootstrap small buys on strongest signals.
        out = action.copy()
        k = min(3, self.action_dim)
        scores = np.abs(np.asarray(raw, dtype=float).reshape(-1))
        if scores.size < self.action_dim:
            scores = np.pad(scores, (0, self.action_dim - scores.size), mode="constant")
        top_idx = np.argsort(scores)[-k:]
        out[top_idx] = 0.25
        return np.clip(out, -1.0, 1.0)

    def _decision_system_prompt(self) -> str:
        if self.is_trading_mode:
            return (
                "You are a disciplined trading executor. "
                "Given a fixed strategy and latest trade results, output ONLY JSON with key 'action'. "
                "Action values may be numbers in [-1,1] or tokens buy/sell/hold. "
                "Optionally include 'rationale_short' with <= 18 words. "
                "Do not include markdown or extra keys."
            )
        return (
            "You are a portfolio allocator with a fixed strategy. "
            "Output ONLY JSON with key 'action' as non-negative weights summing to 1."
        )

    def _state_semantics_text(self) -> str:
        n = self.action_dim
        total_len = n * 6 + 1 if self.is_trading_mode else -1
        if self.is_trading_mode:
            return (
                "State semantics: the env observation has "
                f"{total_len} features in this order -> "
                f"[0:{n}) normalized prices, "
                f"[{n}:{2*n}) MACD, "
                f"[{2*n}:{3*n}) RSI/100, "
                f"[{3*n}:{4*n}) stochastic oscillator/100, "
                f"[{4*n}:{5*n}) FR, "
                f"[{5*n}:{6*n}) normalized current holdings, "
                f"[{6*n}] cash weight."
            )
        return "State semantics: state_head is a compact preview of the full observation vector."

    def _action_semantics_text(self) -> str:
        if self.is_trading_mode:
            return (
                "Action semantics: return exactly one action value per asset. "
                "Each value must be in [-1,1]. Positive means buy pressure, "
                "negative means sell pressure, near zero means hold."
            )
        return (
            "Action semantics: return one non-negative weight per asset plus cash (index 0), "
            "and weights must sum to 1."
        )

    def _execution_constraints_text(self) -> str:
        if not self.is_trading_mode:
            return "Execution constraints: ensure diversification and risk control."

        return (
            "Long-only execution constraints: sell orders only execute if shares already exist; "
            "with zero holdings, negative actions effectively become HOLD. "
            "So if all holdings are zero, include at least one positive action to open positions."
        )

    def _action_index_map_text(self) -> str:
        if self.blind_mode or self.env is None:
            return "Action index map: unavailable"
        tickers = list(getattr(self.env, "tickers", []))
        if not tickers:
            return "Action index map: unavailable"
        pairs = [f"action[{i}] -> {t}" for i, t in enumerate(tickers[: self.action_dim])]
        return "Action index map: " + ", ".join(pairs)

    def _portfolio_feedback_text(self) -> str:
        if self.env is None:
            return "Portfolio feedback: unavailable"

        current_value = float(getattr(self.env, "portfolio_value", np.nan))
        initial_capital = float(getattr(self.env, "initial_capital", np.nan))
        cash = float(getattr(self.env, "cash", np.nan))
        allocated = np.nan

        snapshots_fn = getattr(self.env, "get_episode_snapshots", None)
        if callable(snapshots_fn):
            try:
                snaps = snapshots_fn()
                if isinstance(snaps, pd.DataFrame) and not snaps.empty:
                    latest = snaps.iloc[-1]
                    if "account_value_after" in latest:
                        current_value = float(latest["account_value_after"])
                    if "cash" in latest:
                        cash = float(latest["cash"])
                    if "total_allocated" in latest:
                        allocated = float(latest["total_allocated"])
            except Exception:
                pass

        ret = 0.0
        if np.isfinite(current_value) and np.isfinite(initial_capital) and initial_capital > 1e-12:
            ret = (current_value / initial_capital) - 1.0

        alloc_ratio = 0.0
        if np.isfinite(allocated) and np.isfinite(current_value) and current_value > 1e-12:
            alloc_ratio = allocated / current_value
        elif np.isfinite(cash) and np.isfinite(current_value) and current_value > 1e-12:
            alloc_ratio = 1.0 - (cash / current_value)

        return (
            "Portfolio feedback: "
            f"portfolio_return_since_start={ret:.6f}, "
            f"portfolio_value={current_value:.2f}, "
            f"cash={cash:.2f}, "
            f"allocated_ratio={alloc_ratio:.6f}"
        )

    def _decision_user_prompt(self, observation: Any) -> str:
        state_txt = self.format_state_to_text(observation)
        feedback = self._summarize_last_trade_feedback()
        strategy = self.selected_strategy or "risk_off"
        return (
            "Use the selected strategy and latest trade feedback to decide next action.\n"
            f"selected_strategy={strategy}; day={self._current_day_text()}; {feedback}\n"
            f"{self._portfolio_feedback_text()}\n"
            f"{self._action_index_map_text()}\n"
            f"{self._state_semantics_text()}\n"
            f"{self._action_semantics_text()}\n"
            f"{self._execution_constraints_text()}\n"
            "Return strict JSON only, for example: "
            '{"action":[0.2,-0.1,0.0],"rationale_short":"brief reason"}.\n'
            f"'action' must have exactly {self.action_dim} values.\n"
            f"{state_txt}"
        )

    def _request_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        event_prefix: str,
    ) -> Dict[str, Any]:
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
            "max_tokens": int(max_tokens),
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

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
                    f"{event_prefix}_request",
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
                    f"{event_prefix}_response",
                    {
                        "attempt": attempt,
                        "response_body": body,
                        "response_content": content,
                    },
                )

                if self.verbose >= 2:
                    preview = content[:220].replace("\n", " ")
                    self._log(f"{event_prefix} response preview: {preview}")

                try:
                    return self._extract_json(content)
                except ValueError:
                    if event_prefix == "strategy":
                        fallback_strategy = self._extract_strategy_fallback(content)
                        if fallback_strategy:
                            if self.verbose:
                                self._log("strategy JSON was truncated; fallback strategy extraction applied")
                            return {"strategy": fallback_strategy}
                    if event_prefix == "decision":
                        fallback_action = self._extract_action_fallback(content)
                        if fallback_action.size > 0:
                            fallback_rationale = self._extract_rationale_fallback(content)
                            if self.verbose:
                                self._log("decision JSON was truncated; fallback action extraction applied")
                            payload = {"action": fallback_action.tolist()}
                            if fallback_rationale:
                                payload["rationale_short"] = fallback_rationale
                            return payload
                    raise
            except error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                last_error = RuntimeError(f"HTTPError {exc.code}: {exc.reason}. body={body[:300]}")
                self._append_llm_log(
                    f"{event_prefix}_error",
                    {
                        "attempt": attempt,
                        "error_type": "HTTPError",
                        "error_message": str(last_error),
                    },
                )
            except error.URLError as exc:
                last_error = RuntimeError(f"URLError during request: {exc}")
                self._append_llm_log(
                    f"{event_prefix}_error",
                    {
                        "attempt": attempt,
                        "error_type": "URLError",
                        "error_message": str(last_error),
                    },
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"failed to parse model response: {exc}")
                self._append_llm_log(
                    f"{event_prefix}_error",
                    {
                        "attempt": attempt,
                        "error_type": "ParseError",
                        "error_message": str(last_error),
                    },
                )

            if self.verbose:
                self._log(
                    f"{event_prefix} attempt {attempt}/{self.max_request_retries} failed: {last_error}"
                )

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_request_retries} retries"
        ) from last_error

    def _select_strategy_once(self, observation: Any) -> None:
        if self.selected_strategy is not None:
            return

        parsed = self._request_json(
            [
                {"role": "system", "content": self._strategy_system_prompt()},
                {"role": "user", "content": self._strategy_user_prompt(observation)},
            ],
            max_tokens=self.strategy_max_tokens,
            event_prefix="strategy",
        )

        strategy = str(
            parsed.get("strategy")
            or parsed.get("trading_strategy")
            or parsed.get("style")
            or "risk_off"
        ).strip()
        if strategy not in self.strategy_candidates:
            strategy = "risk_off"

        self.selected_strategy = strategy
        self.selected_strategy_payload = parsed

        if self.verbose:
            self._log(f"selected strategy={self.selected_strategy}")

    @staticmethod
    def _map_token_to_signal(token: str) -> float:
        cleaned = token.strip().lower()
        mapping = {
            "buy": 1.0,
            "long": 1.0,
            "sell": -1.0,
            "short": -1.0,
            "hold": 0.0,
            "neutral": 0.0,
        }
        if cleaned in mapping:
            return mapping[cleaned]
        return float(cleaned)

    def _action_to_array(self, action_value: Any) -> np.ndarray:
        if isinstance(action_value, (list, tuple)):
            out: list[float] = []
            for item in action_value:
                if isinstance(item, str):
                    try:
                        out.append(self._map_token_to_signal(item))
                    except ValueError:
                        pass
                else:
                    try:
                        out.append(float(item))
                    except (TypeError, ValueError):
                        pass
            if out:
                return np.asarray(out, dtype=float)

        if isinstance(action_value, str):
            words = [w for w in action_value.replace(",", " ").split() if w.strip()]
            mapped: list[float] = []
            for w in words:
                try:
                    mapped.append(self._map_token_to_signal(w))
                except ValueError:
                    continue
            if mapped:
                return np.asarray(mapped, dtype=float)

        return super()._action_to_array(action_value)

    def predict(self, observation, deterministic=True):
        """
        First call selects strategy. Every call sends latest day result and receives action.
        """
        self._predict_calls += 1
        self._select_strategy_once(observation)

        parsed = self._request_json(
            [
                {"role": "system", "content": self._decision_system_prompt()},
                {"role": "user", "content": self._decision_user_prompt(observation)},
            ],
            max_tokens=self.decision_max_tokens,
            event_prefix="decision",
        )

        raw = self._action_to_array(parsed.get("action", parsed))
        if raw.size == 0:
            raise RuntimeError("response JSON had no 'action' values")

        rationale = str(
            parsed.get("rationale_short")
            or parsed.get("rationale")
            or parsed.get("reason")
            or ""
        ).strip()
        self.last_decision_rationale = rationale
        if rationale:
            self._append_llm_log(
                "decision_rationale",
                {
                    "rationale": rationale,
                    "strategy": self.selected_strategy or "",
                    "day": self._current_day_text(),
                },
            )

        action = self._normalize_action(raw)
        action = self._ensure_long_only_bootstrap_trade(action, raw)
        fallback_action = self._fallback_action()
        if np.allclose(action, fallback_action):
            self._fallback_count += 1

        if self.verbose and (self._predict_calls % self.log_every_n_steps == 0):
            self._log(
                f"predict_calls={self._predict_calls} api_calls={self._api_calls} "
                f"fallbacks={self._fallback_count} strategy={self.selected_strategy}"
            )
        return action, None


def build_openrouter_strategist_class(model_name: str):
    """Factory to create evaluate_agent-compatible classes for fixed OpenRouter strategist models."""

    class _FixedModelLLMStrategist(LLMStrategist):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("model_name", model_name)
            super().__init__(*args, **kwargs)

    _FixedModelLLMStrategist.__name__ = f"OpenRouterStrategist_{model_name.replace('/', '_').replace('-', '_')}"
    return _FixedModelLLMStrategist
