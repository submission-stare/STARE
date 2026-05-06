import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from data.preprocessors.features import normalize_technical_indicator_selection, resolve_feature_column_name


class TradingEnv(gym.Env):
    """
    Long-only stock trading environment with integer share execution.

    The agent emits one continuous action per ticker in [-1, 1]. Each action is
    scaled by ``hmax`` and rounded toward zero to an integer trade size. Sells
    are processed before buys, transaction costs are applied per trade, and a
    turbulence threshold can force full liquidation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        tickers: list,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.0005,
        turbulence_threshold: float = 1e9,
        hmax: int = 100,
        reward_scaling: float = 1e-4,
        buy_cost_pct: float | None = None,
        sell_cost_pct: float | None = None,
        technical_indicator_columns: list[str] | tuple[str, ...] | str | None = None,
    ):
        super().__init__()
        self.rl_mode = "trading"
        self.data = data
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.buy_cost_pct = float(commission if buy_cost_pct is None else buy_cost_pct)
        self.sell_cost_pct = float(commission if sell_cost_pct is None else sell_cost_pct)
        self.turbulence_threshold = float(turbulence_threshold)
        self.hmax = int(hmax)
        self.reward_scaling = float(reward_scaling)
        self.technical_indicator_columns = normalize_technical_indicator_selection(technical_indicator_columns)

        self.dates = sorted(list(self.data.index.unique()))
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=int)
        self.portfolio_value = self.initial_capital
        self.portfolio_return = 0.0
        self._episode_transactions: list[dict] = []
        self._episode_snapshots: list[dict] = []

        self._build_matrices()

        feature_dim = self.n_assets * (len(self.technical_indicator_columns) + 2) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def _build_matrices(self) -> None:
        self.C = self.data.pivot(columns="Ticker", values="Close").reindex(columns=self.tickers).values
        self.indicator_matrices = {
            column: self._extract_indicator_matrix(column)
            for column in self.technical_indicator_columns
        }
        self.close_scale = np.maximum(np.nanmax(self.C, axis=0), 1e-8)
        if "Turbulence" in self.data.columns:
            self.Turbulence = self.data["Turbulence"].groupby(level=0).first().reindex(self.dates).fillna(0.0).to_numpy()
        else:
            self.Turbulence = np.zeros(len(self.dates), dtype=float)

    def _extract_indicator_matrix(self, column: str) -> np.ndarray:
        resolved = resolve_feature_column_name(self.data, column)
        if resolved is None:
            return np.zeros((len(self.dates), self.n_assets), dtype=float)
        return (
            self.data.pivot(columns="Ticker", values=resolved)
            .reindex(index=self.dates, columns=self.tickers)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def _normalize_indicator_values(self, column: str, values: np.ndarray, prices: np.ndarray) -> np.ndarray:
        safe_prices = np.where(prices < 1e-8, 1.0, prices)

        if column in {"rsi_30", "dx_30", "stoch_k", "stoch_d"}:
            normalized = values / 100.0
        elif column == "cci_30":
            normalized = np.clip(values / 200.0, -5.0, 5.0)
        elif column in {"macd", "atr_14"}:
            normalized = values / safe_prices
        elif column in {"boll_ub", "boll_lb", "close_30_sma", "close_60_sma"}:
            normalized = (values / safe_prices) - 1.0
        elif column in {"qs_sharpe_30", "qs_sortino_30"}:
            normalized = np.clip(values / 10.0, -5.0, 5.0)
        elif column == "return_1d":
            normalized = np.clip(values, -1.0, 1.0)
        elif column == "volatility_20":
            normalized = np.clip(values, 0.0, 1.0)
        elif column == "fr_20":
            normalized = np.clip(values, 0.0, 1.0)
        else:
            normalized = values

        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_prices(self, date_idx: int) -> np.ndarray:
        return np.nan_to_num(self.C[date_idx].astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    def _current_total_asset(self, prices: np.ndarray | None = None) -> float:
        prices = self._get_prices(self.current_step) if prices is None else prices
        return float(self.cash + np.dot(prices, self.holdings.astype(float)))

    def _record_snapshot(self, step: int, date_idx: int, raw_action: np.ndarray | None) -> None:
        prices = self._get_prices(date_idx)
        holdings = self.holdings.astype(float)
        values = prices * holdings
        total_allocated = float(values.sum())
        total_asset = float(self.cash + total_allocated)
        cash_weight = float(self.cash / total_asset) if total_asset > 0 else 1.0

        row = {
            "step": int(step),
            "date": pd.Timestamp(self.dates[date_idx]),
            "account_value_before": float(self.portfolio_value),
            "account_value_after": float(total_asset),
            "step_transaction_cost": float(
                sum(
                    txn["transaction_cost"]
                    for txn in self._episode_transactions
                    if txn["step"] == int(step)
                )
            ),
            "cash": float(self.cash),
            "total_allocated": total_allocated,
            "total_asset": total_asset,
            "cash_weight": cash_weight,
            "cash_raw_action": 0.0,
        }
        if raw_action is None:
            raw_action = np.zeros(self.n_assets, dtype=float)
        for idx, ticker in enumerate(self.tickers):
            row[f"{ticker}_price"] = float(prices[idx])
            row[f"{ticker}_shares"] = int(self.holdings[idx])
            row[f"{ticker}_value"] = float(values[idx])
            row[f"{ticker}_weight"] = float(values[idx] / total_asset) if total_asset > 0 else 0.0
            row[f"{ticker}_raw_action"] = float(raw_action[idx])
        self._episode_snapshots.append(row)

    def _build_observation(self) -> np.ndarray:
        prices = self._get_prices(self.current_step)
        total_asset = max(self._current_total_asset(prices), 1e-8)
        indicator_values = [
            self._normalize_indicator_values(column, self.indicator_matrices[column][self.current_step], prices)
            for column in self.technical_indicator_columns
        ]
        features = np.concatenate(
            [
                prices / self.close_scale,
                *indicator_values,
                self.holdings.astype(float) / max(self.hmax, 1),
                np.array([self.cash / total_asset], dtype=float),
            ]
        )
        return features.astype(np.float32)

    def _append_transaction(
        self,
        step: int,
        date_idx: int,
        ticker_idx: int,
        action_type: str,
        raw_action: float,
        shares_traded: int,
        price: float,
        transaction_cost: float,
        cash_before: float,
        cash_after: float,
        holdings_before: int,
        holdings_after: int,
    ) -> None:
        gross_value = float(price * shares_traded)
        total_asset = self._current_total_asset(self._get_prices(date_idx))
        position_value = float(self._get_prices(date_idx)[ticker_idx] * self.holdings[ticker_idx])
        self._episode_transactions.append(
            {
                "step": int(step),
                "date": str(pd.Timestamp(self.dates[date_idx]).date()),
                "ticker": self.tickers[ticker_idx],
                "ticker_idx": int(ticker_idx),
                "action_type": action_type,
                "raw_action": float(raw_action),
                "shares_traded": int(shares_traded),
                "price": float(price),
                "gross_value": gross_value,
                "transaction_cost": float(transaction_cost),
                "net_value": float(
                    gross_value + transaction_cost
                    if action_type == "BUY"
                    else gross_value - transaction_cost
                    if action_type == "SELL"
                    else 0.0
                ),
                "cash_before": float(cash_before),
                "cash_after": float(cash_after),
                "holdings_before": int(holdings_before),
                "holdings_after": int(holdings_after),
                "total_allocated": float(total_asset - self.cash),
                "total_unallocated": float(self.cash),
                "total_asset": float(total_asset),
                "portfolio_weight": float(position_value / total_asset) if total_asset > 0 else 0.0,
            }
        )

    def _sell_stock(self, ticker_idx: int, shares_requested: int, prices: np.ndarray, raw_action: float, forced: bool = False) -> int:
        price = float(prices[ticker_idx])
        holdings_before = int(self.holdings[ticker_idx])
        cash_before = float(self.cash)
        executed = int(min(max(shares_requested, 0), max(holdings_before, 0))) if not forced else int(max(holdings_before, 0))
        if executed > 0 and price > 0:
            gross_value = price * executed
            tx_cost = gross_value * self.sell_cost_pct
            self.cash += gross_value - tx_cost
            self.holdings[ticker_idx] -= executed
            action_type = "SELL"
        else:
            tx_cost = 0.0
            action_type = "HOLD"
            executed = 0
        self._append_transaction(
            step=len(self._episode_snapshots),
            date_idx=self.current_step,
            ticker_idx=ticker_idx,
            action_type=action_type,
            raw_action=raw_action,
            shares_traded=executed,
            price=price,
            transaction_cost=tx_cost,
            cash_before=cash_before,
            cash_after=float(self.cash),
            holdings_before=holdings_before,
            holdings_after=int(self.holdings[ticker_idx]),
        )
        return executed

    def _buy_stock(self, ticker_idx: int, shares_requested: int, prices: np.ndarray, raw_action: float) -> int:
        price = float(prices[ticker_idx])
        holdings_before = int(self.holdings[ticker_idx])
        cash_before = float(self.cash)
        if price <= 0:
            affordable = 0
        else:
            affordable = int(self.cash // (price * (1.0 + self.buy_cost_pct)))
        executed = int(min(max(shares_requested, 0), max(affordable, 0)))
        if executed > 0:
            gross_value = price * executed
            tx_cost = gross_value * self.buy_cost_pct
            self.cash -= gross_value + tx_cost
            self.holdings[ticker_idx] += executed
            action_type = "BUY"
        else:
            tx_cost = 0.0
            action_type = "HOLD"
            executed = 0
        self._append_transaction(
            step=len(self._episode_snapshots),
            date_idx=self.current_step,
            ticker_idx=ticker_idx,
            action_type=action_type,
            raw_action=raw_action,
            shares_traded=executed,
            price=price,
            transaction_cost=tx_cost,
            cash_before=cash_before,
            cash_after=float(self.cash),
            holdings_before=holdings_before,
            holdings_after=int(self.holdings[ticker_idx]),
        )
        return executed

    def get_episode_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._episode_snapshots).copy()

    def get_episode_snapshots(self) -> pd.DataFrame:
        return pd.DataFrame(self._episode_snapshots).copy()

    def get_episode_transactions(self) -> pd.DataFrame:
        return pd.DataFrame(self._episode_transactions).copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_assets, dtype=int)
        self.portfolio_value = self.initial_capital
        self.portfolio_return = 0.0
        self._episode_transactions = []
        self._episode_snapshots = []
        self._record_snapshot(step=0, date_idx=0, raw_action=None)
        return self._build_observation(), {}

    def step(self, action):
        raw_action = np.nan_to_num(np.asarray(action, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        prices = self._get_prices(self.current_step)
        previous_value = self._current_total_asset(prices)

        if self.Turbulence[self.current_step] > self.turbulence_threshold:
            sell_orders = [(idx, self.holdings[idx], 0.0, True) for idx in range(self.n_assets)]
            buy_orders: list[tuple[int, int, float]] = []
        else:
            scaled_actions = np.trunc(raw_action * self.hmax).astype(int)
            sell_orders = [(idx, abs(int(scaled_actions[idx])), float(raw_action[idx]), False) for idx in np.argsort(scaled_actions) if scaled_actions[idx] < 0]
            buy_orders = [(idx, int(scaled_actions[idx]), float(raw_action[idx])) for idx in np.argsort(scaled_actions)[::-1] if scaled_actions[idx] > 0]

        if self.Turbulence[self.current_step] > self.turbulence_threshold:
            for idx, shares_requested, raw_value, forced in sell_orders:
                self._sell_stock(idx, shares_requested, prices, raw_value, forced=forced)
        else:
            for idx, shares_requested, raw_value, forced in sell_orders:
                self._sell_stock(idx, shares_requested, prices, raw_value, forced=forced)
            traded_buys = set()
            for idx, shares_requested, raw_value in buy_orders:
                traded_buys.add(idx)
                self._buy_stock(idx, shares_requested, prices, raw_value)
            for idx in range(self.n_assets):
                if idx not in traded_buys and idx not in {order[0] for order in sell_orders}:
                    self._append_transaction(
                        step=len(self._episode_snapshots),
                        date_idx=self.current_step,
                        ticker_idx=idx,
                        action_type="HOLD",
                        raw_action=float(raw_action[idx]),
                        shares_traded=0,
                        price=float(prices[idx]),
                        transaction_cost=0.0,
                        cash_before=float(self.cash),
                        cash_after=float(self.cash),
                        holdings_before=int(self.holdings[idx]),
                        holdings_after=int(self.holdings[idx]),
                    )

        next_step = min(self.current_step + 1, len(self.dates) - 1)
        next_prices = self._get_prices(next_step)
        self.current_step = next_step
        self.portfolio_value = self._current_total_asset(next_prices)
        
        step_reward = self.portfolio_value - previous_value
        self.portfolio_return = step_reward / (previous_value + 1e-8)
        
        # Approximate risk from recent realized returns.
        start_idx = max(0, self.current_step - 10)
        price_window = self.C[start_idx : self.current_step + 1]
        ret_window = np.empty((0, self.n_assets), dtype=float)
        if price_window.shape[0] > 1:
            ret_window = np.diff(price_window, axis=0) / (price_window[:-1] + 1e-8)
            ret_window = np.nan_to_num(ret_window)
        
        risk = 0.0
        if ret_window.shape[0] > 1:
            cov_matrix = np.cov(ret_window.T)
            weights = np.zeros(self.n_assets + 1)
            weights[0] = self.cash / (self.portfolio_value + 1e-8)
            for i in range(self.n_assets):
                weights[i+1] = (self.holdings[i] * next_prices[i]) / (self.portfolio_value + 1e-8)
            risk = np.dot(weights[1:].T, np.dot(cov_matrix, weights[1:]))
            if np.isnan(risk) or np.isinf(risk):
                risk = 0.0
                
        # Clipped Reward
        reward = self.portfolio_return - risk
        reward = np.clip(reward, -10.0, 10.0)
        
        self._record_snapshot(step=len(self._episode_snapshots), date_idx=self.current_step, raw_action=raw_action)

        done = self.current_step >= len(self.dates) - 1
        return self._build_observation(), float(reward), done, False, {}
