import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from data.preprocessors.features import normalize_technical_indicator_selection, resolve_feature_column_name

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PortfolioEnv(gym.Env):
    """
    A PyTorch-compatible Gym Environment for sequential multi-period portfolio optimization.
    Features clipping and numerical stability bounds for continuous deep networks.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data: pd.DataFrame, tickers: list, window_size: int = 10, initial_capital: float = 1.0, commission: float = 0.0005, turbulence_threshold: float = 1e9, technical_indicator_columns: list[str] | tuple[str, ...] | str | None = None):
        super().__init__()
        self.rl_mode = "portfolio"
        self.data = data
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.commission = commission
        self.dates = sorted(list(self.data.index.unique()))
        self.turbulence_threshold = turbulence_threshold
        self.technical_indicator_columns = normalize_technical_indicator_selection(technical_indicator_columns)
        if 'Turbulence' in self.data.columns:
            self.Turbulence = self.data['Turbulence'].groupby(level=0).first().values
        else:
            self.Turbulence = np.zeros(len(self.dates))

        self.state_dim = window_size * (((5 + len(self.technical_indicator_columns)) * self.n_assets) + 2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        self.current_step = self.window_size
        self.portfolio_value = self.initial_capital
        self.portfolio_return = 0.0
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[0] = 1.0 
        self._episode_history = []
        
        self._build_matrices()

    def _build_matrices(self):
        self.O = self.data.pivot(columns='Ticker', values='Open').reindex(columns=self.tickers).values
        self.H = self.data.pivot(columns='Ticker', values='High').reindex(columns=self.tickers).values
        self.L = self.data.pivot(columns='Ticker', values='Low').reindex(columns=self.tickers).values
        self.C = self.data.pivot(columns='Ticker', values='Close').reindex(columns=self.tickers).values
        self.V = self.data.pivot(columns='Ticker', values='Volume').reindex(columns=self.tickers).values
        self.indicator_matrices = {
            column: self._extract_indicator_matrix(column)
            for column in self.technical_indicator_columns
        }

    def _extract_indicator_matrix(self, column: str) -> np.ndarray:
        resolved = resolve_feature_column_name(self.data, column)
        if resolved is None:
            return np.zeros((len(self.dates), self.n_assets), dtype=float)
        return (
            self.data.pivot(columns='Ticker', values=resolved)
            .reindex(index=self.dates, columns=self.tickers)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def _normalize_indicator_block(self, column: str, values: np.ndarray, close_reference: np.ndarray) -> np.ndarray:
        safe_close = np.where(close_reference < 1e-8, 1.0, close_reference)

        if column in {'rsi_30', 'dx_30', 'stoch_k', 'stoch_d'}:
            normalized = values / 100.0
        elif column == 'cci_30':
            normalized = np.clip(values / 200.0, -5.0, 5.0)
        elif column in {'macd', 'atr_14'}:
            normalized = values / safe_close
        elif column in {'boll_ub', 'boll_lb', 'close_30_sma', 'close_60_sma'}:
            normalized = (values / safe_close) - 1.0
        elif column in {'qs_sharpe_30', 'qs_sortino_30'}:
            normalized = np.clip(values / 10.0, -5.0, 5.0)
        elif column == 'return_1d':
            normalized = np.clip(values, -1.0, 1.0)
        elif column == 'volatility_20':
            normalized = np.clip(values, 0.0, 1.0)
        elif column == 'fr_20':
            normalized = np.clip(values, 0.0, 1.0)
        else:
            normalized = values

        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def _record_history_row(
        self,
        step: int,
        date_idx: int,
        account_value_before: float,
        account_value_after: float,
        weights: np.ndarray,
        prices: np.ndarray,
        raw_action: np.ndarray,
        step_transaction_cost: float,
    ) -> None:
        total_asset = float(account_value_after)
        cash_weight = float(weights[0])
        cash_value = total_asset * cash_weight
        row = {
            "step": int(step),
            "date": pd.Timestamp(self.dates[date_idx]),
            "account_value_before": float(account_value_before),
            "account_value_after": total_asset,
            "step_transaction_cost": float(step_transaction_cost),
            "cash": float(cash_value),
            "total_allocated": float(total_asset - cash_value),
            "total_asset": total_asset,
            "cash_weight": cash_weight,
            "cash_raw_action": float(raw_action[0]),
        }
        for idx, ticker in enumerate(self.tickers):
            row[f"{ticker}_price"] = float(prices[idx])
            row[f"{ticker}_weight"] = float(weights[idx + 1])
            row[f"{ticker}_raw_action"] = float(raw_action[idx + 1])
        self._episode_history.append(row)

    def get_episode_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._episode_history).copy()

    def get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step
        
        o = self.O[start:end]
        h = self.H[start:end]
        l = self.L[start:end]
        c = self.C[start:end]
        v = self.V[start:end]
        norm_factor = c[-1, :] + 1e-8
        norm_factor = np.where(norm_factor < 1e-4, 1.0, norm_factor)

        indicator_blocks = [
            self._normalize_indicator_block(column, self.indicator_matrices[column][start:end], c)
            for column in self.technical_indicator_columns
        ]

        state_features = np.concatenate([
            o / norm_factor, 
            h / norm_factor, 
            l / norm_factor, 
            c / norm_factor, 
            v / (np.max(self.V, axis=0) + 1e-8), 
            *indicator_blocks,
        ], axis=1)
        
        if np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value):
            self.portfolio_value = 1.0
        if np.isnan(self.portfolio_return) or np.isinf(self.portfolio_return):
            self.portfolio_return = 0.0

        endogenous = np.zeros((self.window_size, 2))
        endogenous[:, 0] = self.portfolio_value
        endogenous[:, 1] = self.portfolio_return
        
        full_state = np.concatenate([state_features, endogenous], axis=1)
        full_state = np.nan_to_num(full_state)
        full_state = np.clip(full_state, -100.0, 100.0)
        return full_state.flatten().astype(np.float32)

    def step(self, action):
        # Stable softmax protected from NaNs
        raw_action = np.nan_to_num(np.asarray(action, dtype=float))
        e_x = np.exp(raw_action - np.max(raw_action))
        action = e_x / e_x.sum()
        
        current_prices = self.C[self.current_step - 1]
        next_prices = self.C[self.current_step]
        
        # Avoid zero division and wipeouts for assets not yet trading (price = 0)
        safe_current = np.where(current_prices < 1e-4, 1.0, current_prices)
        safe_next = np.where(current_prices < 1e-4, 1.0, next_prices)
        
        if self.Turbulence[self.current_step - 1] > self.turbulence_threshold:
            action = np.zeros(self.n_assets + 1)
            action[0] = 1.0
            
        price_change = safe_next / safe_current
        asset_returns = np.insert(price_change, 0, 1.0)
        asset_returns = np.nan_to_num(asset_returns, nan=1.0, posinf=1.0, neginf=1.0)
        
        transaction_cost = np.sum(np.abs(action - self.weights) * self.portfolio_value * self.commission)
        
        new_portfolio_value = (self.portfolio_value - transaction_cost) * np.sum(action * asset_returns)
        step_reward = new_portfolio_value - self.portfolio_value
        
        previous_value = float(self.portfolio_value)
        self.portfolio_return = (new_portfolio_value - self.portfolio_value) / (self.portfolio_value + 1e-8)
        self.portfolio_value = new_portfolio_value
        self.weights = action
        self._record_history_row(
            step=len(self._episode_history),
            date_idx=self.current_step,
            account_value_before=previous_value,
            account_value_after=float(new_portfolio_value),
            weights=self.weights.copy(),
            prices=next_prices.copy(),
            raw_action=raw_action.copy(),
            step_transaction_cost=float(transaction_cost),
        )
        
        ret_window = self.C[self.current_step - self.window_size : self.current_step] / (self.C[self.current_step - self.window_size - 1 : self.current_step - 1] + 1e-8)
        ret_window = np.nan_to_num(ret_window)
        cov_matrix = np.cov(ret_window.T)
        
        # Clip risk and ensure positive semi-definiteness issues don't give huge NaNs
        risk = np.dot(action[1:].T, np.dot(cov_matrix, action[1:]))
        if np.isnan(risk) or np.isinf(risk):
            risk = 0.0
            
        reward = self.portfolio_return - risk
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        reward = np.clip(reward, -10.0, 10.0)
        
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        return self.get_state(), float(reward), done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size + 1
        self.portfolio_value = self.initial_capital
        self.portfolio_return = 0.0
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[0] = 1.0
        self._episode_history = []
        self._record_history_row(
            step=0,
            date_idx=self.current_step - 1,
            account_value_before=float(self.initial_capital),
            account_value_after=float(self.initial_capital),
            weights=self.weights.copy(),
            prices=self.C[self.current_step - 1].copy(),
            raw_action=np.zeros(self.n_assets + 1, dtype=float),
            step_transaction_cost=0.0,
        )
        return self.get_state(), {}

    def get_episode_snapshots(self) -> pd.DataFrame:
        return pd.DataFrame(self._episode_history).copy()

    def get_episode_transactions(self) -> pd.DataFrame:
        df = pd.DataFrame(self._episode_history)
        if df.empty:
            return pd.DataFrame(columns=["date", "ticker", "price", "action_type", "shares_traded"])

        rows = []
        for _, row in df.iterrows():
            date_val = row.get("date")
            if date_val is None:
                continue
            for ticker in self.tickers:
                rows.append(
                    {
                        "date": date_val,
                        "ticker": ticker,
                        "price": float(row.get(f"{ticker}_price", np.nan)),
                        "action_type": "HOLD",
                        "shares_traded": 0,
                    }
                )

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["date", "ticker", "price", "action_type", "shares_traded"])
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out.dropna(subset=["date", "ticker", "price"]).reset_index(drop=True)
        return out
