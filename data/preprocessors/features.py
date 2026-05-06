import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    import quantstats as qs
except ImportError:
    qs = None


REQUESTED_TECHNICAL_INDICATOR_COLUMNS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_5_sma",
    "close_10_sma",
    "close_20_sma",
    "close_30_sma",
    "close_60_sma",
]

ADDITIONAL_TECHNICAL_INDICATOR_COLUMNS = [
    "atr_14",
    "stoch_k",
    "stoch_d",
    "return_1d",
    "volatility_20",
    "qs_sharpe_30",
    "qs_sortino_30",
    "fr_20",
    "obv",
]

TECHNICAL_INDICATOR_COLUMNS = REQUESTED_TECHNICAL_INDICATOR_COLUMNS + ADDITIONAL_TECHNICAL_INDICATOR_COLUMNS

ENV_TECHNICAL_INDICATOR_COLUMNS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_5_sma",
    "close_10_sma",
    "close_20_sma",
    "close_30_sma",
    "close_60_sma",
    "atr_14",
    "stoch_k",
    "stoch_d",
    "return_1d",
    "volatility_20",
    "qs_sharpe_30",
    "qs_sortino_30",
    "fr_20",
    "us_rate",
]

FEATURE_COLUMN_FALLBACKS = {
    "macd": ["macd", "MACD"],
    "rsi_30": ["rsi_30", "RSI"],
    "stoch_k": ["stoch_k", "SO"],
    "fr_20": ["fr_20", "FR"],
    "us_rate": ["us_rate", "US_RATE"],
}

TECHNICAL_INDICATOR_ALIASES = {
    candidate: canonical
    for canonical, candidates in FEATURE_COLUMN_FALLBACKS.items()
    for candidate in candidates
}
for column in ENV_TECHNICAL_INDICATOR_COLUMNS:
    TECHNICAL_INDICATOR_ALIASES.setdefault(column, column)


def _default_series(index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    return pd.Series(fill_value, index=index, dtype=float)


def _as_series(value, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    if value is None:
        return _default_series(index, fill_value=fill_value)
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return _default_series(index, fill_value=fill_value)
        value = value.iloc[:, 0]
    series = pd.Series(value, index=index, dtype=float)
    return series.replace([np.inf, -np.inf], np.nan)


def _extract_frame_column(frame, index: pd.Index, prefix: str, fallback_idx: int, fill_value: float = 0.0) -> pd.Series:
    if frame is None or getattr(frame, "empty", True):
        return _default_series(index, fill_value=fill_value)
    for column in frame.columns:
        if str(column).upper().startswith(prefix.upper()):
            return _as_series(frame[column], index=index, fill_value=fill_value)
    if len(frame.columns) <= fallback_idx:
        return _default_series(index, fill_value=fill_value)
    return _as_series(frame.iloc[:, fallback_idx], index=index, fill_value=fill_value)


def _rolling_quantstats_stat(returns: pd.Series, window: int, stat_name: str) -> pd.Series:
    if qs is None:
        return _default_series(returns.index, fill_value=0.0)

    stat_fn = getattr(qs.stats, stat_name)
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _apply(window_values) -> float:
        window_series = pd.Series(window_values).replace([np.inf, -np.inf], np.nan).dropna()
        if len(window_series) < window:
            return 0.0
        if np.isclose(window_series.std(ddof=0), 0.0):
            return 0.0
        try:
            value = float(stat_fn(window_series, periods=252))
        except TypeError:
            value = float(stat_fn(window_series))
        if not np.isfinite(value):
            return 0.0
        return value

    return clean.rolling(window=window, min_periods=window).apply(_apply, raw=False).fillna(0.0)


def _build_dx_series(high: pd.Series, low: pd.Series, close: pd.Series, index: pd.Index, length: int = 30) -> pd.Series:
    adx = ta.adx(high, low, close, length=length)
    if adx is None or getattr(adx, "empty", True):
        return _default_series(index, fill_value=0.0)

    dmp = _extract_frame_column(adx, index, prefix="DMP", fallback_idx=1, fill_value=0.0)
    dmn = _extract_frame_column(adx, index, prefix="DMN", fallback_idx=2, fill_value=0.0)
    denominator = (dmp + dmn).replace(0.0, np.nan)
    dx = ((dmp - dmn).abs() / denominator) * 100.0
    return dx.fillna(0.0)


def resolve_feature_column_name(df: pd.DataFrame, column: str) -> str | None:
    for candidate in FEATURE_COLUMN_FALLBACKS.get(column, [column]):
        if candidate in df.columns:
            return candidate
    return None


def normalize_technical_indicator_selection(selected_columns: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if selected_columns is None:
        return list(ENV_TECHNICAL_INDICATOR_COLUMNS)

    if isinstance(selected_columns, str):
        if selected_columns.strip().lower() == "all":
            return list(ENV_TECHNICAL_INDICATOR_COLUMNS)
        selected_columns = [part.strip() for part in selected_columns.split(",") if part.strip()]

    normalized: list[str] = []
    unknown: list[str] = []
    for raw_name in selected_columns:
        key = str(raw_name).strip()
        canonical = TECHNICAL_INDICATOR_ALIASES.get(key)
        if canonical is None:
            canonical = TECHNICAL_INDICATOR_ALIASES.get(key.lower())
        if canonical is None:
            unknown.append(key)
            continue
        if canonical not in normalized:
            normalized.append(canonical)

    if unknown:
        raise ValueError(
            "Unknown technical_indicator_columns: "
            + ", ".join(unknown)
            + ". Supported values: "
            + ", ".join(ENV_TECHNICAL_INDICATOR_COLUMNS)
        )

    if not normalized:
        raise ValueError("technical_indicator_columns must contain at least one valid indicator")

    return normalized


def calculate_technical_features(df: pd.DataFrame, tickers: list):
    """Calculate canonical technical indicators and preserve legacy aliases."""
    print("Engineering features...")
    features = []
    for _, group in df.groupby("Ticker"):
        g = group.sort_index().copy()

        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)
        volume = g["Volume"].astype(float)

        macd = ta.macd(close, fast=12, slow=26, signal=9)
        bbands = ta.bbands(close, length=20, std=2.0)
        stoch = ta.stoch(high, low, close)

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rolling_max = high.rolling(20, min_periods=1).max()
        rolling_min = low.rolling(20, min_periods=1).min()

        g["macd"] = _extract_frame_column(macd, g.index, prefix="MACD", fallback_idx=0)
        g["boll_ub"] = _extract_frame_column(bbands, g.index, prefix="BBU", fallback_idx=2)
        g["boll_lb"] = _extract_frame_column(bbands, g.index, prefix="BBL", fallback_idx=0)
        g["rsi_30"] = _as_series(ta.rsi(close, length=30), g.index, fill_value=50.0)
        g["cci_30"] = _as_series(ta.cci(high, low, close, length=30), g.index, fill_value=0.0)
        g["dx_30"] = _build_dx_series(high, low, close, g.index, length=30)
        g["close_5_sma"] = _as_series(ta.sma(close, length=5), g.index, fill_value=close.iloc[0])
        g["close_10_sma"] = _as_series(ta.sma(close, length=10), g.index, fill_value=close.iloc[0])
        g["close_20_sma"] = _as_series(ta.sma(close, length=20), g.index, fill_value=close.iloc[0])
        g["close_30_sma"] = _as_series(ta.sma(close, length=30), g.index, fill_value=close.iloc[0])
        g["close_60_sma"] = _as_series(ta.sma(close, length=60), g.index, fill_value=close.iloc[0])
        g["atr_14"] = _as_series(ta.atr(high, low, close, length=14), g.index, fill_value=0.0)
        g["stoch_k"] = _extract_frame_column(stoch, g.index, prefix="STOCHK", fallback_idx=0, fill_value=50.0)
        g["stoch_d"] = _extract_frame_column(stoch, g.index, prefix="STOCHD", fallback_idx=1, fill_value=50.0)
        g["return_1d"] = returns
        g["volatility_20"] = returns.rolling(20, min_periods=1).std().fillna(0.0)
        g["qs_sharpe_30"] = _rolling_quantstats_stat(returns, window=30, stat_name="sharpe")
        g["qs_sortino_30"] = _rolling_quantstats_stat(returns, window=30, stat_name="sortino")
        g["fr_20"] = (close - rolling_min) / (rolling_max - rolling_min + 1e-8)
        g["obv"] = _as_series(ta.obv(close, volume), g.index, fill_value=0.0)

        for column in TECHNICAL_INDICATOR_COLUMNS:
            g[column] = pd.to_numeric(g[column], errors="coerce")

        g["MACD"] = g["macd"]
        g["RSI"] = g["rsi_30"]
        g["SO"] = g["stoch_k"]
        g["FR"] = g["fr_20"]
        if "us_rate" not in g.columns:
            g["us_rate"] = 0.0

        g = g.ffill().fillna(0.0)
        features.append(g)

    df_full = pd.concat(features)
    date_counts = df_full.groupby(level=0).size()
    valid_dates = date_counts[date_counts == len(tickers)].index
    df_full = df_full.loc[valid_dates]
    df_full = df_full.sort_index()

    df_full = calculate_turbulence(df_full)
    return df_full

def calculate_turbulence(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    print("Calculating Turbulence Index...")
    df_pivot = df.pivot(columns='Ticker', values='Close')
    returns = df_pivot.pct_change().dropna()
    
    turbulence = pd.Series(index=returns.index, dtype=np.float64)
    turbulence.iloc[:window] = 0.0
    
    for i in range(window, len(returns)):
        past_returns = returns.iloc[i-window:i]
        mu = past_returns.mean()
        current_return = returns.iloc[i]
        
        cov = past_returns.cov()
        try:
            cov_inv = np.linalg.pinv(cov.values)
        except Exception:
            cov_inv = np.eye(len(cov.columns))
            
        diff = (current_return - mu).values
        turb = float(diff.dot(cov_inv).dot(diff.T))
        turbulence.iloc[i] = turb
        
    df['Turbulence'] = df.index.map(turbulence).fillna(0.0)
    return df
