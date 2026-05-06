import os

import pandas as pd

from data.fetchers.yfinance_fetcher import _validate_downloaded_frame, download_sp_data


def _resolve_column(df: pd.DataFrame, configured: str | None, candidates: list[str]) -> str:
    if configured and configured in df.columns:
        return configured
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not resolve required column. Tried: {candidates}")


def load_synthetic_csv_data(
    csv_path: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    allow_missing_tickers: bool = False,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Synthetic CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    column_map = column_map or {}

    date_col = _resolve_column(df, column_map.get("date"), ["date", "Date"])
    ticker_col = _resolve_column(df, column_map.get("ticker"), ["tic", "Ticker", "ticker"])
    open_col = _resolve_column(df, column_map.get("open"), ["open", "Open"])
    high_col = _resolve_column(df, column_map.get("high"), ["high", "High"])
    low_col = _resolve_column(df, column_map.get("low"), ["low", "Low"])
    close_col = _resolve_column(df, column_map.get("close"), ["close", "Close"])
    volume_col = _resolve_column(df, column_map.get("volume"), ["volume", "Volume"])

    normalized = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "Ticker": df[ticker_col].astype(str),
            "Open": pd.to_numeric(df[open_col], errors="coerce"),
            "High": pd.to_numeric(df[high_col], errors="coerce"),
            "Low": pd.to_numeric(df[low_col], errors="coerce"),
            "Close": pd.to_numeric(df[close_col], errors="coerce"),
            "Volume": pd.to_numeric(df[volume_col], errors="coerce"),
        }
    )

    normalized = normalized.dropna(subset=["Date"]).set_index("Date").sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    normalized = normalized[(normalized.index >= start_ts) & (normalized.index <= end_ts)]
    normalized = normalized[normalized["Ticker"].isin(tickers)]

    return _validate_downloaded_frame(
        normalized,
        tickers,
        f"synthetic CSV {csv_path}",
        allow_missing_tickers=allow_missing_tickers,
    )


def load_experiment_price_data(
    config: dict,
    tickers: list[str],
    base_dir: str,
    default_cache_filename: str = "data_cache.csv",
    allow_missing_tickers: bool = False,
) -> pd.DataFrame:
    data_source = str(config.get("data_source", "yahoo")).strip().lower()

    if data_source in {"yahoo", "yfinance"}:
        cache_filename = str(config.get("data_cache_file", default_cache_filename)).strip()
        cache_path = cache_filename if os.path.isabs(cache_filename) else os.path.join(base_dir, cache_filename)
        return download_sp_data(
            tickers,
            config["start_date"],
            config["test_end"],
            cache_path=cache_path,
            allow_missing_tickers=allow_missing_tickers,
        )

    if data_source in {"synthetic_csv", "synthetic", "csv"}:
        synthetic_path = config.get("synthetic_data_path")
        if not synthetic_path:
            raise ValueError("data_source='synthetic_csv' requires 'synthetic_data_path' in config.")
        csv_path = synthetic_path if os.path.isabs(synthetic_path) else os.path.join(base_dir, synthetic_path)
        column_map = config.get("synthetic_column_map")
        return load_synthetic_csv_data(
            csv_path=csv_path,
            tickers=tickers,
            start_date=config["start_date"],
            end_date=config["test_end"],
            allow_missing_tickers=allow_missing_tickers,
            column_map=column_map,
        )

    raise ValueError(
        f"Unsupported data_source '{data_source}'. Supported values: 'yahoo', 'synthetic_csv'."
    )