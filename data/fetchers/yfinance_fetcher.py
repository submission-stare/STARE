import pandas as pd
import yfinance as yf
import os


def _validate_downloaded_frame(
    df: pd.DataFrame,
    requested_tickers: list,
    source_label: str,
    allow_missing_tickers: bool = False,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No market data was loaded from {source_label}.")

    df = df.copy()
    required_price_columns = ["Open", "High", "Low", "Close"]
    present_price_columns = [col for col in required_price_columns if col in df.columns]
    if present_price_columns:
        df = df.dropna(subset=present_price_columns, how="all")

    if df.empty:
        raise ValueError(f"All downloaded rows from {source_label} were empty.")

    if "Ticker" in df.columns and "Close" in df.columns:
        valid_tickers = set(df.loc[df["Close"].notna(), "Ticker"].astype(str))
        missing_tickers = [ticker for ticker in requested_tickers if ticker not in valid_tickers]
        if missing_tickers:
            if allow_missing_tickers:
                print(
                    "Warning: missing usable price history for ticker(s): "
                    + ", ".join(missing_tickers)
                    + f" from {source_label}. Continuing with available tickers."
                )
                return df
            raise ValueError(
                "Missing usable price history for ticker(s): "
                + ", ".join(missing_tickers)
                + f" from {source_label}. Update the ticker list or refresh the cache with valid symbols."
            )

    return df


def download_sp_data(
    tickers: list,
    start_date: str,
    end_date: str,
    cache_path: str = "data_cache.csv",
    allow_missing_tickers: bool = False,
):
    """
    Downloads historical ticker data using yfinance. Uses a local CSV cache if available.
    """
    if os.path.exists(cache_path):
        print(f"Loading data from cache: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        df = df.set_index("Date")
        df = df[df["Ticker"].isin(tickers)]
        df = df.sort_index()
        return _validate_downloaded_frame(
            df,
            tickers,
            f"cache {cache_path}",
            allow_missing_tickers=allow_missing_tickers,
        )

    print(f"Downloading tickers via yfinance...")
    data = yf.download(tickers, start=start_date, end=end_date)
    df = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
    df = df.sort_values(['Date', 'Ticker']).set_index('Date')
    df = _validate_downloaded_frame(
        df,
        tickers,
        "yfinance",
        allow_missing_tickers=allow_missing_tickers,
    )
    df.to_csv(cache_path)
    return df
