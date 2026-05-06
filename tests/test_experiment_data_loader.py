import pandas as pd
import pytest

from data.fetchers.experiment_data_loader import load_experiment_price_data, load_synthetic_csv_data


def test_load_synthetic_csv_data_normalizes_columns_and_filters(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    csv_path.write_text(
        "date,tic,open,high,low,close,volume\n"
        "2019-01-01,AAA,10,11,9,10.5,1000\n"
        "2019-01-01,BBB,20,21,19,20.5,2000\n"
        "2019-01-02,AAA,10.5,11.5,10,11,1100\n"
        "2019-01-02,BBB,20.5,21.5,20,21,2100\n"
        "2019-01-03,CCC,30,31,29,30.5,3000\n",
        encoding="utf-8",
    )

    df = load_synthetic_csv_data(
        csv_path=str(csv_path),
        tickers=["AAA", "BBB"],
        start_date="2019-01-01",
        end_date="2019-01-02",
    )

    assert df.index.min() == pd.Timestamp("2019-01-01")
    assert df.index.max() == pd.Timestamp("2019-01-02")
    assert set(df["Ticker"].unique()) == {"AAA", "BBB"}
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)


def test_load_synthetic_csv_data_raises_when_ticker_missing(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    csv_path.write_text(
        "date,tic,open,high,low,close,volume\n"
        "2019-01-01,AAA,10,11,9,10.5,1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing usable price history"):
        load_synthetic_csv_data(
            csv_path=str(csv_path),
            tickers=["AAA", "BBB"],
            start_date="2019-01-01",
            end_date="2019-01-02",
        )


def test_load_experiment_price_data_uses_synthetic_source(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    csv_path.write_text(
        "date,tic,open,high,low,close,volume\n"
        "2019-01-01,AAA,10,11,9,10.5,1000\n"
        "2019-01-01,BBB,20,21,19,20.5,2000\n",
        encoding="utf-8",
    )

    config = {
        "data_source": "synthetic_csv",
        "synthetic_data_path": str(csv_path),
        "start_date": "2019-01-01",
        "test_end": "2019-01-01",
    }

    df = load_experiment_price_data(config=config, tickers=["AAA", "BBB"], base_dir=str(tmp_path))

    assert len(df) == 2
    assert set(df["Ticker"].unique()) == {"AAA", "BBB"}