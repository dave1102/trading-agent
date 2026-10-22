import pandas as pd

from trading_agent.market_data.forex import load_generic_csv, load_histdata_csv


def test_histdata_converts_est_to_utc(tmp_path):
    f = tmp_path / "eurusd.csv"
    f.write_text(
        "20240102 170000;1.10427;1.10430;1.10420;1.10425;0\n"
        "20240102 170100;1.10425;1.10440;1.10425;1.10435;0\n"
    )
    df = load_histdata_csv(f)
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2024-01-02 22:00", tz="UTC")
    assert df["close"].iloc[1] == 1.10435


def test_generic_csv_without_volume(tmp_path):
    f = tmp_path / "gbpusd.csv"
    f.write_text(
        "Timestamp,Open,High,Low,Close\n"
        "2024-01-02 10:00,1.27,1.28,1.26,1.275\n"
        "2024-01-02 11:00,1.275,1.29,1.27,1.28\n"
    )
    df = load_generic_csv(f)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df["volume"] == 0).all()
    assert str(df.index.tz) == "UTC"
