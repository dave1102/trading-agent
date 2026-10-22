import pandas as pd

from trading_agent.market_data.storage import data_path, load_ohlcv, save_ohlcv


def candles(start, n):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0}, index=idx
    )


def test_data_path_is_filesystem_safe(tmp_path):
    p = data_path("crypto", "btc/usdt", "1h", base_dir=tmp_path)
    assert p == tmp_path / "crypto" / "BTC-USDT_1h.csv"


def test_round_trip_and_merge(tmp_path):
    path = tmp_path / "crypto" / "BTC-USDT_1h.csv"
    save_ohlcv(candles("2024-01-01 00:00", 5), path)
    merged = save_ohlcv(candles("2024-01-01 03:00", 5), path)  # overlaps by 2 candles

    assert len(merged) == 8
    loaded = load_ohlcv(path)
    assert list(loaded.index) == list(merged.index)
    assert (loaded.values == merged.values).all()
