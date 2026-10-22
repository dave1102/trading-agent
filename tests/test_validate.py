import pandas as pd
import pytest

from trading_agent.market_data.validate import (
    clean_ohlcv,
    find_gaps,
    resample_ohlcv,
    timeframe_to_seconds,
)


def make_candles(times, closes=None):
    closes = closes or [100.0] * len(times)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [5.0] * len(times),
        },
        index=pd.to_datetime(times, utc=True),
    )


def test_timeframe_to_seconds():
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("4h") == 4 * 3600
    assert timeframe_to_seconds("1d") == 86400
    with pytest.raises(ValueError):
        timeframe_to_seconds("1M")


def test_clean_sorts_dedupes_and_drops_bad_rows():
    df = make_candles(["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00"])
    df.iloc[3, df.columns.get_loc("close")] = 100.5  # later duplicate wins
    bad = make_candles(["2024-01-01 03:00"])
    bad["high"] = 50.0  # high below close is impossible
    out = clean_ohlcv(pd.concat([df, bad]))

    assert list(out.index.hour) == [0, 1, 2]
    assert out.loc["2024-01-01 01:00", "close"].item() == 100.5
    assert str(out.index.tz) == "UTC"


def test_find_gaps():
    df = make_candles(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 04:00"])
    gaps = find_gaps(df, "1h")
    assert len(gaps) == 1
    assert gaps.loc[0, "missing_bars"] == 2
    assert gaps.loc[0, "gap_start"] == pd.Timestamp("2024-01-01 02:00", tz="UTC")


def test_resample_to_hourly():
    times = pd.date_range("2024-01-01 00:00", periods=120, freq="1min", tz="UTC")
    df = make_candles(list(times), closes=[float(i + 10) for i in range(120)])
    hourly = resample_ohlcv(df, "1h")

    assert len(hourly) == 2
    first = hourly.iloc[0]
    assert first["open"] == 10 and first["close"] == 69
    assert first["high"] == 70 and first["low"] == 9
    assert first["volume"] == 5.0 * 60
