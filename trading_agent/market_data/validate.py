"""Cleaning and sanity checks shared by every data source.

Every OHLCV DataFrame in this project has the same shape:
- index: UTC timestamps named "timestamp" (the candle's OPEN time), sorted, unique
- columns: open, high, low, close, volume (floats)
"""

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert a timeframe like "15m", "1h" or "1d" into seconds."""
    amount, unit = timeframe[:-1], timeframe[-1]
    if unit not in _UNIT_SECONDS or not amount.isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe!r} (use e.g. 1m, 15m, 1h, 4h, 1d)")
    return int(amount) * _UNIT_SECONDS[unit]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, de-duplicated copy with impossible candles removed."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df[OHLCV_COLUMNS].astype(float)

    # If the same candle appears twice (e.g. after merging downloads), keep the newest copy.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=PRICE_COLUMNS)

    prices = df[PRICE_COLUMNS]
    valid = (
        (prices > 0).all(axis=1)
        & (df["high"] >= prices.max(axis=1))
        & (df["low"] <= prices.min(axis=1))
        & (df["volume"] >= 0)
    )
    return df[valid]


def find_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """List places where candles are missing.

    Crypto trades 24/7, so gaps usually mean exchange outages. Forex closes at
    weekends, so weekly gaps there are normal.
    """
    step = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
    diffs = df.index.to_series().diff()
    gaps = diffs[diffs > step]
    return pd.DataFrame(
        {
            "gap_start": gaps.index - gaps + step,
            "gap_end": gaps.index,
            "missing_bars": (gaps / step - 1).astype(int),
        },
    ).reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Combine small candles into bigger ones, e.g. 1m -> 1h."""
    rule = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])
