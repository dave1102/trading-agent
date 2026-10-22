"""Import historical forex candles from CSV files.

Forex has no single public exchange, so we import files downloaded from a
legitimate provider instead of scraping. Two formats are supported:

- HistData.com "Generic ASCII" M1 files (free, no account needed)
- Any CSV with timestamp, open, high, low, close[, volume] columns
  (e.g. exports from Dukascopy or your broker's demo platform)
"""

from pathlib import Path

import pandas as pd

from .validate import OHLCV_COLUMNS, clean_ohlcv

# HistData timestamps are in EST *without* daylight saving, i.e. always UTC-5.
HISTDATA_UTC_OFFSET = pd.Timedelta(hours=5)


def load_histdata_csv(path: Path) -> pd.DataFrame:
    """Read a HistData file with lines like `20240102 170000;1.10427;1.10430;1.10420;1.10425;0`."""
    df = pd.read_csv(
        path, sep=";", header=None, names=["timestamp", *OHLCV_COLUMNS], dtype={"timestamp": str}
    )
    local_time = pd.to_datetime(df.pop("timestamp"), format="%Y%m%d %H%M%S")
    df.index = (local_time + HISTDATA_UTC_OFFSET).dt.tz_localize("UTC")
    return clean_ohlcv(df)


def load_generic_csv(path: Path, timestamp_col: str = "timestamp", tz: str = "UTC") -> pd.DataFrame:
    """Read a CSV with a header row. `tz` is the timezone of timestamps that lack one."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = pd.to_datetime(df.pop(timestamp_col.lower()))
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    df.index = ts.dt.tz_convert("UTC")
    if "volume" not in df.columns:
        df["volume"] = 0.0  # many forex sources have no real volume
    return clean_ohlcv(df)
