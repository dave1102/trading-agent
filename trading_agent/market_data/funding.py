"""Download perpetual-futures funding rates (public data, no API keys).

In perpetual futures, long and short traders pay each other a small "funding
rate" every few hours. A high positive rate means many traders are paying to
stay long, i.e. the market is crowded with buyers. We use it as market-wide
sentiment for every coin.

Files are stored as data/raw/funding/<EXCHANGE>_<SYMBOL>.csv with columns
timestamp, funding_rate.
"""

import time
from pathlib import Path

import pandas as pd

from .crypto import to_ms


def fetch_funding_history(
    symbol: str,
    exchange,
    since,
    now_ms: int | None = None,
    limit: int = 1000,
    max_requests: int = 10_000,
) -> pd.Series:
    """Page through funding history from `since` up to now."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    cursor = to_ms(since)
    rows = []
    for _ in range(max_requests):
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1]["timestamp"] + 1
        if next_cursor <= cursor or next_cursor >= now_ms:
            break
        cursor = next_cursor
    return rows_to_series(rows)


def rows_to_series(rows: list) -> pd.Series:
    times = pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True)
    series = pd.Series([float(r["fundingRate"]) for r in rows], index=times, name="funding_rate")
    series.index.name = "timestamp"
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.dropna()


def funding_path(exchange_id: str, symbol: str, base_dir: Path) -> Path:
    """e.g. ("deribit", "BTC/USD:BTC") -> funding/DERIBIT_BTC-USD-BTC.csv"""
    safe = symbol.replace("/", "-").replace(":", "-").upper()
    return Path(base_dir) / "funding" / f"{exchange_id.upper()}_{safe}.csv"


def load_funding(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df["funding_rate"].astype(float)


def save_funding(series: pd.Series, path: Path) -> pd.Series:
    """Write to `path`, merging with any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        series = pd.concat([load_funding(path), series])
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.rename("funding_rate").to_frame().to_csv(path)
    return series
