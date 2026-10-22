"""Download historical crypto candles from public exchange endpoints via CCXT.

The exchange client is created WITHOUT API keys, so it can read public market
data but is physically unable to place orders or touch any account.
"""

import time

import ccxt
import pandas as pd

from .validate import OHLCV_COLUMNS, clean_ohlcv, timeframe_to_seconds


def make_exchange(exchange_id: str = "binance"):
    """Create a public, keyless CCXT client."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def to_ms(value) -> int:
    """Convert a date string / Timestamp to milliseconds since 1970 (UTC)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def rows_to_frame(rows: list) -> pd.DataFrame:
    """CCXT returns [[timestamp_ms, open, high, low, close, volume], ...]."""
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return clean_ohlcv(df.set_index("timestamp"))


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    since=None,
    until=None,
    exchange=None,
    limit: int = 1000,
    max_requests: int = 10_000,
    now_ms: int | None = None,
) -> pd.DataFrame:
    """Download candles for one symbol, paging through history `limit` candles at a time.

    Only fully closed candles are returned: the still-forming current candle
    would otherwise leak a fake "final" price into backtests.
    """
    exchange = exchange or make_exchange()
    step_ms = timeframe_to_seconds(timeframe) * 1000
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    end_ms = min(to_ms(until), now_ms) if until is not None else now_ms
    cursor = to_ms(since) if since is not None else end_ms - limit * step_ms

    rows = []
    for _ in range(max_requests):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + step_ms
        if next_cursor <= cursor or next_cursor >= end_ms:
            break
        cursor = next_cursor

    df = rows_to_frame(rows)
    step = pd.Timedelta(milliseconds=step_ms)
    closed = df.index + step <= pd.Timestamp(now_ms, unit="ms", tz="UTC")
    in_range = df.index < pd.Timestamp(end_ms, unit="ms", tz="UTC")
    return df[closed & in_range]
