"""1-minute price candles of a pool from GeckoTerminal (public API, no key).

A candle exists only for minutes with at least one trade, so quiet pools have gaps.
"""

import pandas as pd

from .sources import GT_BASE

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def parse_candles(payload: dict | None) -> pd.DataFrame:
    """GeckoTerminal ohlcv response -> candles sorted oldest first, `ts` = minute start (UTC)."""
    rows = ((payload or {}).get("data") or {}).get("attributes", {}).get("ohlcv_list") or []
    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS) if rows else pd.DataFrame(columns=CANDLE_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def fetch_candles(client, network: str, pool_address: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Candles from `start` to `end` (at most ~1000 minutes per request)."""
    minutes = int((end - start) / pd.Timedelta(minutes=1)) + 5
    url = (f"{GT_BASE}/networks/{network}/pools/{pool_address}/ohlcv/minute?aggregate=1"
           f"&before_timestamp={int(end.timestamp())}&limit={min(minutes, 1000)}&currency=usd&token=base")
    candles = parse_candles(client.get_json(url))
    return candles[(candles["ts"] >= start.floor("min")) & (candles["ts"] <= end)].reset_index(drop=True)
