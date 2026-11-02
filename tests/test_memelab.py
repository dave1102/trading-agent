import random

import pandas as pd

from trading_agent import memelab
from trading_agent.memecoins.candles import fetch_candles, parse_candles

T0 = pd.Timestamp("2026-10-30 10:00:30", tz="UTC")


def ohlcv_payload(rows):
    return {"data": {"attributes": {"ohlcv_list": rows}}}


def test_parse_candles_sorts_oldest_first():
    t = int(T0.floor("min").timestamp())
    df = parse_candles(ohlcv_payload([[t + 60, 2, 3, 1, 2, 10], [t, 1, 1, 1, 1, 5]]))
    assert list(df["open"]) == [1, 2]
    assert df["ts"].iloc[0] == T0.floor("min")
    assert parse_candles(None).empty


class FakeClient:
    min_interval = 0

    def __init__(self, rows):
        self.rows, self.urls = rows, []

    def get_json(self, url):
        self.urls.append(url)
        return ohlcv_payload(self.rows)


def test_fetch_candles_keeps_only_the_window():
    t = int(T0.floor("min").timestamp())
    client = FakeClient([[t - 600, 1, 1, 1, 1, 1], [t, 1, 1, 1, 1, 1], [t + 60, 1, 1, 1, 1, 1]])
    df = fetch_candles(client, "solana", "POOL", T0, T0 + pd.Timedelta(minutes=30))
    assert len(df) == 2
    assert "before_timestamp=" in client.urls[0] and "token=base" in client.urls[0]


def test_fetch_skips_done_pools_and_unfinished_windows(tmp_path):
    entries = pd.DataFrame({
        "network": ["solana", "solana", "base"],
        "pool_address": ["A", "B", "C"],
        "seen_at": [T0.isoformat(), T0.isoformat(), (T0 + pd.Timedelta(hours=10)).isoformat()],
    })
    t = int(T0.floor("min").timestamp())
    client = FakeClient([[t + 60, 1, 2, 1, 2, 5]])
    now = T0 + pd.Timedelta(hours=5)  # C's 4-hour window has not ended yet
    candles, fetched = tmp_path / "c.csv", tmp_path / "f.csv"
    assert memelab.fetch(client, entries, now, candles, fetched, random.Random(1)) == 2
    assert memelab.fetch(client, entries, now, candles, fetched, random.Random(1)) == 0  # resumes
    assert sorted(pd.read_csv(fetched)["pool_address"]) == ["A", "B"]
    assert len(pd.read_csv(candles)) == 2
