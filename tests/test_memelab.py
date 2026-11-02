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
        "liquidity_usd": [1_000, 50_000, 50_000],
    })
    t = int(T0.floor("min").timestamp())
    client = FakeClient([[t + 60, 1, 2, 1, 2, 5]])
    now = T0 + pd.Timedelta(hours=5)  # C's 4-hour window has not ended yet
    candles, fetched = tmp_path / "c.csv", tmp_path / "f.csv"
    assert memelab.fetch(client, entries, now, candles, fetched, random.Random(1)) == 2
    assert memelab.fetch(client, entries, now, candles, fetched, random.Random(1)) == 0  # resumes
    assert sorted(pd.read_csv(fetched)["pool_address"]) == ["A", "B"]
    assert len(pd.read_csv(candles)) == 2
    assert list(pd.read_csv(fetched)["pool_address"]) == ["B", "A"]  # the bigger pool first


def test_load_downloads_merges_machines_without_duplicates(tmp_path):
    row = {"network": "solana", "seen_at": T0.isoformat()}
    pd.DataFrame([{**row, "pool_address": "A", "candles": 1}]).to_csv(tmp_path / "lab_fetched.csv", index=False)
    pd.DataFrame([{**row, "pool_address": p, "candles": 1} for p in "AB"]).to_csv(
        tmp_path / "lab_fetched_pc.csv", index=False)
    for name, pools in [("lab_candles.csv", "A"), ("lab_candles_pc.csv", "AB")]:
        pd.DataFrame([{**row, "pool_address": p, "ts": T0.isoformat(), "open": 1, "high": 1, "low": 1,
                       "close": 1, "volume": 1} for p in pools]).to_csv(tmp_path / name, index=False)
    fetched, candles = memelab.load_downloads(tmp_path)
    assert sorted(fetched["pool_address"]) == ["A", "B"]
    assert sorted(candles["pool_address"]) == ["A", "B"]


COSTS = {"trade_usd": 100, "dex_fee": 0.0, "gas_usd": {"solana": 0.0}}
ENTRY = pd.Series({"price_usd": 1.0, "liquidity_usd": 1e12, "network": "solana", "sell_tax": ""})


def bars(rows):
    """rows of (minute, open, high, low, close)."""
    return pd.DataFrame(rows, columns=["minute", "open", "high", "low", "close"])


def test_replay_target_stop_and_time():
    up = bars([(1, 1.0, 1.05, 0.98, 1.04), (2, 1.04, 1.35, 1.0, 1.3)])
    r = memelab.replay(ENTRY, up, None, 0.3, 0.2, 60, COSTS)
    assert r["reason"] == "target" and abs(r["gross"] - 0.3) < 1e-9
    down = bars([(1, 1.0, 1.0, 0.75, 0.8)])
    r = memelab.replay(ENTRY, down, None, 0.3, 0.2, 60, COSTS)
    assert r["reason"] == "stop" and abs(r["gross"] + 0.2) < 1e-9
    r = memelab.replay(ENTRY, up, None, None, None, 60, COSTS)
    assert r["reason"] == "time" and abs(r["gross"] - 0.3) < 1e-9
    r = memelab.replay(ENTRY, up, None, None, None, 2, COSTS)  # only minute 1 is before the 2-minute exit
    assert abs(r["gross"] - 0.04) < 1e-9


def test_replay_is_pessimistic():
    both = bars([(1, 1.0, 1.5, 0.7, 1.0)])  # target and stop in the same minute
    assert memelab.replay(ENTRY, both, None, 0.3, 0.2, 60, COSTS)["reason"] == "stop"
    gap = bars([(1, 0.5, 0.5, 0.4, 0.45)])  # opened far below the stop: sold at the open
    assert abs(memelab.replay(ENTRY, gap, None, 0.3, 0.2, 60, COSTS)["gross"] + 0.5) < 1e-9


def test_replay_rug_and_quiet_pool():
    quiet = bars([])
    r = memelab.replay(ENTRY, quiet, None, 0.3, 0.2, 60, COSTS)
    assert r["reason"] == "time" and r["gross"] == 0
    assert memelab.replay(ENTRY, quiet, 15, 0.3, 0.2, 60, COSTS)["net"] == -1.0  # rugged before the exit
    up = bars([(1, 1.0, 1.35, 1.0, 1.3)])
    assert memelab.replay(ENTRY, up, 15, 0.3, 0.2, 60, COSTS)["reason"] == "target"  # sold before the rug
    busy = bars([(m, 1.0, 1.0, 1.0, 1.0) for m in range(1, 50)])
    assert memelab.replay(ENTRY, busy, 15, None, None, 30, COSTS)["reason"] == "time"  # still trading at 30m


def test_rug_times_uses_last_good_check():
    checks = pd.DataFrame({
        "network": ["solana"] * 3, "pool_address": ["A"] * 3,
        "checked_at": ["2026-10-30T10:05:00+00:00", "2026-10-30T10:15:00+00:00", "2026-10-30T10:30:00+00:00"],
        "rugged": [False, False, True],
    })
    assert memelab.rug_times(checks)[("solana", "A")] == pd.Timestamp("2026-10-30T10:15:00+00:00")
