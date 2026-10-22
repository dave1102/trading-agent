import pandas as pd

from trading_agent.market_data.crypto import fetch_ohlcv, make_exchange, to_ms

HOUR_MS = 3_600_000
START_MS = to_ms("2024-01-01")


class FakeExchange:
    """Pretends to be a CCXT exchange holding `n` hourly candles."""

    def __init__(self, n):
        self.candles = [
            [START_MS + i * HOUR_MS, 100 + i, 101 + i, 99 + i, 100.5 + i, 10.0] for i in range(n)
        ]
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        return [c for c in self.candles if c[0] >= since][:limit]


def test_paginates_through_history():
    ex = FakeExchange(250)
    df = fetch_ohlcv("BTC/USDT", "1h", since="2024-01-01", exchange=ex, limit=100,
                     now_ms=START_MS + 250 * HOUR_MS)
    assert len(df) == 250
    assert ex.calls == 3
    assert df.index.is_unique and df.index.is_monotonic_increasing


def test_drops_candle_that_is_still_forming():
    ex = FakeExchange(250)
    df = fetch_ohlcv("BTC/USDT", "1h", since="2024-01-01", exchange=ex, limit=100,
                     now_ms=START_MS + int(249.5 * HOUR_MS))
    assert len(df) == 249
    assert df.index[-1] == pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=248)


def test_respects_until():
    ex = FakeExchange(250)
    df = fetch_ohlcv("BTC/USDT", "1h", since="2024-01-01", until="2024-01-03", exchange=ex,
                     limit=100, now_ms=START_MS + 250 * HOUR_MS)
    assert len(df) == 48


def test_exchange_client_has_no_credentials():
    ex = make_exchange("binance")
    assert not ex.apiKey and not ex.secret
