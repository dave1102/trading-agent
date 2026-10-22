import pandas as pd
import pytest

from trading_agent.backtest.engine import Costs, RiskRules
from trading_agent.confidence import confidence_table, pooled_stats
from trading_agent.market_data.funding import fetch_funding_history, load_funding, save_funding
from trading_agent.market_data.synthetic import random_walk_ohlcv


class FakeFundingExchange:
    def __init__(self, n, start_ms=1_704_067_200_000, step_ms=3_600_000):
        self.rows = [{"timestamp": start_ms + i * step_ms, "fundingRate": 1e-5 * i} for i in range(n)]

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        return [r for r in self.rows if r["timestamp"] >= since][:limit]


def test_funding_download_pages_and_saves(tmp_path):
    ex = FakeFundingExchange(250)
    series = fetch_funding_history("BTC/USD:BTC", ex, "2024-01-01", now_ms=ex.rows[-1]["timestamp"] + 1, limit=100)
    assert len(series) == 250 and series.index.is_monotonic_increasing
    path = tmp_path / "funding" / "X.csv"
    save_funding(series.iloc[:150], path)
    merged = save_funding(series.iloc[100:], path)  # overlapping download
    assert len(merged) == 250
    assert load_funding(path).iloc[-1] == pytest.approx(series.iloc[-1])


def test_pooled_stats():
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 20.0, -5.0], "return": [0.01, -0.005, 0.02, -0.005]})
    stats = pooled_stats(trades)
    assert stats["trades"] == 4
    assert stats["win_rate"] == 0.5
    assert stats["profit_per_trade"] == pytest.approx(0.005)
    assert stats["profit_factor"] == pytest.approx(3.0)
    assert pooled_stats(trades.iloc[:0])["trades"] == 0


def test_confidence_table_shape_and_holdout():
    frames = {"AAA": random_walk_ohlcv(n=3000, seed=31), "BBB": random_walk_ohlcv(n=3000, seed=32)}
    leader = frames["AAA"]
    funding = {"btc": pd.Series(1e-4, index=leader.index)}
    holdout = leader.index[2000]
    table = confidence_table(frames, leader, funding, {"horizon": 1, "allow_short": False},
                             Costs(), RiskRules(), holdout, thresholds=(0.5, 0.6))
    assert len(table) == 3 * 2  # three input sets x two confidence levels
    assert set(table["inputs"]) == {"price", "+market", "+market+funding"}
    assert (table["trades"] >= 0).all()
    # One model per input set: its accuracy does not depend on the confidence level.
    assert (table.groupby("inputs")["accuracy"].nunique() == 1).all()
