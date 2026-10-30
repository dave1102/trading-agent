import numpy as np
import pandas as pd
import pytest

from trading_agent import trendpaper

COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
START = pd.Timestamp("2026-11-01", tz="UTC")  # a Sunday


def daily_prices(days_before=200, days_after=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START - pd.Timedelta(days=days_before), periods=days_before + days_after, freq="1D", tz="UTC")
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.003, 0.03, (len(idx), len(COINS))), axis=0)),
                          index=idx, columns=COINS)
    opens = closes.shift(1).bfill()
    return opens, closes


def run(opens, closes, today):
    return trendpaper.forward_results(opens, closes, START, today, cost_rate=0.0016, initial_cash=10_000)


def test_nothing_happens_before_the_start():
    opens, closes = daily_prices()
    assert run(opens, closes, START - pd.Timedelta(days=3)) == {}


def test_first_trade_is_on_monday_after_the_start():
    opens, closes = daily_prices()
    results = run(opens, closes, START + pd.Timedelta(days=10))
    for name in trendpaper.STRATEGIES:
        traded_days = results[name].turnover[results[name].turnover > 0].index
        assert traded_days[0] == START + pd.Timedelta(days=1)  # Monday 2 November
        assert results[name].equity.iloc[0] == 10_000


def test_todays_unfinished_close_is_never_used():
    opens, closes = daily_prices()
    today = START + pd.Timedelta(days=14)  # a Sunday: a rebalance decision day
    changed = closes.copy()
    changed.loc[changed.index >= today] *= 3  # wild move in the still-running day
    a = run(opens, closes, today)["trend"]
    b = run(opens, changed, today)["trend"]
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.weights, b.weights)


def test_snapshots_are_appended_once_and_drift_is_reported(tmp_path):
    opens, closes = daily_prices()
    log = tmp_path / "equity_log.csv"
    first = run(opens, closes, START + pd.Timedelta(days=5))
    assert trendpaper.append_snapshots(log, first) == (6, 0.0)
    later = run(opens, closes, START + pd.Timedelta(days=8))
    added, drift = trendpaper.append_snapshots(log, later)
    assert added == 3 and drift < 1e-5  # only cent rounding in the log
    assert len(pd.read_csv(log)) == 9

    revised = opens.copy()
    revised.loc[START + pd.Timedelta(days=3)] *= 1.2  # the exchange "changed" an old price
    _, drift = trendpaper.append_snapshots(log, run(revised, closes, START + pd.Timedelta(days=8)))
    assert drift > 0.01
    assert len(pd.read_csv(log)) == 9  # the log itself is not rewritten


def test_outputs_and_report(tmp_path, capsys):
    opens, closes = daily_prices(seed=3)
    results = run(opens, closes, START + pd.Timedelta(days=20))
    trendpaper.append_snapshots(tmp_path / "equity_log.csv", results)
    trendpaper.write_outputs(tmp_path, results)
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert set(trades["action"]) <= {"buy", "sell"} and len(trades) > 0
    assert trades["cost_usd"].sum() == pytest.approx(results["trend"].costs.sum(), rel=1e-3)
    cfg = {"trend_paper": {"start": "2026-11-01", "initial_cash": 10_000}}
    trendpaper.report(cfg, tmp_path)
    out = capsys.readouterr().out
    assert "trend" in out and "basket_hold" in out and "btc_hold" in out and "Last rebalance" in out


def test_report_before_start(tmp_path, capsys):
    trendpaper.report({"trend_paper": {"start": "2026-11-01", "initial_cash": 10_000}}, tmp_path)
    assert "Not started yet" in capsys.readouterr().out
