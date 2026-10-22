import numpy as np
import pandas as pd
import pytest

from trading_agent.portfolio.backtest import run_portfolio_backtest, summarize_portfolio
from trading_agent.portfolio.strategies import (
    equal_weights,
    momentum_signal,
    momentum_weights,
    trend_signal,
    trend_weights,
    vol_sized,
    weekly,
)

DAYS = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")


def prices(**paths):
    return pd.DataFrame(paths, index=DAYS[: len(next(iter(paths.values())))])


def random_prices(n_coins=4, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    data = 100 * np.exp(np.cumsum(rng.normal(drift, 0.03, (len(DAYS), n_coins)), axis=0))
    return pd.DataFrame(data, index=DAYS, columns=[f"C{i}" for i in range(n_coins)])


def test_full_weight_without_costs_tracks_the_coin():
    opens = random_prices(2)
    weights = pd.DataFrame({"C0": 1.0, "C1": 0.0}, index=DAYS)
    result = run_portfolio_backtest(opens, weights, cost_rate=0.0)
    expected = 10_000 * opens["C0"].iloc[-1] / opens["C0"].iloc[1]  # bought at day 2's open
    assert result.equity.iloc[-1] == pytest.approx(expected)


def test_costs_are_charged_on_turnover():
    opens = prices(A=[100.0] * 10)
    weights = pd.DataFrame({"A": [1.0, 0.0] * 5}, index=DAYS[:10])  # flip every day
    result = run_portfolio_backtest(opens, weights, cost_rate=0.001)
    assert result.costs.sum() > 0
    assert result.equity.iloc[-1] < 10_000
    assert result.turnover.iloc[1] == pytest.approx(1.0)  # first buy: 100% of the account


def test_positions_drift_between_rebalances():
    opens = prices(A=[100.0, 100.0, 200.0, 200.0], B=[100.0] * 4)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=DAYS[:4])
    only_first = pd.Series([True, False, False, False], index=DAYS[:4])
    result = run_portfolio_backtest(opens, weights, cost_rate=0.0, rebalance=only_first)
    assert result.weights["A"].iloc[-1] == pytest.approx(2 / 3)  # A doubled and was not trimmed
    assert result.turnover.iloc[2:].sum() == 0


def test_rejects_leverage_and_shorts():
    opens = random_prices(2)
    with pytest.raises(ValueError, match="more than 100%"):
        run_portfolio_backtest(opens, pd.DataFrame({"C0": 0.8, "C1": 0.8}, index=DAYS), 0.0)
    with pytest.raises(ValueError, match="Short"):
        run_portfolio_backtest(opens, pd.DataFrame({"C0": -0.5, "C1": 0.0}, index=DAYS), 0.0)


def test_strategies_never_look_into_the_future():
    close = random_prices(5, seed=1)
    cutoff = 250
    changed = close.copy()
    changed.iloc[cutoff + 1:] *= 5
    for make in (trend_weights, momentum_weights, equal_weights):
        a = make(close).iloc[: cutoff + 1]
        b = make(changed).iloc[: cutoff + 1]
        pd.testing.assert_frame_equal(a, b)


def test_weights_respect_limits():
    close = random_prices(5, seed=2, drift=0.002)
    for make in (trend_weights, momentum_weights):
        w = make(close)
        assert (w >= 0).all().all()
        assert (w <= 0.25 + 1e-12).all().all()
        assert (w.sum(axis=1) <= 1 + 1e-9).all()


def test_trend_signal_votes():
    up = pd.DataFrame({"A": np.linspace(100, 300, 400)}, index=DAYS)
    down = pd.DataFrame({"A": np.linspace(300, 100, 400)}, index=DAYS)
    assert trend_signal(up)["A"].iloc[-1] == 1.0
    assert trend_signal(down)["A"].iloc[-1] == 0.0
    assert trend_signal(up)["A"].iloc[:120].eq(0).all()  # not enough history yet


def test_momentum_picks_best_positive_coins():
    close = pd.DataFrame({
        "A": np.linspace(100, 200, 400), "B": np.linspace(100, 150, 400),
        "C": np.linspace(100, 120, 400), "D": np.linspace(100, 50, 400),
    }, index=DAYS)
    last = momentum_signal(close, lookback=60, top=3).iloc[-1]
    assert list(last[last == 1].index) == ["A", "B", "C"]
    assert last["D"] == 0  # falling coin never held


def test_vol_sizing_gives_calmer_coins_more_weight():
    rng = np.random.default_rng(3)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 400)))
    close = pd.DataFrame({"calm": calm, "wild": wild}, index=DAYS)
    on = pd.DataFrame(1.0, index=DAYS, columns=close.columns)
    w = vol_sized(on, close, max_weight=1.0).iloc[-1]  # no cap, so sizing alone decides
    assert w["calm"] > 3 * w["wild"]  # about 5x calmer -> several times the weight


def test_weekly_rebalance_is_sunday_and_summary_runs():
    mask = weekly(DAYS)
    assert set(DAYS[mask.to_numpy()].day_of_week) == {6}
    close = random_prices(3, seed=4)
    stats = summarize_portfolio(run_portfolio_backtest(close, trend_weights(close), 0.0016, mask))
    assert stats["max_drawdown"] <= 0 and 0 <= stats["winning_weeks"] <= 1
