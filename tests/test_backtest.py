import pandas as pd
import pytest

from trading_agent.backtest.engine import Costs, RiskRules, run_backtest
from trading_agent.backtest.metrics import max_drawdown, summarize
from trading_agent.market_data.synthetic import ohlcv_from_closes, random_walk_ohlcv

NO_COSTS = Costs(fee_rate=0, spread=0, slippage=0)


def candles(rows):
    """rows of (open, high, low, close), hourly from 2024-01-01."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


def test_flat_signal_does_nothing():
    df = random_walk_ohlcv(n=100)
    result = run_backtest(df, pd.Series(0, index=df.index))
    assert (result.equity == 10_000).all()
    assert result.trades.empty


def test_always_long_without_costs_matches_price_move():
    df = random_walk_ohlcv(n=100)
    result = run_backtest(df, pd.Series(1, index=df.index), costs=NO_COSTS)
    expected = 10_000 * df["close"].iloc[-1] / df["open"].iloc[1]
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert len(result.trades) == 1
    assert result.trades["exit_reason"].iloc[0] == "end_of_data"


def test_costs_are_charged_on_entry_and_exit():
    df = candles([(100, 100, 100, 100)] * 5)  # price never moves
    costs = Costs(fee_rate=0.001, spread=0.0002, slippage=0.0005)
    result = run_backtest(df, pd.Series(1, index=df.index), costs=costs)
    # Buy at 100.06, sell at 99.94, plus ~0.1% fee each way.
    assert result.equity.iloc[-1] == pytest.approx(10_000 * 0.9968, rel=1e-4)
    assert result.trades["pnl"].iloc[0] < 0


def test_signal_is_executed_on_the_next_candle():
    df = random_walk_ohlcv(n=20)
    signal = pd.Series(0, index=df.index)
    signal.iloc[5:] = 1
    trade = run_backtest(df, signal, costs=NO_COSTS).trades.iloc[0]
    assert trade["entry_time"] == df.index[6]
    assert trade["entry_price"] == df["open"].iloc[6]


def test_short_profits_when_price_falls():
    df = ohlcv_from_closes([100 - i for i in range(10)], start_price=100)
    result = run_backtest(df, pd.Series(-1, index=df.index), costs=NO_COSTS)
    assert result.equity.iloc[-1] > 10_000


def test_stop_loss_exits_and_waits_for_signal_change():
    df = candles([
        (100, 100, 100, 100),
        (100, 101, 99, 100),   # enter long at 100
        (100, 100, 90, 92),    # falls through the 2% stop at 98
        (92, 93, 91, 92),      # signal still long: must NOT re-enter
        (92, 93, 91, 92),
    ])
    result = run_backtest(df, pd.Series(1, index=df.index), costs=NO_COSTS,
                          risk=RiskRules(stop_loss=0.02))
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(98)
    assert result.equity.iloc[-1] == pytest.approx(9_800)


def test_stop_fills_at_open_when_price_gaps_past_it():
    df = candles([
        (100, 100, 100, 100),
        (100, 101, 99, 100),
        (90, 91, 89, 90),  # opens far below the stop
    ])
    result = run_backtest(df, pd.Series(1, index=df.index), costs=NO_COSTS,
                          risk=RiskRules(stop_loss=0.02))
    assert result.trades["exit_price"].iloc[0] == pytest.approx(90)


def test_position_fraction_limits_exposure():
    df = candles([(100, 100, 100, 100), (100, 100, 100, 100), (50, 50, 50, 50)])
    result = run_backtest(df, pd.Series(1, index=df.index), costs=NO_COSTS,
                          risk=RiskRules(position_fraction=0.5))
    assert result.equity.iloc[-1] == pytest.approx(7_500)  # half the money halved


def test_drawdown_halt_stops_trading():
    df = ohlcv_from_closes([100, 100, 80, 70, 120, 130], start_price=100)
    result = run_backtest(df, pd.Series(1, index=df.index), costs=NO_COSTS,
                          risk=RiskRules(max_drawdown_halt=0.15))
    assert result.trades["exit_reason"].iloc[0] == "drawdown_halt"
    assert len(result.trades) == 1
    assert (result.positions.iloc[3:] == 0).all()


def test_metrics():
    eq = pd.Series([100, 120, 90, 130], index=pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC"))
    assert max_drawdown(eq) == pytest.approx(-0.25)

    df = random_walk_ohlcv(n=500, seed=4)
    stats = summarize(run_backtest(df, pd.Series(1, index=df.index)))
    assert stats["trades"] == 1
    assert stats["exposure"] > 0.99
    assert stats["max_drawdown"] <= 0
