import ast
from pathlib import Path

import pandas as pd
import pytest

import trading_agent.paper
from trading_agent.backtest.engine import Costs
from trading_agent.market_data.synthetic import random_walk_ohlcv
from trading_agent.paper.account import PaperAccount
from trading_agent.paper.trader import Candle, PaperRiskLimits, PaperTrader, replay
from trading_agent.strategies import sma_crossover

NO_COSTS = Costs(fee_rate=0, spread=0, slippage=0)
NO_LIMITS = PaperRiskLimits(daily_loss_limit=None, max_drawdown_halt=None)


def candle(hour, o, h=None, l=None, c=None, day="2025-01-06"):
    h = max(o, c or o) if h is None else h
    l = min(o, c or o) if l is None else l
    return Candle(pd.Timestamp(f"{day} {hour:02d}:00", tz="UTC"), o, h, l, o if c is None else c)


def make_trader(tmp_path, limits=NO_LIMITS):
    return PaperTrader(PaperAccount(tmp_path / "acct", costs=NO_COSTS), limits)


def read_trades(tmp_path):
    return pd.read_csv(tmp_path / "acct" / "trades.csv")


def test_open_and_close_accounting(tmp_path):
    acct = PaperAccount(tmp_path / "acct", costs=NO_COSTS)
    acct.open_position("EURUSD", 1, 100, 2_500, "t1", "signal")
    acct.state.last_price["EURUSD"] = 110
    assert acct.equity() == pytest.approx(10_250)
    pnl = acct.close_position("EURUSD", 110, "t2", "signal")
    assert pnl == pytest.approx(250)
    assert acct.state.cash == pytest.approx(10_250)
    assert list(read_trades(tmp_path)["action"]) == ["open", "close"]


def test_short_accounting(tmp_path):
    acct = PaperAccount(tmp_path / "acct", costs=NO_COSTS)
    acct.open_position("EURUSD", -1, 100, 2_500, "t1", "signal")
    pnl = acct.close_position("EURUSD", 90, "t2", "signal")
    assert pnl == pytest.approx(250)
    assert acct.state.cash == pytest.approx(10_250)


def test_state_survives_restart(tmp_path):
    acct = PaperAccount(tmp_path / "acct", costs=NO_COSTS)
    acct.open_position("EURUSD", 1, 100, 2_500, "t1", "signal")
    acct.save()
    reloaded = PaperAccount(tmp_path / "acct", costs=NO_COSTS)
    assert reloaded.state.cash == pytest.approx(7_500)
    assert reloaded.state.positions["EURUSD"].qty == pytest.approx(25)


def test_signal_fills_at_next_open(tmp_path):
    trader = make_trader(tmp_path)
    trader.process_candle("EURUSD", candle(0, 100), signal=1)
    assert not trader.account.state.positions
    trader.process_candle("EURUSD", candle(1, 101), signal=1)
    assert trader.account.state.positions["EURUSD"].entry_price == 101


def test_max_open_positions_rejects_and_logs_once(tmp_path):
    trader = make_trader(tmp_path, PaperRiskLimits(max_open_positions=2, daily_loss_limit=None, max_drawdown_halt=None))
    for hour in range(3):
        for sym in ["A", "B", "C"]:
            trader.process_candle(sym, candle(hour, 100), signal=1)
    assert set(trader.account.state.positions) == {"A", "B"}
    rejected = read_trades(tmp_path).query("action == 'rejected'")
    assert list(rejected["symbol"]) == ["C"]
    assert rejected["reason"].iloc[0] == "max_open_positions"


def test_stop_loss_and_no_reentry(tmp_path):
    trader = make_trader(tmp_path, PaperRiskLimits(stop_loss=0.02, daily_loss_limit=None, max_drawdown_halt=None))
    trader.process_candle("EURUSD", candle(0, 100), 1)
    trader.process_candle("EURUSD", candle(1, 100, l=97, c=97), 1)  # enter at 100, stop at 98
    trader.process_candle("EURUSD", candle(2, 97), 1)  # signal still long: stay out
    assert not trader.account.state.positions
    trades = read_trades(tmp_path)
    assert trades["reason"].iloc[-1] == "stop_loss"
    assert trades["price"].iloc[-1] == pytest.approx(98)
    trader.process_candle("EURUSD", candle(3, 97), 0)
    trader.process_candle("EURUSD", candle(4, 97), 1)
    trader.process_candle("EURUSD", candle(5, 97), 1)  # signal changed and came back: may re-enter
    assert "EURUSD" in trader.account.state.positions


def test_daily_loss_limit_pauses_until_next_day(tmp_path):
    limits = PaperRiskLimits(position_fraction=1.0, daily_loss_limit=0.03, max_drawdown_halt=None)
    trader = make_trader(tmp_path, limits)
    trader.process_candle("EURUSD", candle(0, 100), 1)
    trader.process_candle("EURUSD", candle(1, 100, c=95), 1)  # -5% in a day
    assert trader.account.state.daily_stopped
    assert not trader.account.state.positions
    trader.process_candle("EURUSD", candle(2, 95), 1)
    assert not trader.account.state.positions  # still paused today
    trader.process_candle("EURUSD", candle(0, 95, day="2025-01-07"), 1)
    assert "EURUSD" in trader.account.state.positions  # new day: allowed again


def test_drawdown_halt_is_permanent(tmp_path):
    limits = PaperRiskLimits(position_fraction=1.0, daily_loss_limit=None, max_drawdown_halt=0.1)
    trader = make_trader(tmp_path, limits)
    trader.process_candle("EURUSD", candle(0, 100), 1)
    trader.process_candle("EURUSD", candle(1, 100, c=85), 1)
    assert trader.account.state.halted
    trader.process_candle("EURUSD", candle(0, 85, day="2025-02-01"), 1)
    trader.process_candle("EURUSD", candle(1, 85, day="2025-02-01"), 1)
    assert not trader.account.state.positions


def test_resumed_replay_matches_single_run(tmp_path):
    frames = {
        "AAA": random_walk_ohlcv(n=600, seed=1),
        "BBB": random_walk_ohlcv(n=600, seed=2),
    }
    signals = {sym: sma_crossover(df, 10, 30, allow_short=True) for sym, df in frames.items()}
    limits = PaperRiskLimits(stop_loss=0.02)
    costs = Costs()

    full = PaperTrader(PaperAccount(tmp_path / "full", costs=costs), limits)
    assert replay(full, frames, signals) == 1200

    split_dir = tmp_path / "split"
    first = PaperTrader(PaperAccount(split_dir, costs=costs), limits)
    replay(first, {s: df.iloc[:300] for s, df in frames.items()}, signals)
    first.account.save()
    second = PaperTrader(PaperAccount(split_dir, costs=costs), limits)
    assert replay(second, frames, signals) == 600  # only the new candles

    assert second.account.state.cash == pytest.approx(full.account.state.cash)
    full_trades = pd.read_csv(tmp_path / "full" / "trades.csv")
    split_trades = pd.read_csv(split_dir / "trades.csv")
    assert len(full_trades) == len(split_trades) > 10


def test_paper_package_cannot_reach_an_exchange():
    """Safety: no network or exchange library may be imported by paper-trading code."""
    forbidden = {"ccxt", "requests", "urllib", "http", "socket", "aiohttp", "websockets"}
    package_dir = Path(trading_agent.paper.__file__).parent
    for source in package_dir.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in forbidden, f"{source.name} imports {name}"
