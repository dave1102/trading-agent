"""Drives a PaperAccount one candle at a time, applying the risk limits.

For each candle of a symbol, in this order:
1. At the open: trade towards the position the previous candle's signal asked for.
2. During the candle: exit if the stop-loss or take-profit was touched.
3. At the close: update prices, check account-wide limits, store the new signal.
"""

from dataclasses import dataclass

import pandas as pd

from .account import PaperAccount


@dataclass
class PaperRiskLimits:
    position_fraction: float = 0.25  # share of equity per new position
    max_open_positions: int = 3
    max_gross_exposure: float = 1.0  # total position value / equity (1.0 = no leverage)
    stop_loss: float | None = None
    take_profit: float | None = None
    daily_loss_limit: float | None = 0.03  # close all and pause until the next UTC day
    max_drawdown_halt: float | None = 0.3  # close all and stop for good


@dataclass
class Candle:
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float


class PaperTrader:
    def __init__(self, account: PaperAccount, limits: PaperRiskLimits):
        self.account = account
        self.limits = limits

    def process_candle(self, symbol: str, candle: Candle, signal: int) -> None:
        s = self.account.state
        self._start_day_if_new(candle.time)
        self._execute_pending(symbol, candle)
        self._check_exits(symbol, candle)
        s.last_price[symbol] = candle.close
        self._check_account_limits(candle.time)
        s.pending[symbol] = int(signal)
        s.last_time[symbol] = candle.time.isoformat()

    def _start_day_if_new(self, time: pd.Timestamp) -> None:
        s = self.account.state
        day = time.date().isoformat()
        if day != s.day:
            s.day = day
            s.day_start_equity = self.account.equity()
            s.daily_stopped = False

    def _execute_pending(self, symbol: str, candle: Candle) -> None:
        s = self.account.state
        target = 0 if (s.halted or s.daily_stopped) else s.pending.get(symbol, 0)
        blocked = s.blocked.get(symbol, 0)
        if blocked and target != blocked:
            del s.blocked[symbol]
        elif blocked:
            target = 0
        if s.rejected.get(symbol) != target:
            s.rejected.pop(symbol, None)

        position = s.positions.get(symbol)
        current = position.side if position else 0
        if target == current:
            return
        if position:
            self.account.close_position(symbol, candle.open, candle.time, "signal")
        if target:
            reason, value = self._entry_check(target)
            if reason is None:
                self.account.open_position(symbol, target, candle.open, value, candle.time, "signal")
            elif s.rejected.get(symbol) != target:
                self.account.log_rejection(symbol, target, candle.time, reason)
                s.rejected[symbol] = target

    def _entry_check(self, side: int) -> tuple[str | None, float]:
        """Return (reason the order is refused, or None) and the position value to use."""
        s = self.account.state
        equity = self.account.equity()
        value = self.limits.position_fraction * equity
        gross = sum(p.qty * s.last_price[sym] for sym, p in s.positions.items())
        if len(s.positions) >= self.limits.max_open_positions:
            return "max_open_positions", value
        if gross + value > self.limits.max_gross_exposure * equity + 1e-9:
            return "max_gross_exposure", value
        if side == 1 and value > s.cash:
            return "insufficient_cash", value
        return None, value

    def _check_exits(self, symbol: str, candle: Candle) -> None:
        s = self.account.state
        position = s.positions.get(symbol)
        if not position:
            return
        side, ref = position.side, position.entry_price
        exit_price = reason = None
        if self.limits.stop_loss:
            stop = ref * (1 - side * self.limits.stop_loss)
            if (side == 1 and candle.low <= stop) or (side == -1 and candle.high >= stop):
                exit_price = min(candle.open, stop) if side == 1 else max(candle.open, stop)
                reason = "stop_loss"
        if exit_price is None and self.limits.take_profit:
            target = ref * (1 + side * self.limits.take_profit)
            if (side == 1 and candle.high >= target) or (side == -1 and candle.low <= target):
                exit_price = max(candle.open, target) if side == 1 else min(candle.open, target)
                reason = "take_profit"
        if exit_price is not None:
            self.account.close_position(symbol, exit_price, candle.time, reason)
            s.blocked[symbol] = side

    def _check_account_limits(self, time: pd.Timestamp) -> None:
        s = self.account.state
        equity = self.account.equity()
        s.peak_equity = max(s.peak_equity, equity)
        if self.limits.max_drawdown_halt and not s.halted and equity < s.peak_equity * (1 - self.limits.max_drawdown_halt):
            self._close_all(time, "drawdown_halt")
            s.halted = True
        elif (
            self.limits.daily_loss_limit
            and not s.daily_stopped
            and equity < s.day_start_equity * (1 - self.limits.daily_loss_limit)
        ):
            self._close_all(time, "daily_loss_limit")
            s.daily_stopped = True

    def _close_all(self, time, reason: str) -> None:
        s = self.account.state
        for symbol in list(s.positions):
            self.account.close_position(symbol, s.last_price[symbol], time, reason)


def replay(trader: PaperTrader, frames: dict[str, pd.DataFrame], signals: dict[str, pd.Series]) -> int:
    """Feed candles from several symbols to the trader in time order.

    Candles already processed in an earlier run are skipped, so a run can be
    stopped and resumed. Returns the number of candles processed.
    """
    s = trader.account.state
    events = pd.concat([df.assign(symbol=sym) for sym, df in frames.items()])
    events = events.sort_values("symbol", kind="stable").sort_index(kind="stable")
    processed = 0
    for time, group in events.groupby(level=0, sort=True):
        did_any = False
        for row in group.itertuples():
            last = s.last_time.get(row.symbol)
            if last is not None and time <= pd.Timestamp(last):
                continue
            candle = Candle(time, row.open, row.high, row.low, row.close)
            signal = signals[row.symbol].get(time, 0)
            trader.process_candle(row.symbol, candle, signal)
            processed += 1
            did_any = True
        if did_any:
            trader.account.log_equity(time)
    return processed
