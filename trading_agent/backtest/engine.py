"""Candle-by-candle backtester.

Timing rules (these prevent the most common backtesting lies):
- A signal decided at candle t's close is executed at candle t+1's OPEN.
- Stop-loss / take-profit are checked against each candle's high and low.
  If both could have been hit in the same candle, we assume the stop (worst case).
- Every fill pays half the spread plus slippage, and a fee on the trade value.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    fee_rate: float = 0.001  # commission, as a fraction of trade value (0.001 = 0.1%)
    spread: float = 0.0002  # full bid-ask spread as a fraction of price; each fill pays half
    slippage: float = 0.0005  # extra adverse price movement per fill

    def fill_price(self, price: float, side: int) -> float:
        """side=+1 when buying, -1 when selling. You always get a slightly worse price."""
        return price * (1 + side * (self.spread / 2 + self.slippage))


@dataclass
class RiskRules:
    position_fraction: float = 1.0  # share of current equity put into each position
    stop_loss: float | None = None  # e.g. 0.02 = exit if price moves 2% against us
    take_profit: float | None = None  # e.g. 0.04 = exit after a 4% favourable move
    max_drawdown_halt: float | None = None  # e.g. 0.25 = stop trading after a 25% fall from peak


@dataclass
class BacktestResult:
    equity: pd.Series  # account value at each candle's close
    positions: pd.Series  # -1 / 0 / 1 held at each candle's close
    trades: pd.DataFrame  # one row per completed trade


TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "qty", "entry_price", "exit_price",
    "fees", "pnl", "return", "exit_reason",
]


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    costs: Costs | None = None,
    risk: RiskRules | None = None,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    costs = costs or Costs()
    risk = risk or RiskRules()
    signal = signal.reindex(df.index).fillna(0).clip(-1, 1).astype(int).to_numpy()
    o, h, l, c = (df[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close"))
    times = df.index
    n = len(df)

    cash, qty = initial_cash, 0.0  # qty is signed: negative = short
    entry = None  # details of the open position, or None when flat
    blocked_side = 0  # after a stop/target exit, don't re-enter that side until the signal changes
    halted = False
    peak = initial_cash
    trades = []
    equity = np.empty(n)
    positions = np.zeros(n, dtype=int)

    def open_position(i, side, price):
        nonlocal cash, qty, entry
        fill = costs.fill_price(price, side)
        units = risk.position_fraction * cash / (fill * (1 + costs.fee_rate))
        fee = units * fill * costs.fee_rate
        qty = side * units
        cash -= qty * fill + fee
        entry = {"entry_time": times[i], "side": side, "qty": units, "entry_price": fill, "fees": fee}

    def close_position(i, price, reason):
        nonlocal cash, qty, entry
        side = entry["side"]
        fill = costs.fill_price(price, -side)
        fee = entry["qty"] * fill * costs.fee_rate
        cash += qty * fill - fee
        fees = entry["fees"] + fee
        pnl = side * entry["qty"] * (fill - entry["entry_price"]) - fees
        trades.append({
            **entry,
            "exit_time": times[i],
            "exit_price": fill,
            "fees": fees,
            "pnl": pnl,
            "return": pnl / (entry["qty"] * entry["entry_price"]),
            "exit_reason": reason,
        })
        qty, entry = 0.0, None

    for i in range(n):
        # 1. At the open: move to the position the previous candle's signal asked for.
        target = 0 if (i == 0 or halted) else int(signal[i - 1])
        if blocked_side and target != blocked_side:
            blocked_side = 0
        if blocked_side:
            target = 0
        current = entry["side"] if entry else 0
        if target != current:
            if entry:
                close_position(i, o[i], "signal")
            if target:
                open_position(i, target, o[i])

        # 2. During the candle: did price touch the stop-loss or take-profit?
        if entry:
            side, ref = entry["side"], entry["entry_price"]
            exit_price = reason = None
            if risk.stop_loss:
                stop = ref * (1 - side * risk.stop_loss)
                if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
                    # If the candle opened beyond the stop (a gap), we get the open price.
                    exit_price = min(o[i], stop) if side == 1 else max(o[i], stop)
                    reason = "stop_loss"
            if exit_price is None and risk.take_profit:
                target_price = ref * (1 + side * risk.take_profit)
                if (side == 1 and h[i] >= target_price) or (side == -1 and l[i] <= target_price):
                    exit_price = max(o[i], target_price) if side == 1 else min(o[i], target_price)
                    reason = "take_profit"
            if exit_price is not None:
                close_position(i, exit_price, reason)
                blocked_side = side

        # 3. At the close: value the account and apply the drawdown circuit breaker.
        value = cash + qty * c[i]
        peak = max(peak, value)
        if risk.max_drawdown_halt and not halted and value < peak * (1 - risk.max_drawdown_halt):
            if entry:
                close_position(i, c[i], "drawdown_halt")
            halted = True
            value = cash
        equity[i] = value
        positions[i] = entry["side"] if entry else 0

    if entry:
        close_position(n - 1, c[-1], "end_of_data")
        equity[-1] = cash
        positions[-1] = 0

    return BacktestResult(
        equity=pd.Series(equity, index=times, name="equity"),
        positions=pd.Series(positions, index=times, name="position"),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
    )
