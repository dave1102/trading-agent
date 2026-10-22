"""A virtual trading account stored as files in one folder:

    state.json   cash, open positions, risk flags (so a run can be resumed)
    trades.csv   every simulated fill, plus orders the risk rules rejected
    equity.csv   account value after each candle
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..backtest.engine import Costs

TRADE_LOG_COLUMNS = ["time", "symbol", "action", "side", "qty", "price", "fee", "pnl", "reason", "cash_after"]
EQUITY_LOG_COLUMNS = ["time", "equity", "cash", "open_positions"]


@dataclass
class Position:
    side: int  # 1 = long, -1 = short
    qty: float  # always positive
    entry_price: float
    entry_time: str
    entry_fee: float


@dataclass
class AccountState:
    initial_cash: float
    cash: float
    positions: dict = field(default_factory=dict)  # symbol -> Position
    last_price: dict = field(default_factory=dict)  # symbol -> latest close
    last_time: dict = field(default_factory=dict)  # symbol -> ISO time of last processed candle
    pending: dict = field(default_factory=dict)  # symbol -> target position for the next open
    blocked: dict = field(default_factory=dict)  # symbol -> side not to re-enter until the signal changes
    rejected: dict = field(default_factory=dict)  # symbol -> side last rejected (to log it only once)
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    daily_stopped: bool = False
    halted: bool = False
    meta: dict = field(default_factory=dict)  # settings the run was started with


class PaperAccount:
    def __init__(self, directory: Path, initial_cash: float = 10_000.0, costs: Costs | None = None):
        self.dir = Path(directory)
        self.costs = costs or Costs()
        self.state_path = self.dir / "state.json"
        self.trades_path = self.dir / "trades.csv"
        self.equity_path = self.dir / "equity.csv"
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            data["positions"] = {sym: Position(**p) for sym, p in data["positions"].items()}
            self.state = AccountState(**data)
        else:
            self.state = AccountState(
                initial_cash=initial_cash,
                cash=initial_cash,
                peak_equity=initial_cash,
                day_start_equity=initial_cash,
            )

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2))
        tmp.replace(self.state_path)  # replace in one step so a crash never leaves half a file

    def equity(self) -> float:
        s = self.state
        return s.cash + sum(p.side * p.qty * s.last_price[sym] for sym, p in s.positions.items())

    def open_position(self, symbol: str, side: int, price: float, value: float, time, reason: str) -> None:
        fill = self.costs.fill_price(price, side)
        qty = value / (fill * (1 + self.costs.fee_rate))
        fee = qty * fill * self.costs.fee_rate
        self.state.cash -= side * qty * fill + fee
        self.state.positions[symbol] = Position(side, qty, fill, str(time), fee)
        self._log_trade(time, symbol, "open", side, qty, fill, fee, "", reason)

    def close_position(self, symbol: str, price: float, time, reason: str) -> float:
        p = self.state.positions.pop(symbol)
        fill = self.costs.fill_price(price, -p.side)
        fee = p.qty * fill * self.costs.fee_rate
        self.state.cash += p.side * p.qty * fill - fee
        pnl = p.side * p.qty * (fill - p.entry_price) - p.entry_fee - fee
        self._log_trade(time, symbol, "close", p.side, p.qty, fill, fee, pnl, reason)
        return pnl

    def log_rejection(self, symbol: str, side: int, time, reason: str) -> None:
        self._log_trade(time, symbol, "rejected", side, 0, "", 0, "", reason)

    def log_equity(self, time) -> None:
        row = [str(time), self.equity(), self.state.cash, len(self.state.positions)]
        self._append(self.equity_path, EQUITY_LOG_COLUMNS, row)

    def _log_trade(self, time, symbol, action, side, qty, price, fee, pnl, reason) -> None:
        row = [str(time), symbol, action, side, qty, price, fee, pnl, reason, self.state.cash]
        self._append(self.trades_path, TRADE_LOG_COLUMNS, row)

    def _append(self, path: Path, columns: list[str], row: list) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(columns)
            writer.writerow(row)
