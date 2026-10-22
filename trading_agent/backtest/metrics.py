"""Performance statistics for a backtest."""

import math

import pandas as pd

from .engine import BacktestResult

SECONDS_PER_YEAR = 365.25 * 86400


def max_drawdown(equity: pd.Series) -> float:
    """Worst fall from a previous peak, as a negative fraction (-0.25 = lost 25% from the top)."""
    return float((equity / equity.cummax() - 1).min())


def summarize(result: BacktestResult) -> dict:
    eq = result.equity
    trades = result.trades
    returns = eq.pct_change().dropna()

    bar_seconds = pd.Series(eq.index).diff().dt.total_seconds().median()
    bars_per_year = SECONDS_PER_YEAR / bar_seconds
    years = (eq.index[-1] - eq.index[0]).total_seconds() / SECONDS_PER_YEAR

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else float("nan")
    std = returns.std()
    sharpe = returns.mean() / std * math.sqrt(bars_per_year) if std > 0 else 0.0

    wins = trades["pnl"][trades["pnl"] > 0].sum()
    losses = -trades["pnl"][trades["pnl"] <= 0].sum()
    profit_factor = wins / losses if losses > 0 else float("inf") if wins > 0 else float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "trades": len(trades),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else float("nan"),
        # Average profit per trade after all costs, as a share of the position: the number that matters.
        "avg_trade_return": float(trades["return"].mean()) if len(trades) else float("nan"),
        "profit_factor": profit_factor,
        "fees_paid": float(trades["fees"].sum()),
        "exposure": float((result.positions != 0).mean()),
    }
