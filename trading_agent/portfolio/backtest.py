"""Portfolio backtester.

A strategy gives target weights per coin (share of the account, 0 to 1) decided
at each day's close. On rebalance days those targets are traded at the NEXT
day's open, paying `cost_rate` on every dollar bought or sold. Between
rebalances, positions drift with prices (no free rebalancing).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    equity: pd.Series  # account value at each day's open, after any trading
    weights: pd.DataFrame  # share of the account held in each coin, after trading
    turnover: pd.Series  # dollars traded each day / account value
    costs: pd.Series  # dollars paid in costs each day
    trades: pd.DataFrame  # dollars bought (+) or sold (-) per coin each day


def run_portfolio_backtest(
    opens: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_rate: float,
    rebalance: pd.Series | None = None,
    initial_cash: float = 10_000.0,
) -> PortfolioResult:
    """`opens`: daily open prices (dates x coins). `target_weights`: same shape, decided at each close."""
    target_weights = target_weights.reindex(index=opens.index, columns=opens.columns).fillna(0.0)
    if (target_weights < 0).any().any():
        raise ValueError("Short positions are not supported (spot trading only)")
    if (target_weights.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("Target weights add up to more than 100% (no leverage)")
    rebalance = (
        pd.Series(True, index=opens.index) if rebalance is None
        else rebalance.reindex(opens.index, fill_value=False)
    )

    prices = opens.to_numpy(dtype=float)
    targets = target_weights.to_numpy(dtype=float)
    do_rebalance = rebalance.to_numpy(dtype=bool)
    n, m = prices.shape
    holdings = np.zeros(m)  # dollars in each coin
    cash = initial_cash
    equity = np.empty(n)
    weights = np.zeros((n, m))
    turnover = np.zeros(n)
    costs = np.zeros(n)
    trades = np.zeros((n, m))

    for i in range(n):
        if i > 0:
            growth = np.where(np.isfinite(prices[i] / prices[i - 1]), prices[i] / prices[i - 1], 1.0)
            holdings = holdings * growth  # prices moved from yesterday's open to today's
        value = cash + holdings.sum()
        if i > 0 and do_rebalance[i - 1]:
            # Yesterday's close decided these targets; we trade them now, at today's open.
            tradable = np.isfinite(prices[i])
            target = np.where(tradable, targets[i - 1] * value, holdings)
            trades[i] = target - holdings
            traded = np.abs(trades[i]).sum()
            cost = traded * cost_rate
            cash = value - target.sum() - cost
            holdings = target
            turnover[i] = traded / value
            costs[i] = cost
            value = cash + holdings.sum()
        equity[i] = value
        weights[i] = holdings / value

    return PortfolioResult(
        equity=pd.Series(equity, index=opens.index, name="equity"),
        weights=pd.DataFrame(weights, index=opens.index, columns=opens.columns),
        turnover=pd.Series(turnover, index=opens.index, name="turnover"),
        costs=pd.Series(costs, index=opens.index, name="costs"),
        trades=pd.DataFrame(trades, index=opens.index, columns=opens.columns),
    )


def summarize_portfolio(result: PortfolioResult) -> dict:
    eq = result.equity
    daily = eq.pct_change().dropna()
    weekly = eq.resample("W").last().pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    std = daily.std()
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan"),
        "sharpe": daily.mean() / std * np.sqrt(365) if std > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "winning_weeks": float((weekly > 0).mean()) if len(weekly) else float("nan"),
        "avg_invested": float(result.weights.sum(axis=1).mean()),
        "turnover_per_year": float(result.turnover.sum() / years) if years > 0 else float("nan"),
        "costs_paid": float(result.costs.sum() / eq.iloc[0]),
    }
