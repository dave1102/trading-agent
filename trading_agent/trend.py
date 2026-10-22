"""Test the multi-coin trend and momentum strategies against simple holding.

Pass rule (fixed before any results were seen): in EVERY period and coin group,
the strategy must beat the equal-weight basket on Sharpe ratio AND have a worst
drawdown at least one third smaller than the basket's.

Example:
    python -m trading_agent.trend
"""

import argparse

import pandas as pd

from .backtest.engine import Costs
from .config import load_config
from .market_data.storage import data_path, load_ohlcv
from .market_data.validate import resample_ohlcv
from .portfolio.backtest import run_portfolio_backtest, summarize_portfolio
from .portfolio.strategies import equal_weights, momentum_weights, single_coin, trend_weights, weekly

# Starts in May 2022 so the 120-day trend lookback has history; every strategy uses the same dates.
PERIODS = {"2022-05 to 2024": ("2022-05-01", "2024-12-31"), "2025 to now": ("2025-01-01", None)}


def load_daily(symbols: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for symbol in symbols:
        daily = resample_ohlcv(load_ohlcv(data_path("crypto", symbol, cfg["crypto"]["timeframe"])), "1d")
        opens[symbol], closes[symbol] = daily["open"], daily["close"]
    return pd.DataFrame(opens), pd.DataFrame(closes)


def evaluate_group(opens, closes, cost_rate) -> pd.DataFrame:
    """One row per (period, strategy). Signals use all history; trading starts at each period."""
    strategies = {
        "trend": trend_weights(closes),
        "momentum": momentum_weights(closes),
        "basket_hold": equal_weights(closes),
    }
    if "BTC/USDT" in closes.columns:
        strategies["btc_hold"] = single_coin(closes, "BTC/USDT")
    rows = []
    for period, (start, end) in PERIODS.items():
        o = opens.loc[start:end]
        for name, weights in strategies.items():
            result = run_portfolio_backtest(o, weights.loc[o.index], cost_rate, weekly(o.index))
            rows.append({"period": period, "strategy": name, **summarize_portfolio(result)})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, strategy: str) -> list[str]:
    """Check the pass rule for `strategy` in every period of one coin group."""
    failures = []
    for period, group in table.groupby("period"):
        s = group.set_index("strategy")
        if s.loc[strategy, "sharpe"] <= s.loc["basket_hold", "sharpe"]:
            failures.append(f"{period}: Sharpe {s.loc[strategy, 'sharpe']:.2f} not above basket "
                            f"{s.loc['basket_hold', 'sharpe']:.2f}")
        if s.loc[strategy, "max_drawdown"] > 0 or s.loc[strategy, "max_drawdown"] < (2 / 3) * s.loc["basket_hold", "max_drawdown"]:
            failures.append(f"{period}: drawdown {s.loc[strategy, 'max_drawdown']:.0%} not 1/3 smaller than basket "
                            f"{s.loc['basket_hold', 'max_drawdown']:.0%}")
    return failures


def show(table: pd.DataFrame) -> str:
    shown = table.copy()
    for col in ("total_return", "cagr", "max_drawdown", "winning_weeks", "avg_invested", "costs_paid"):
        shown[col] = table[col].map("{:+.1%}".format if col in ("total_return", "cagr") else "{:.1%}".format)
    shown["sharpe"] = table["sharpe"].map("{:.2f}".format)
    shown["turnover_per_year"] = table["turnover_per_year"].map("{:.1f}x".format)
    return shown.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-coin trend/momentum vs holding, with a fixed pass rule.")
    parser.parse_args()
    cfg = load_config()
    pcfg = cfg["portfolio"]
    c = Costs(**cfg["costs"]["crypto"])
    cost_rate = c.fee_rate + c.spread / 2 + c.slippage  # paid on every dollar traded

    opens, closes = load_daily(pcfg["symbols"], cfg)
    new = [s for s in pcfg["symbols"] if s not in pcfg["original"]]
    groups = {"all 13 coins": pcfg["symbols"], "original 3": pcfg["original"], "10 new coins": new}

    results = {}
    for group, symbols in groups.items():
        table = evaluate_group(opens[symbols], closes[symbols], cost_rate)
        results[group] = table
        print(f"\n=== {group} (costs {cost_rate:.2%} per dollar traded) ===")
        print(show(table))

    doubled = evaluate_group(opens, closes, 2 * cost_rate)
    print(f"\n=== all 13 coins with DOUBLE costs ({2 * cost_rate:.2%}) ===")
    print(show(doubled[doubled["strategy"].isin(["trend", "momentum"])]))

    print("\n=== VERDICT (rule fixed in advance) ===")
    for strategy in ("trend", "momentum"):
        failures = [f"[{g}] {f}" for g, t in results.items() for f in verdict(t, strategy)]
        if failures:
            print(f"{strategy}: FAIL")
            for f in failures:
                print("   ", f)
        else:
            print(f"{strategy}: PASS in every period and coin group")


if __name__ == "__main__":
    main()
