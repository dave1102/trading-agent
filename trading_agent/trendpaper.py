"""Forward paper test of the multi-coin trend strategy (the one that passed its historical test).

Settings are exactly those tested in trading_agent.trend; nothing is tuned. Every hour it
adds the newest MEXC candles, uses only COMPLETED days, and replays three virtual $10,000
portfolios from the fixed start date:

    trend        the strategy (weekly rebalance, volatility sizing, 25% cap, full costs)
    basket_hold  all coins in equal shares (benchmark)
    btc_hold     100% BTC (benchmark)

Because every decision uses only past prices, replaying from the start date gives the same
result as a live paper account. Each day's values are also appended to equity_log.csv and
never rewritten, so a later change in the exchange's history would show up as a warning.

Examples:
    python -m trading_agent.trendpaper update     # fetch new candles and update the portfolios once
    python -m trading_agent.trendpaper report
    python -m trading_agent.trendpaper run        # update every hour until stopped (used on the server)
"""

import argparse
import time
import traceback
from pathlib import Path

import ccxt
import pandas as pd

from .backtest.engine import Costs
from .config import PROJECT_ROOT, load_config
from .market_data import crypto
from .market_data.storage import data_path, load_ohlcv, save_ohlcv
from .memescan import log_to_file, single_instance
from .portfolio.backtest import PortfolioResult, run_portfolio_backtest, summarize_portfolio
from .portfolio.strategies import equal_weights, single_coin, trend_weights, weekly
from .trend import load_daily

OUT_DIR = PROJECT_ROOT / "data" / "trend"
HISTORY_DAYS = 220  # enough for the 120-day trend and 60-day volatility lookbacks
STRATEGIES = ("trend", "basket_hold", "btc_hold")


def update_data(cfg: dict, exchange=None) -> None:
    """Add the newest hourly candles for every coin (the first run fetches ~220 days)."""
    tcfg = cfg["trend_paper"]
    exchange = exchange or crypto.make_exchange(tcfg["exchange"])
    history_start = pd.Timestamp(tcfg["start"], tz="UTC") - pd.Timedelta(days=HISTORY_DAYS)
    for symbol in cfg["portfolio"]["symbols"]:
        path = data_path("crypto", symbol, cfg["crypto"]["timeframe"])
        since = load_ohlcv(path).index[-1] if path.exists() else history_start
        try:
            save_ohlcv(crypto.fetch_ohlcv(symbol, cfg["crypto"]["timeframe"], since=since, exchange=exchange), path)
        except ccxt.BaseError as e:
            print(f"  {symbol}: not updated ({type(e).__name__}); will retry next hour")


def forward_results(opens: pd.DataFrame, closes: pd.DataFrame, start: pd.Timestamp, today: pd.Timestamp,
                    cost_rate: float, initial_cash: float) -> dict[str, PortfolioResult]:
    """Replay the three portfolios from `start` using only days completed before `today`.

    Decisions use closes of finished days only; today's open (already known) is used to
    value the portfolios and to fill trades decided at yesterday's close.
    """
    closes = closes[closes.index < today]
    opens = opens[(opens.index >= start) & (opens.index <= today)]
    if opens.empty:
        return {}
    weights = {
        "trend": trend_weights(closes),
        "basket_hold": equal_weights(closes),
        "btc_hold": single_coin(closes, "BTC/USDT"),
    }
    rebalance = weekly(opens.index)
    return {
        name: run_portfolio_backtest(opens, w.reindex(opens.index), cost_rate, rebalance, initial_cash)
        for name, w in weights.items()
    }


def append_snapshots(path: Path, results: dict[str, PortfolioResult]) -> tuple[int, float]:
    """Append each new day's portfolio values to the log; existing days are never rewritten.

    Returns (days added, largest relative difference between logged and recomputed values).
    A difference means the exchange changed past prices; it is reported, never hidden.
    """
    table = pd.DataFrame({name: r.equity for name, r in results.items()})
    table.index = table.index.strftime("%Y-%m-%d")
    table.index.name = "date"
    drift = 0.0
    if path.exists():
        logged = pd.read_csv(path, index_col="date")
        common = logged.index.intersection(table.index)
        if len(common):
            drift = float((table.loc[common, logged.columns] / logged.loc[common] - 1).abs().max().max())
        table = table[~table.index.isin(logged.index)]
    if not table.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.round(2).to_csv(path, mode="a", header=not path.exists())
    return len(table), drift


def write_outputs(out_dir: Path, results: dict[str, PortfolioResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trend = results["trend"]
    trades = trend.trades.stack().rename("usd").reset_index()
    trades.columns = ["date", "symbol", "usd"]
    trades = trades[trades["usd"].abs() > 0.01]
    trades["action"] = trades["usd"].map(lambda v: "buy" if v > 0 else "sell")
    trades["cost_usd"] = trades["usd"].abs() * (trend.costs.reindex(trades["date"]).to_numpy()
                                                / trend.trades.abs().sum(axis=1).reindex(trades["date"]).to_numpy())
    trades.round({"usd": 2, "cost_usd": 2}).to_csv(out_dir / "trades.csv", index=False)
    latest = trend.weights.iloc[-1]
    held = latest[latest > 0.001].sort_values(ascending=False).rename("weight").rename_axis("symbol")
    held.round(4).to_csv(out_dir / "holdings.csv")


def update(cfg: dict, out_dir: Path = OUT_DIR, today: pd.Timestamp | None = None, fetch: bool = True) -> dict:
    tcfg = cfg["trend_paper"]
    if fetch:
        update_data(cfg)
    opens, closes = load_daily(cfg["portfolio"]["symbols"], cfg)
    c = Costs(**cfg["costs"]["crypto"])
    today = today or pd.Timestamp.now(tz="UTC").floor("1D")
    results = forward_results(opens, closes, pd.Timestamp(tcfg["start"], tz="UTC"), today,
                              c.fee_rate + c.spread / 2 + c.slippage, tcfg["initial_cash"])
    if not results:
        print(f"[{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC] prices updated; "
              f"waiting for the start date {tcfg['start']}")
    else:
        added, drift = append_snapshots(out_dir / "equity_log.csv", results)
        write_outputs(out_dir, results)
        print(f"[{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC] updated to {today:%Y-%m-%d}"
              f"{f', logged {added} new day(s)' if added else ''}")
        if drift > 0.005:
            print(f"  WARNING: recomputed past values differ from the log by up to {drift:.1%} "
                  "(the exchange changed some past prices). The log keeps the original values.")
    return results


def report(cfg: dict, out_dir: Path = OUT_DIR) -> None:
    tcfg = cfg["trend_paper"]
    print(f"TREND STRATEGY - forward paper test (started {tcfg['start']}, ${tcfg['initial_cash']:,} each)")
    log_path = out_dir / "equity_log.csv"
    if not log_path.exists():
        print(f"  Not started yet: the first decision is at the close of {tcfg['start']} (UTC),")
        print("  the first trade at the next day's open.")
        return
    log = pd.read_csv(log_path, index_col="date", parse_dates=["date"])
    days = len(log)
    print(f"  {days} day(s) so far, latest {log.index[-1]:%Y-%m-%d}\n")
    rows = []
    for name in STRATEGIES:
        eq = log[name]
        rows.append({"portfolio": name, "value": f"${eq.iloc[-1]:,.0f}",
                     "return": f"{eq.iloc[-1] / tcfg['initial_cash'] - 1:+.1%}",
                     "worst_drop": f"{(eq / eq.cummax() - 1).min():.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))

    holdings = out_dir / "holdings.csv"
    if holdings.exists():
        h = pd.read_csv(holdings)
        invested = h["weight"].sum()
        print(f"\n  Trend portfolio now: {invested:.0%} invested, {1 - invested:.0%} cash")
        for _, row in h.iterrows():
            print(f"    {row['symbol']:<10} {row['weight']:.1%}")
    trades = out_dir / "trades.csv"
    if trades.exists():
        t = pd.read_csv(trades)
        if not t.empty:
            last = t[t["date"] == t["date"].max()]
            print(f"\n  Last rebalance {last['date'].iloc[0][:10]}: {len(last)} trade(s), "
                  f"costs ${last['cost_usd'].sum():.2f}; {t['date'].nunique()} rebalance(s) in total")
    print("\n  One data point a week: judge it after 6+ months, not weeks. Paper only, no real money.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward paper test of the multi-coin trend strategy.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update")
    sub.add_parser("report")
    p_run = sub.add_parser("run", help="update every hour until stopped")
    p_run.add_argument("--every", type=float, default=60, help="minutes between updates")
    p_run.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.log:
        log_to_file(args.log)
    cfg = load_config()
    if args.command == "update":
        update(cfg)
    elif args.command == "report":
        report(cfg)
    else:
        with single_instance(OUT_DIR / "trendpaper.lock"):
            print(f"Trend paper test started: updating every {args.every:g} minutes.")
            while True:
                try:
                    update(cfg)
                except Exception:  # running unattended: log it and retry next time
                    traceback.print_exc()
                time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
