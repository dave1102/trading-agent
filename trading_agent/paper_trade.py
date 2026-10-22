"""Paper trade by replaying historical candles one at a time into a virtual account.

Examples:
    python -m trading_agent.paper_trade --market forex --start 2025-01-01
    python -m trading_agent.paper_trade --market forex --symbols EURUSD --strategy sma --account sma_test
    python -m trading_agent.paper_trade --market forex --start 2025-01-01 --reset

Running the same command again resumes the account; only new candles are processed.

The model only ever sees data from before --start for training. Signals for
the replay period are computed up front, which gives the same result as
computing them candle by candle because every feature uses past data only
(tests/test_features.py checks this).
"""

import argparse
import shutil

import pandas as pd

from . import strategies
from .backtest.engine import Costs
from .config import PROJECT_ROOT, load_config
from .features import build_dataset, make_features
from .market_data.storage import data_path, load_ohlcv
from .models.baseline import make_model, probabilities_to_signal
from .paper.account import PaperAccount
from .paper.trader import PaperRiskLimits, PaperTrader, replay

ACCOUNTS_DIR = PROJECT_ROOT / "paper_accounts"


def ml_signal(df: pd.DataFrame, start: pd.Timestamp, model_cfg: dict) -> pd.Series:
    horizon = model_cfg["horizon"]
    X, y = build_dataset(df, horizon)
    before = X.index < start
    # Drop the last rows before start: their labels peek at prices after start.
    X_train, y_train = X[before].iloc[: -(horizon + 1)], y[before].iloc[: -(horizon + 1)]
    model = make_model().fit(X_train, y_train)

    features = make_features(df).loc[start:]
    usable = features.dropna()
    proba = pd.Series(model.predict_proba(usable)[:, 1], index=usable.index)
    signal = probabilities_to_signal(proba, model_cfg["threshold"], model_cfg["allow_short"])
    return signal.reindex(features.index, fill_value=0)


def limits_from_config(cfg: dict) -> PaperRiskLimits:
    return PaperRiskLimits(
        position_fraction=cfg["position_fraction"],
        max_open_positions=cfg["max_open_positions"],
        max_gross_exposure=cfg["max_gross_exposure"],
        stop_loss=cfg["stop_loss"] or None,
        take_profit=cfg["take_profit"] or None,
        daily_loss_limit=cfg["daily_loss_limit"] or None,
        max_drawdown_halt=cfg["max_drawdown_halt"] or None,
    )


def print_summary(account: PaperAccount) -> None:
    s = account.state
    equity = account.equity()
    print(f"\nEquity: {equity:,.2f}  (started with {s.initial_cash:,.2f}, {equity / s.initial_cash - 1:+.2%})")
    if account.trades_path.exists():
        trades = pd.read_csv(account.trades_path)
        closed = trades[trades["action"] == "close"]
        if len(closed):
            pnl = closed["pnl"].astype(float)
            print(f"Closed trades: {len(closed)}, win rate {(pnl > 0).mean():.1%}, "
                  f"profit per trade {pnl.mean():+,.2f}, total P&L {pnl.sum():+,.2f}")
            print("Exit reasons:", closed["reason"].value_counts().to_dict())
        rejected = trades[trades["action"] == "rejected"]
        if len(rejected):
            print("Rejected orders:", rejected["reason"].value_counts().to_dict())
    if account.equity_path.exists():
        eq = pd.read_csv(account.equity_path)["equity"]
        print(f"Max drawdown: {(eq / eq.cummax() - 1).min():.2%}")
    print("Open positions:", {sym: "long" if p.side == 1 else "short" for sym, p in s.positions.items()} or "none")
    if s.halted:
        print("ACCOUNT HALTED by the max-drawdown rule.")
    print(f"Logs: {account.trades_path}\n      {account.equity_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trade on historical candles (no real orders).")
    parser.add_argument("--market", choices=["crypto", "forex"], default="forex")
    parser.add_argument("--symbols", nargs="+", help="override symbols from config.toml")
    parser.add_argument("--strategy", choices=["ml", "sma"], default="ml")
    parser.add_argument("--start", default="2025-01-01", help="first candle to paper trade")
    parser.add_argument("--end", help="last candle to process (default: all data)")
    parser.add_argument("--account", help="account name (default: <market>_<strategy>)")
    parser.add_argument("--reset", action="store_true", help="delete the account and start over")
    args = parser.parse_args()

    cfg = load_config()
    symbols = sorted(args.symbols or cfg[args.market]["symbols"])
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    account_dir = ACCOUNTS_DIR / (args.account or f"{args.market}_{args.strategy}")
    if args.reset and account_dir.exists():
        shutil.rmtree(account_dir)

    account = PaperAccount(account_dir, cfg["paper"]["initial_cash"], Costs(**cfg["costs"][args.market]))
    meta = {"market": args.market, "symbols": symbols, "strategy": args.strategy, "start": args.start}
    if account.state.meta and account.state.meta != meta:
        raise SystemExit(
            f"Account {account_dir.name} was started with {account.state.meta}.\n"
            "Use the same settings to resume, --reset to start over, or --account <new name>."
        )
    account.state.meta = meta

    frames, signals = {}, {}
    for sym in symbols:
        df = load_ohlcv(data_path(args.market, sym, cfg[args.market]["timeframe"]))
        if args.strategy == "ml":
            signals[sym] = ml_signal(df, start, cfg["model"])
        else:
            signals[sym] = strategies.sma_crossover(df, allow_short=cfg["model"]["allow_short"])
        frames[sym] = df.loc[start:end] if end is not None else df.loc[start:]

    trader = PaperTrader(account, limits_from_config(cfg["paper"]))
    print(f"Paper trading {', '.join(symbols)} with the '{args.strategy}' strategy from {args.start} ...")
    try:
        count = replay(trader, frames, signals)
    finally:
        account.save()  # keep progress even if interrupted with Ctrl+C
    print(f"Processed {count:,} new candles.")
    print_summary(account)


if __name__ == "__main__":
    main()
