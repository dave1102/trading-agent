"""Try many settings honestly: sweep on older data, confirm once on locked-away newer data.

Trying many settings and keeping the best one is itself a way to fool yourself:
some setting will look good by luck. So:
1. Everything from --holdout-start onwards is hidden from the sweep.
2. Each setting is walk-forward tested on the older data.
3. ONE setting is chosen: the best average across all symbols (not per symbol).
4. That setting is tested once on the hidden data. That result is the verdict.

Examples:
    python -m trading_agent.sweep --market crypto
    python -m trading_agent.sweep --market forex --holdout-start 2025-01-01
"""

import argparse
import itertools

import pandas as pd

from .backtest.engine import Costs, RiskRules
from .config import PROJECT_ROOT, load_config
from .evaluation import holdout_test, walk_forward
from .experiment import risk_from_config
from .market_data.storage import data_path, load_ohlcv
from .market_data.validate import resample_ohlcv

# (candle size, how many candles ahead the model predicts)
TIMEFRAME_HORIZONS = [("1h", 1), ("1h", 4), ("1h", 24), ("4h", 1), ("4h", 6), ("1d", 1), ("1d", 5)]
THRESHOLDS = [0.55, 0.60]
TREND_FILTER = [False, True]
DEFAULT_HOLDOUT = {"crypto": "2026-01-01", "forex": "2025-01-01"}
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def run_sweep(
    hourly: dict[str, pd.DataFrame],
    model_cfg: dict,
    costs: Costs,
    risk: RiskRules,
    holdout_start: pd.Timestamp,
    timeframe_horizons=TIMEFRAME_HORIZONS,
    thresholds=THRESHOLDS,
    trend_filters=TREND_FILTER,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (every setting x symbol, settings ranked by average, holdout results of the best)."""
    rows = []
    for symbol, df_1h in hourly.items():
        for timeframe, horizon in timeframe_horizons:
            df = df_1h if timeframe == "1h" else resample_ohlcv(df_1h, timeframe)
            research = df[df.index < holdout_start]
            for threshold, trend in itertools.product(thresholds, trend_filters):
                cfg = {**model_cfg, "horizon": horizon, "threshold": threshold}
                folds = walk_forward(research, "logistic", cfg, costs, risk, n_folds, trend_filter=trend)
                rows.append({
                    "symbol": symbol, "timeframe": timeframe, "horizon": horizon,
                    "threshold": threshold, "trend_filter": trend,
                    "avg_return": folds["return"].mean(),
                    "profitable_folds": int((folds["return"] > 0).sum()),
                    "avg_trades": folds["trades"].mean(),
                    "profit_per_trade": folds["profit_per_trade"].mean(),
                    "buy_hold_return": folds["buy_hold_return"].mean(),
                })
    results = pd.DataFrame(rows)

    setting = ["timeframe", "horizon", "threshold", "trend_filter"]
    ranked = (
        results.groupby(setting)
        .agg(avg_return=("avg_return", "mean"),
             profitable_folds=("profitable_folds", "sum"),
             avg_trades=("avg_trades", "mean"),
             profit_per_trade=("profit_per_trade", "mean"),
             buy_hold_return=("buy_hold_return", "mean"))
        .sort_values("avg_return", ascending=False)
        .reset_index()
    )
    ranked["profitable_folds"] = ranked["profitable_folds"].astype(str) + f"/{n_folds * len(hourly)}"

    best = ranked.iloc[0]
    holdout_rows = []
    for symbol, df_1h in hourly.items():
        df = df_1h if best["timeframe"] == "1h" else resample_ohlcv(df_1h, best["timeframe"])
        cfg = {**model_cfg, "horizon": int(best["horizon"]), "threshold": float(best["threshold"])}
        result = holdout_test(df, "logistic", cfg, costs, risk, holdout_start, bool(best["trend_filter"]))
        holdout_rows.append({"symbol": symbol, **result})
    return results, ranked, pd.DataFrame(holdout_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep settings, then confirm the best on hidden data.")
    parser.add_argument("--market", choices=["crypto", "forex"], default="crypto")
    parser.add_argument("--symbols", nargs="+", help="override symbols from config.toml")
    parser.add_argument("--holdout-start", help="hidden period starts here (default: crypto 2026-01-01, forex 2025-01-01)")
    args = parser.parse_args()

    cfg = load_config()
    holdout_start = pd.Timestamp(args.holdout_start or DEFAULT_HOLDOUT[args.market], tz="UTC")
    symbols = args.symbols or cfg[args.market]["symbols"]
    hourly = {s: load_ohlcv(data_path(args.market, s, cfg[args.market]["timeframe"])) for s in symbols}
    print(f"Sweeping {len(TIMEFRAME_HORIZONS) * len(THRESHOLDS) * len(TREND_FILTER)} settings on "
          f"{', '.join(symbols)} using data before {holdout_start:%Y-%m-%d} ...")

    results, ranked, holdout = run_sweep(
        hourly, cfg["model"], Costs(**cfg["costs"][args.market]), risk_from_config(cfg["risk"]), holdout_start
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"sweep_{args.market}.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print("\nTop 10 settings (average over all symbols, walk-forward on research data):")
        print(ranked.head(10).to_string(index=False))
        print("\nBottom 3 settings:")
        print(ranked.tail(3).to_string(index=False))
        best = ranked.iloc[0]
        print(f"\nVERDICT - best setting tested once on hidden data from {holdout_start:%Y-%m-%d}: "
              f"{best['timeframe']} candles, horizon {best['horizon']}, threshold {best['threshold']}, "
              f"trend filter {'on' if best['trend_filter'] else 'off'}")
        print(holdout[["symbol", "test_start", "test_end", "return", "buy_hold_return", "trades",
                       "win_rate", "profit_per_trade", "max_drawdown", "accuracy"]].to_string(index=False))
    print(f"\nAll results saved to {RESULTS_DIR / f'sweep_{args.market}.csv'}")


if __name__ == "__main__":
    main()
