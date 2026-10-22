"""Does extra information make the model's confident trades better?

For each set of inputs, a model is trained per symbol on data before
--holdout-start. On the unseen period after it, we trade at several confidence
levels and pool the trades of all symbols. Judge by profit per trade and profit
factor, with enough trades to trust them - not by win rate alone.

Examples:
    python -m trading_agent.confidence
    python -m trading_agent.confidence --timeframe 1d --horizon 5
"""

import argparse

import pandas as pd

from .backtest.engine import Costs, RiskRules, run_backtest
from .config import load_config
from .evaluation import model_signal
from .experiment import risk_from_config
from .features import build_dataset
from .market_data import funding as funding_data
from .market_data.storage import RAW_DATA_DIR, data_path, load_ohlcv
from .market_data.validate import resample_ohlcv
from .models.baseline import make_model

FEATURE_SETS = {
    "price": {},
    "+market": {"calendar": True, "leader": True},
    "+market+funding": {"calendar": True, "leader": True, "funding": True},
}
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)


def make_context(spec: dict, leader: pd.DataFrame | None, funding: dict | None) -> dict:
    return {
        "calendar": spec.get("calendar", False),
        "leader": leader if spec.get("leader") else None,
        "funding": funding if spec.get("funding") else None,
    }


def pooled_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": float("nan"), "profit_per_trade": float("nan"), "profit_factor": float("nan")}
    wins = trades["pnl"][trades["pnl"] > 0].sum()
    losses = -trades["pnl"][trades["pnl"] <= 0].sum()
    return {
        "trades": len(trades),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "profit_per_trade": float(trades["return"].mean()),
        "profit_factor": wins / losses if losses > 0 else float("inf"),
    }


def confidence_table(
    frames: dict[str, pd.DataFrame],
    leader: pd.DataFrame | None,
    funding: dict | None,
    model_cfg: dict,
    costs: Costs,
    risk: RiskRules,
    holdout_start: pd.Timestamp,
    feature_sets: dict = FEATURE_SETS,
    thresholds=THRESHOLDS,
) -> pd.DataFrame:
    horizon = model_cfg["horizon"]
    rows = []
    for set_name, spec in feature_sets.items():
        context = make_context(spec, leader, funding)
        trades_by_threshold = {t: [] for t in thresholds}
        accuracies = []
        for df in frames.values():
            X, y = build_dataset(df, horizon, context)
            before = X.index < holdout_start
            X_train, y_train = X[before].iloc[: -(horizon + 1)], y[before].iloc[: -(horizon + 1)]
            X_test, y_test = X[~before], y[~before]
            model = make_model().fit(X_train, y_train)
            accuracies.append(model.score(X_test, y_test))
            period = df.loc[X_test.index[0]:]
            for t in thresholds:
                signal = model_signal(model, X_test, {**model_cfg, "threshold": t}, df, trend_filter=False)
                trades_by_threshold[t].append(run_backtest(period, signal, costs, risk).trades)
        for t in thresholds:
            rows.append({
                "inputs": set_name,
                "confidence": t,
                **pooled_stats(pd.concat(trades_by_threshold[t])),
                "accuracy": sum(accuracies) / len(accuracies),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare input sets by the quality of confident trades.")
    parser.add_argument("--timeframe", default="1h", help="candle size: 1h, 4h or 1d")
    parser.add_argument("--horizon", type=int, default=1, help="candles ahead the model predicts")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--leader", default="BTC/USDT", help="market leader used for +market inputs")
    args = parser.parse_args()

    cfg = load_config()
    tf = args.timeframe
    holdout_start = pd.Timestamp(args.holdout_start, tz="UTC")

    def load(symbol):
        df = load_ohlcv(data_path("crypto", symbol, cfg["crypto"]["timeframe"]))
        return df if tf == cfg["crypto"]["timeframe"] else resample_ohlcv(df, tf)

    frames = {s: load(s) for s in cfg["crypto"]["symbols"]}
    leader = frames.get(args.leader)
    if leader is None:
        leader = load(args.leader)
    fcfg = cfg["funding"]
    funding = {}
    for symbol in fcfg["symbols"]:
        path = funding_data.funding_path(fcfg["exchange"], symbol, RAW_DATA_DIR)
        if path.exists():
            funding[symbol.split("/")[0].lower()] = funding_data.load_funding(path)
    if not funding:
        print("No funding data found; run: python -m trading_agent.download funding")

    print(f"{', '.join(frames)} | {tf} candles, predicting {args.horizon} ahead | "
          f"trained before {holdout_start:%Y-%m-%d}, tested after (unseen)\n")
    table = confidence_table(
        frames, leader, funding or None, {**cfg["model"], "horizon": args.horizon},
        Costs(**cfg["costs"]["crypto"]), risk_from_config(cfg["risk"]), holdout_start,
    )
    shown = table.copy()
    for col in ("win_rate", "accuracy"):
        shown[col] = table[col].map("{:.1%}".format)
    shown["profit_per_trade"] = table["profit_per_trade"].map("{:+.2%}".format)
    shown["profit_factor"] = table["profit_factor"].map("{:.2f}".format)
    shown["confidence"] = table["confidence"].map("{:.0%}".format)
    print(shown.to_string(index=False))
    print("\nWorth trusting: profit_per_trade above 0 and profit_factor above ~1.3 with 100+ trades.")


if __name__ == "__main__":
    main()
