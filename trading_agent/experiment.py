"""Train the baseline model and compare it with simple strategies on unseen data.

Examples:
    python -m trading_agent.experiment --demo
    python -m trading_agent.experiment --market crypto --symbol BTC/USDT
    python -m trading_agent.experiment --market forex --symbol EURUSD
"""

import argparse

import pandas as pd

from . import strategies
from .backtest.engine import Costs, RiskRules, run_backtest
from .backtest.metrics import summarize
from .config import load_config
from .features import build_dataset
from .market_data.storage import data_path, load_ohlcv
from .market_data.synthetic import random_walk_ohlcv
from .models.baseline import make_model, probabilities_to_signal, time_split


def risk_from_config(cfg: dict) -> RiskRules:
    """Zero in config.toml means "rule switched off"."""
    return RiskRules(
        position_fraction=cfg["position_fraction"],
        stop_loss=cfg["stop_loss"] or None,
        take_profit=cfg["take_profit"] or None,
        max_drawdown_halt=cfg["max_drawdown_halt"] or None,
    )


def run_experiment(df: pd.DataFrame, costs: Costs, risk: RiskRules, model_cfg: dict) -> pd.DataFrame:
    X, y = build_dataset(df, model_cfg["horizon"])
    X_train, X_test, y_train, y_test = time_split(
        X, y, model_cfg["test_fraction"], gap=model_cfg["horizon"] + 1
    )
    model = make_model().fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Train: {X_train.index[0]:%Y-%m-%d} to {X_train.index[-1]:%Y-%m-%d} ({len(X_train):,} candles)")
    print(f"Test:  {X_test.index[0]:%Y-%m-%d} to {X_test.index[-1]:%Y-%m-%d} ({len(X_test):,} candles, unseen)")
    print(f"Model accuracy on test: {accuracy:.1%}  (share of up-moves: {y_test.mean():.1%})")

    test_df = df.loc[X_test.index[0]:]
    proba_up = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    signals = {
        "ml_baseline": probabilities_to_signal(proba_up, model_cfg["threshold"], model_cfg["allow_short"]),
        "sma_crossover": strategies.sma_crossover(df, allow_short=model_cfg["allow_short"]),
        "buy_and_hold": strategies.buy_and_hold(df),
    }
    rows = {}
    for name, signal in signals.items():
        # Buy-and-hold is the market itself, so it ignores stops and position sizing.
        rules = RiskRules() if name == "buy_and_hold" else risk
        rows[name] = summarize(run_backtest(test_df, signal, costs, rules))
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and backtest on unseen data.")
    parser.add_argument("--demo", action="store_true", help="use random synthetic prices")
    parser.add_argument("--market", choices=["crypto", "forex"], default="crypto")
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()

    cfg = load_config()
    if args.demo:
        print("DEMO: random-walk prices. There is no real pattern, so expect no edge.\n")
        df = random_walk_ohlcv(n=8000, seed=42)
    else:
        df = load_ohlcv(data_path(args.market, args.symbol, cfg[args.market]["timeframe"]))
    costs = Costs(**cfg["costs"][args.market])

    table = run_experiment(df, costs, risk_from_config(cfg["risk"]), cfg["model"])
    pct = ["total_return", "cagr", "max_drawdown", "win_rate", "exposure"]
    shown = table.copy().astype(object)
    for col in table.columns:
        fmt = "{:.1%}" if col in pct else "{:,.0f}" if col in ("trades", "fees_paid") else "{:.2f}"
        shown[col] = table[col].map(fmt.format)
    print("\n" + shown.to_string())


if __name__ == "__main__":
    main()
