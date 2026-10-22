"""Command line for walk-forward model comparison.

Examples:
    python -m trading_agent.evaluate --market forex --symbol EURUSD
    python -m trading_agent.evaluate --market crypto --symbol BTC/USDT --models logistic gboost --folds 6
"""

import argparse

import pandas as pd

from .backtest.engine import Costs
from .config import load_config
from .evaluation import compare_models
from .experiment import risk_from_config
from .market_data.storage import data_path, load_ohlcv
from .models.baseline import MODEL_KINDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward comparison of models on unseen data.")
    parser.add_argument("--market", choices=["crypto", "forex"], default="forex")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--models", nargs="+", choices=MODEL_KINDS, default=list(MODEL_KINDS))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    df = load_ohlcv(data_path(args.market, args.symbol, cfg[args.market]["timeframe"]))
    folds, summary = compare_models(
        df, args.models, cfg["model"], Costs(**cfg["costs"][args.market]),
        risk_from_config(cfg["risk"]), args.folds,
    )
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(folds.to_string(index=False))
        print("\nAverages per model:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
