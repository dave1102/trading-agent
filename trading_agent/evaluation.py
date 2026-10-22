"""Walk-forward evaluation: the honest way to test a trading model.

One train/test split can be lucky. Walk-forward repeats it through time:

    fold 1: train [=====]       test [--]
    fold 2: train [=======]     test   [--]
    fold 3: train [=========]   test     [--]

Each fold trains only on data before its test period (an expanding window),
so every test result is on data the model has never seen.
"""

import pandas as pd

from .backtest.engine import Costs, RiskRules, run_backtest
from .backtest.metrics import summarize
from .features import build_dataset
from .models.baseline import make_model, probabilities_to_signal
from .strategies import buy_and_hold


def walk_forward(
    df: pd.DataFrame,
    model_kind: str,
    model_cfg: dict,
    costs: Costs,
    risk: RiskRules,
    n_folds: int = 5,
    min_train_fraction: float = 0.4,
) -> pd.DataFrame:
    """Return one row per fold with test-period results for the model and buy-and-hold."""
    horizon = model_cfg["horizon"]
    gap = horizon + 1
    X, y = build_dataset(df, horizon)
    first_test = int(len(X) * min_train_fraction)
    fold_size = (len(X) - first_test) // n_folds
    if fold_size < 50:
        raise ValueError("Not enough data for this many folds")

    rows = []
    for k in range(n_folds):
        test_start = first_test + k * fold_size
        test_end = len(X) if k == n_folds - 1 else test_start + fold_size
        X_train, y_train = X.iloc[: test_start - gap], y.iloc[: test_start - gap]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        model = make_model(model_kind).fit(X_train, y_train)
        proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
        signal = probabilities_to_signal(proba, model_cfg["threshold"], model_cfg["allow_short"])

        period = df.loc[X_test.index[0] : X_test.index[-1]]
        model_stats = summarize(run_backtest(period, signal, costs, risk))
        market_stats = summarize(run_backtest(period, buy_and_hold(period), costs, RiskRules()))
        rows.append({
            "model": model_kind,
            "fold": k + 1,
            "test_start": X_test.index[0],
            "test_end": X_test.index[-1],
            "accuracy": model.score(X_test, y_test),
            "up_share": y_test.mean(),
            "return": model_stats["total_return"],
            "sharpe": model_stats["sharpe"],
            "max_drawdown": model_stats["max_drawdown"],
            "trades": model_stats["trades"],
            "buy_hold_return": market_stats["total_return"],
        })
    return pd.DataFrame(rows)


def compare_models(df, model_kinds, model_cfg, costs, risk, n_folds=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward for several models. Returns (all folds, averages per model)."""
    folds = pd.concat([walk_forward(df, kind, model_cfg, costs, risk, n_folds) for kind in model_kinds])
    summary = folds.groupby("model").agg(
        accuracy=("accuracy", "mean"),
        avg_return=("return", "mean"),
        folds_profitable=("return", lambda r: f"{(r > 0).sum()}/{len(r)}"),
        avg_sharpe=("sharpe", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        avg_trades=("trades", "mean"),
        buy_hold_return=("buy_hold_return", "mean"),
    )
    return folds.reset_index(drop=True), summary
