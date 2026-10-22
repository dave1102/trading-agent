import pytest

from trading_agent.backtest.engine import Costs, RiskRules
from trading_agent.evaluation import compare_models, walk_forward
from trading_agent.market_data.synthetic import random_walk_ohlcv
from trading_agent.models.baseline import make_model

MODEL_CFG = {"horizon": 1, "threshold": 0.55, "allow_short": False}


def test_make_model_kinds():
    assert make_model("logistic") is not None
    assert make_model("gboost") is not None
    with pytest.raises(ValueError):
        make_model("magic")


def test_walk_forward_folds_are_in_order_and_do_not_overlap():
    df = random_walk_ohlcv(n=2000, seed=5)
    folds = walk_forward(df, "logistic", MODEL_CFG, Costs(), RiskRules(), n_folds=4)
    assert list(folds["fold"]) == [1, 2, 3, 4]
    for prev, nxt in zip(folds.itertuples(), folds.iloc[1:].itertuples()):
        assert prev.test_end < nxt.test_start
    assert folds["test_end"].iloc[-1] >= df.index[-3]  # last fold reaches the end of the data


def test_compare_models_summary():
    df = random_walk_ohlcv(n=1500, seed=6)
    folds, summary = compare_models(df, ["logistic", "gboost"], MODEL_CFG, Costs(), RiskRules(), n_folds=3)
    assert len(folds) == 6
    assert set(summary.index) == {"logistic", "gboost"}
    # Random prices: no model should be meaningfully better than a coin flip.
    assert (summary["accuracy"] < 0.56).all()


def test_too_many_folds_is_refused():
    df = random_walk_ohlcv(n=300, seed=7)
    with pytest.raises(ValueError):
        walk_forward(df, "logistic", MODEL_CFG, Costs(), RiskRules(), n_folds=20)
