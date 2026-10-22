import numpy as np
import pandas as pd

from trading_agent.features import build_dataset
from trading_agent.market_data.synthetic import ohlcv_from_closes
from trading_agent.models.baseline import make_model, probabilities_to_signal, time_split


def zigzag_market(n=1500, seed=0):
    """Prices that tend to reverse every candle: a pattern a model should find."""
    rng = np.random.default_rng(seed)
    signs = np.where(np.arange(n) % 2 == 0, 1, -1)
    returns = signs * 0.01 + rng.normal(0, 0.002, n)
    return ohlcv_from_closes(100 * np.exp(np.cumsum(returns)), start_price=100)


def test_time_split_keeps_order_and_gap():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    X = pd.DataFrame({"a": range(100)}, index=idx)
    y = pd.Series(range(100), index=idx)
    X_train, X_test, y_train, y_test = time_split(X, y, test_fraction=0.3, gap=2)
    assert len(X_test) == 30 and len(X_train) == 68
    assert X_train.index.max() < X_test.index.min()


def test_model_learns_an_obvious_pattern_on_unseen_data():
    X, y = build_dataset(zigzag_market())
    X_train, X_test, y_train, y_test = time_split(X, y)
    model = make_model().fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.9


def test_probabilities_to_signal():
    proba = pd.Series([0.9, 0.5, 0.1])
    assert list(probabilities_to_signal(proba, 0.6)) == [1, 0, 0]
    assert list(probabilities_to_signal(proba, 0.6, allow_short=True)) == [1, 0, -1]
