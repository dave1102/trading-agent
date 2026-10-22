import numpy as np
import pandas as pd

from trading_agent.features import build_dataset, make_features, make_target
from trading_agent.market_data.synthetic import ohlcv_from_closes, random_walk_ohlcv


def test_features_never_look_into_the_future():
    df = random_walk_ohlcv(n=300, seed=1)
    cutoff = 200
    changed = df.copy()
    changed.iloc[cutoff + 1:] *= 3  # wildly change every candle after the cutoff

    original = make_features(df).iloc[: cutoff + 1]
    after_change = make_features(changed).iloc[: cutoff + 1]
    pd.testing.assert_frame_equal(original, after_change)


def test_target_uses_next_open_to_following_open():
    df = ohlcv_from_closes([10, 11, 12, 11, 13], start_price=10)
    # opens are [10, 10, 11, 12, 11]
    target = make_target(df, horizon=1)
    assert target.iloc[0] == 1.0  # open[1]=10 -> open[2]=11 rises
    assert target.iloc[2] == 0.0  # open[3]=12 -> open[4]=11 falls
    assert target.iloc[-2:].isna().all()  # no future left


def test_build_dataset_has_no_missing_values():
    df = random_walk_ohlcv(n=300, seed=2)
    X, y = build_dataset(df)
    assert len(X) == len(y) > 200
    assert not X.isna().any().any()
    assert X.index.equals(y.index)


def test_zero_volume_does_not_break_features():
    df = random_walk_ohlcv(n=200, seed=3)
    df["volume"] = 0.0  # typical for forex
    X, _ = build_dataset(df)
    assert len(X) > 100
    assert np.isfinite(X.to_numpy()).all()
