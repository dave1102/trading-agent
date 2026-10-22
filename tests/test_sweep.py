import numpy as np

from trading_agent.backtest.engine import Costs, RiskRules
from trading_agent.evaluation import holdout_test, trend_mask, walk_forward
from trading_agent.market_data.synthetic import ohlcv_from_closes, random_walk_ohlcv
from trading_agent.sweep import run_sweep

MODEL_CFG = {"horizon": 1, "threshold": 0.5, "allow_short": False}


def test_trend_mask_uses_only_finished_days():
    df = random_walk_ohlcv(n=24 * 300, seed=8)
    changed = df.copy()
    last_day = df.index[-1].floor("1D")
    changed.loc[changed.index >= last_day, "close"] *= 10  # change only the final day
    a, b = trend_mask(df), trend_mask(changed)
    assert a.equals(b)  # today's prices cannot affect today's filter


def test_trend_mask_detects_up_and_down_trends():
    up = ohlcv_from_closes([100 * 1.001 ** i for i in range(24 * 300)], timeframe="1h")
    down = ohlcv_from_closes([100 * 0.999 ** i for i in range(24 * 300)], timeframe="1h")
    assert trend_mask(up).iloc[-24:].all()
    assert not trend_mask(down).iloc[-24:].any()
    assert not trend_mask(up).iloc[:24 * 100].any()  # not enough history yet


def test_trend_filter_blocks_longs_in_downtrend():
    n = 24 * 400
    noise = 1 + np.random.default_rng(12).normal(0, 0.002, n)  # some up-hours, but a clear downtrend
    down = ohlcv_from_closes(100 * 0.9997 ** np.arange(n) * noise, timeframe="1h")
    folds = walk_forward(down, "logistic", MODEL_CFG, Costs(), RiskRules(), n_folds=3, trend_filter=True)
    assert (folds["trades"] == 0).all()


def test_holdout_trains_only_on_the_past():
    df = random_walk_ohlcv(n=3000, seed=9)
    start = df.index[2000]
    result = holdout_test(df, "logistic", MODEL_CFG, Costs(), RiskRules(), start)
    assert result["train_end"] < start <= result["test_start"]


def test_run_sweep_small_grid():
    hourly = {"AAA": random_walk_ohlcv(n=24 * 200, seed=10), "BBB": random_walk_ohlcv(n=24 * 200, seed=11)}
    holdout = hourly["AAA"].index[24 * 160]
    results, ranked, best_holdout = run_sweep(
        hourly, MODEL_CFG, Costs(), RiskRules(), holdout,
        timeframe_horizons=[("1h", 1), ("4h", 1)], thresholds=[0.55], trend_filters=[False], n_folds=3,
    )
    assert len(results) == 4  # 2 symbols x 2 settings
    assert len(ranked) == 2
    assert ranked["avg_return"].is_monotonic_decreasing
    assert list(best_holdout["symbol"]) == ["AAA", "BBB"]
    assert (best_holdout["test_start"] >= holdout).all()
