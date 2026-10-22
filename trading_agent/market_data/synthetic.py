"""Fake price data for tests and demos. Random walks have no real edge to find,
which makes them a good sanity check: a model that "wins" here is overfitting."""

import numpy as np
import pandas as pd

from .validate import OHLCV_COLUMNS, timeframe_to_seconds


def random_walk_ohlcv(
    n: int = 2000,
    start: str = "2024-01-01",
    timeframe: str = "1h",
    drift: float = 0.0,
    volatility: float = 0.01,
    start_price: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(drift, volatility, n)))
    return ohlcv_from_closes(close, start=start, timeframe=timeframe, start_price=start_price, seed=seed)


def ohlcv_from_closes(
    close, start: str = "2024-01-01", timeframe: str = "1h", start_price: float | None = None, seed: int = 0
) -> pd.DataFrame:
    """Build plausible candles around a given close series (each open = previous close)."""
    rng = np.random.default_rng(seed)
    close = np.asarray(close, dtype=float)
    open_ = np.r_[close[0] if start_price is None else start_price, close[:-1]]
    wiggle = np.abs(rng.normal(0, 0.002, len(close)))
    step = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + wiggle),
            "low": np.minimum(open_, close) * (1 - wiggle),
            "close": close,
            "volume": rng.uniform(100, 1000, len(close)),
        },
        index=pd.date_range(start, periods=len(close), freq=step, tz="UTC", name="timestamp"),
    )
    return df[OHLCV_COLUMNS]
