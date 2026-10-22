"""Portfolio strategies. Each returns target weights (dates x coins) using closes up to that day only.

The settings below are common textbook values, fixed BEFORE testing; they were
not tuned on this data. Changing them after looking at results would reintroduce
the overfitting this project tries to avoid.
"""

import numpy as np
import pandas as pd

TREND_LOOKBACKS = (20, 60, 120)  # days
MOMENTUM_LOOKBACK = 60  # days
MOMENTUM_TOP = 3  # coins held
VOL_LOOKBACK = 60  # days used to measure each coin's volatility
TARGET_VOL = 0.5  # yearly portfolio volatility aimed for when every coin is "on"
MAX_WEIGHT = 0.25  # at most 25% of the account in one coin


def realized_vol(close: pd.DataFrame, days: int = VOL_LOOKBACK) -> pd.DataFrame:
    """Yearly volatility from the last `days` daily log returns."""
    return np.log(close).diff().rolling(days).std() * np.sqrt(365)


def vol_sized(signal: pd.DataFrame, close: pd.DataFrame, slots: int | None = None,
              target_vol: float = TARGET_VOL, max_weight: float = MAX_WEIGHT) -> pd.DataFrame:
    """Turn 0..1 signals into weights: calmer coins get bigger positions, wilder ones smaller.

    Each coin that is fully "on" is sized to carry an equal share of `target_vol`,
    split over `slots` positions (default: every coin); single coins are capped at
    `max_weight` and the total at 100%.
    """
    per_coin = target_vol / (slots or close.shape[1])
    raw = (signal * per_coin / realized_vol(close)).clip(upper=max_weight)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = raw.sum(axis=1)
    scale = (1 / total).where(total > 1, 1.0)
    return raw.mul(scale, axis=0)


def trend_signal(close: pd.DataFrame, lookbacks=TREND_LOOKBACKS) -> pd.DataFrame:
    """Share of lookbacks over which the coin went up: 0, 1/3, 2/3 or 1."""
    votes = [(close / close.shift(days) - 1 > 0).astype(float) for days in lookbacks]
    signal = sum(votes) / len(votes)
    return signal.where(close.shift(max(lookbacks)).notna(), 0.0)  # no vote without enough history


def momentum_signal(close: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK, top: int = MOMENTUM_TOP) -> pd.DataFrame:
    """1 for the `top` coins with the best `lookback`-day return, if that return is positive."""
    past = close / close.shift(lookback) - 1
    rank = past.rank(axis=1, ascending=False)
    return ((rank <= top) & (past > 0)).astype(float)


def trend_weights(close: pd.DataFrame) -> pd.DataFrame:
    return vol_sized(trend_signal(close), close)


def momentum_weights(close: pd.DataFrame) -> pd.DataFrame:
    return vol_sized(momentum_signal(close), close, slots=MOMENTUM_TOP)


def equal_weights(close: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: the same share in every coin that has a price."""
    present = close.notna().astype(float)
    return present.div(present.sum(axis=1), axis=0).fillna(0.0)


def single_coin(close: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Benchmark: 100% in one coin."""
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights[symbol] = 1.0
    return weights


def weekly(index: pd.DatetimeIndex) -> pd.Series:
    """Rebalance on Sunday's close (traded at Monday's open)."""
    return pd.Series(index.day_of_week == 6, index=index)
