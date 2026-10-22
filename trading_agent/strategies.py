"""Simple rule-based strategies used as benchmarks for the ML model.

A signal is a Series of target positions: 1 = long, -1 = short, 0 = flat,
decided at each candle's close.
"""

import pandas as pd


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index, dtype=int)


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50, allow_short: bool = False) -> pd.Series:
    """Long when the fast moving average is above the slow one."""
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[fast_ma > slow_ma] = 1
    if allow_short:
        signal[fast_ma < slow_ma] = -1
    return signal
