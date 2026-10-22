"""Turn OHLCV candles into model inputs (features) and labels (targets).

Golden rule: a feature for candle t may only use data up to and including
candle t's close. Anything later is "looking into the future" and makes
backtests lie.
"""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100): high = recent gains dominate, low = losses dominate."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = close.pct_change(n)
    f["volatility_24"] = np.log(close).diff().rolling(24).std()
    f["sma_20_gap"] = close / close.rolling(20).mean() - 1
    f["sma_50_gap"] = close / close.rolling(50).mean() - 1
    f["rsi_14"] = rsi(close) / 100 - 0.5
    f["bar_range"] = (df["high"] - df["low"]) / close

    # Forex volume is often all zeros; then this feature is just 0 instead of breaking.
    volume = df["volume"]
    volume_std = volume.rolling(24).std().replace(0, np.nan)
    f["volume_z"] = ((volume - volume.rolling(24).mean()) / volume_std).fillna(0)

    return f.replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """1.0 if price rises from the NEXT candle's open to the open `horizon` candles later.

    This matches how the backtest trades: decide at candle t's close, buy at t+1's open.
    The last rows have no future, so their target is NaN.
    """
    entry = df["open"].shift(-1)
    exit_ = df["open"].shift(-1 - horizon)
    future_return = exit_ / entry - 1
    return (future_return > 0).astype(float).where(future_return.notna())


def build_dataset(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Features and targets with incomplete rows (warm-up period, final rows) removed."""
    X = make_features(df)
    y = make_target(df, horizon)
    usable = X.notna().all(axis=1) & y.notna()
    return X[usable], y[usable]
