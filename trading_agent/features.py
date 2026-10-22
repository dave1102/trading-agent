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


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
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


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour of day and day of week as points on a circle, so 23:00 sits next to 00:00."""
    hour = index.hour + index.minute / 60
    day = index.day_of_week
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * day / 7),
            "dow_cos": np.cos(2 * np.pi * day / 7),
        },
        index=index,
    )


def make_market_features(df: pd.DataFrame, leader: pd.DataFrame) -> pd.DataFrame:
    """What the market leader (BTC) is doing, and how this symbol compares with it.

    `leader` must have the same candle size as `df`; candles are matched by time.
    """
    lead_close = leader["close"].reindex(df.index)
    f = pd.DataFrame(index=df.index)
    f["lead_ret_1"] = lead_close.pct_change(1)
    f["lead_ret_24"] = lead_close.pct_change(24)
    f["lead_volatility_24"] = np.log(lead_close).diff().rolling(24).std()
    f["rel_ret_24"] = df["close"].pct_change(24) - f["lead_ret_24"]
    return f


def make_funding_features(index: pd.DatetimeIndex, funding: dict[str, pd.Series]) -> pd.DataFrame:
    """Recent funding rates, using only rates stamped within an hour of the candle's open.

    Those are all known before the candle closes, which is when the bot decides.
    """
    f = pd.DataFrame(index=index)
    for name, rates in funding.items():
        rates = rates.sort_index()
        hourly = rates.resample("1h").mean().ffill()
        for window in ("8h", "72h"):
            averaged = hourly.rolling(window).mean()
            f[f"funding_{name}_{window}"] = averaged.reindex(index, method="ffill")
    return f * 10_000  # in basis points (0.01%), a friendlier scale


def make_features(df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
    """Price features, plus optional extra information in `context`:

    - "calendar": True to add hour-of-day / day-of-week
    - "leader": OHLCV of the market leader (e.g. BTC) at the same candle size
    - "funding": {name: funding-rate Series}
    """
    f = make_price_features(df)
    context = context or {}
    extra = []
    if context.get("calendar"):
        extra.append(make_calendar_features(df.index))
    if context.get("leader") is not None:
        extra.append(make_market_features(df, context["leader"]))
    if context.get("funding"):
        extra.append(make_funding_features(df.index, context["funding"]))
    if extra:
        f = pd.concat([f, *extra], axis=1)
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


def build_dataset(
    df: pd.DataFrame, horizon: int = 1, context: dict | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and targets with incomplete rows (warm-up period, final rows) removed."""
    X = make_features(df, context)
    y = make_target(df, horizon)
    usable = X.notna().all(axis=1) & y.notna()
    return X[usable], y[usable]
