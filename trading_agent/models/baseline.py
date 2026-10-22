"""A deliberately simple baseline model. Fancier models must beat this to be worth it."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KINDS = ("logistic", "gboost")


def make_model(kind: str = "logistic") -> Pipeline:
    """Build an untrained model.

    logistic: scaled logistic regression with strong regularisation (C=0.1) - the baseline.
    gboost:   small gradient-boosted trees; can find non-linear patterns, but overfits more easily,
              so it is kept shallow and slow-learning.
    """
    if kind == "logistic":
        clf = LogisticRegression(C=0.1, max_iter=1000)
    elif kind == "gboost":
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0
        )
    else:
        raise ValueError(f"Unknown model kind {kind!r}; choose from {MODEL_KINDS}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def time_split(X: pd.DataFrame, y: pd.Series, test_fraction: float = 0.3, gap: int = 2):
    """Split by time: train on the past, test on the future. Never shuffle market data.

    `gap` rows are thrown away between the two parts so training labels (which
    peek a few candles ahead) cannot overlap the test period.
    """
    cut = int(len(X) * (1 - test_fraction))
    train_end = max(cut - gap, 0)
    return X.iloc[:train_end], X.iloc[cut:], y.iloc[:train_end], y.iloc[cut:]


def probabilities_to_signal(
    proba_up: pd.Series, threshold: float = 0.55, allow_short: bool = False
) -> pd.Series:
    """Go long (1) when confident price rises; short (-1) when confident it falls; else flat (0)."""
    signal = pd.Series(0, index=proba_up.index, dtype=int)
    signal[proba_up >= threshold] = 1
    if allow_short:
        signal[proba_up <= 1 - threshold] = -1
    return signal
