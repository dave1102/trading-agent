"""Save and load OHLCV data as CSV files under data/raw/<market>/."""

from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from .validate import clean_ohlcv

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def data_path(market: str, symbol: str, timeframe: str, base_dir: Path = RAW_DATA_DIR) -> Path:
    """e.g. ("crypto", "BTC/USDT", "1h") -> data/raw/crypto/BTC-USDT_1h.csv"""
    safe_symbol = symbol.replace("/", "-").replace(":", "-").upper()
    return Path(base_dir) / market / f"{safe_symbol}_{timeframe}.csv"


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp")
    return clean_ohlcv(df)


def save_ohlcv(df: pd.DataFrame, path: Path, merge: bool = True) -> pd.DataFrame:
    """Write candles to `path`. With merge=True, new candles are added to the existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.exists():
        df = pd.concat([load_ohlcv(path), df])
    df = clean_ohlcv(df)
    df.to_csv(path)
    return df
