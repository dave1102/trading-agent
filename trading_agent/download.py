"""Command-line tool for collecting historical data.

Examples:
    python -m trading_agent.download crypto
    python -m trading_agent.download crypto --symbols BTC/USDT ETH/USDT --start 2024-01-01
    python -m trading_agent.download forex --symbol EURUSD --file downloads/DAT_ASCII_EURUSD_M1_2024.csv
    python -m trading_agent.download histdata --years 2019-2025
"""

import argparse
from pathlib import Path

import ccxt
import pandas as pd
import requests

from .config import load_config
from .market_data import crypto, forex, histdata
from .market_data.storage import data_path, load_ohlcv, save_ohlcv
from .market_data.validate import find_gaps, resample_ohlcv


def summarize(name: str, df, timeframe: str) -> None:
    if df.empty:
        print(f"{name}: no data")
        return
    gaps = find_gaps(df, timeframe)
    print(
        f"{name}: {len(df):,} candles from {df.index[0]} to {df.index[-1]}, "
        f"{len(gaps)} gaps ({gaps['missing_bars'].sum():,} missing candles)"
    )


def download_crypto(cfg: dict, symbols: list[str] | None, start: str | None) -> None:
    exchange = crypto.make_exchange(cfg["exchange"])
    timeframe = cfg["timeframe"]
    for symbol in symbols or cfg["symbols"]:
        path = data_path("crypto", symbol, timeframe)
        since = start or cfg["start"]
        if path.exists() and start is None:
            since = load_ohlcv(path).index[-1]  # resume where the last download stopped
        print(f"Downloading {symbol} {timeframe} from {since} ...")
        try:
            df = crypto.fetch_ohlcv(symbol, timeframe, since=since, exchange=exchange)
        except ccxt.NetworkError as e:
            print(f"  could not reach {cfg['exchange']}: {type(e).__name__}")
            print("  Check your internet/DNS, or try another exchange in config.toml.")
            continue
        summarize(symbol, save_ohlcv(df, path), timeframe)
        print(f"  saved to {path}")


def import_forex(cfg: dict, symbol: str, file: Path, fmt: str) -> None:
    timeframe = cfg["timeframe"]
    df = forex.load_histdata_csv(file) if fmt == "histdata" else forex.load_generic_csv(file)
    df = resample_ohlcv(df, timeframe)
    path = data_path("forex", symbol, timeframe)
    summarize(symbol, save_ohlcv(df, path), timeframe)
    print(f"  saved to {path} (weekend gaps are normal for forex)")


def download_histdata(cfg: dict, symbols: list[str] | None, years: list[int]) -> None:
    timeframe = cfg["timeframe"]
    for symbol in symbols or cfg["symbols"]:
        frames = []
        for year in years:
            try:
                csv_path = histdata.download_year(symbol, year)
            except (RuntimeError, requests.RequestException) as e:
                print(f"  {symbol} {year}: skipped ({e})")
                continue
            frames.append(resample_ohlcv(forex.load_histdata_csv(csv_path), timeframe))
            print(f"  {symbol} {year}: ok")
        if frames:
            path = data_path("forex", symbol, timeframe)
            summarize(symbol, save_ohlcv(pd.concat(frames), path), timeframe)
            print(f"  saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical market data.")
    sub = parser.add_subparsers(dest="market", required=True)

    p_crypto = sub.add_parser("crypto", help="download from a public exchange via CCXT")
    p_crypto.add_argument("--symbols", nargs="+", help="override symbols from config.toml")
    p_crypto.add_argument("--start", help="start date, e.g. 2024-01-01 (re-downloads from there)")

    p_fx = sub.add_parser("forex", help="import a CSV file from a forex data provider")
    p_fx.add_argument("--symbol", required=True, help="e.g. EURUSD")
    p_fx.add_argument("--file", required=True, type=Path)
    p_fx.add_argument("--format", choices=["histdata", "generic"], default="histdata")

    p_hd = sub.add_parser("histdata", help="download forex M1 files from HistData.com and import them")
    p_hd.add_argument("--symbols", nargs="+", help="override forex symbols from config.toml")
    p_hd.add_argument("--years", required=True, help="e.g. 2019-2025 or 2020,2022")

    args = parser.parse_args()
    cfg = load_config()
    if args.market == "crypto":
        download_crypto(cfg["crypto"], args.symbols, args.start)
    elif args.market == "histdata":
        download_histdata(cfg["forex"], args.symbols, histdata.parse_years(args.years))
    else:
        import_forex(cfg["forex"], args.symbol, args.file, args.format)


if __name__ == "__main__":
    main()
