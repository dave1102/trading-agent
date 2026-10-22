"""Catch copycat tokens that borrow the ticker of a famous coin (a common scam).

The tickers of CoinGecko's top 250 coins are downloaded at most once a day and
cached. A brand-new token using one of them is almost never the real coin.
"""

import json
from pathlib import Path

import pandas as pd
import requests

COINGECKO_TOP = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"


def load_top_symbols(client, cache_path: Path, now: pd.Timestamp, max_age=pd.Timedelta(days=1)) -> set[str]:
    cache_path = Path(cache_path)
    cached = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if now - pd.Timestamp(cached["fetched_at"]) < max_age:
            return set(cached["symbols"])
    try:
        payload = client.get_json(COINGECKO_TOP)
    except requests.RequestException:
        payload = None
    if not isinstance(payload, list):
        return set(cached["symbols"]) if cached else set()  # keep using yesterday's list if needed
    symbols = sorted({str(c.get("symbol", "")).lower() for c in payload if c.get("symbol")})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"fetched_at": now.isoformat(), "symbols": symbols}), encoding="utf-8")
    return set(symbols)
