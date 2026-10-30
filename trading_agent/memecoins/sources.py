"""Newly created trading pools from GeckoTerminal (public API, no key)."""

from dataclasses import dataclass

import pandas as pd

GT_BASE = "https://api.geckoterminal.com/api/v2"


@dataclass
class Pool:
    network: str  # GeckoTerminal id: solana, base, bsc, eth
    pool_address: str
    token_address: str
    symbol: str
    name: str
    dex: str
    created_at: pd.Timestamp
    price_usd: float
    liquidity_usd: float
    fdv_usd: float
    volume_h1: float
    buys_h1: int
    sells_h1: int
    coingecko_id: str | None  # set for established tokens that CoinGecko already lists
    change_h1: float = float("nan")  # price change over the last hour, as a fraction (0.5 = +50%)
    change_h6: float = float("nan")


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_pools(payload: dict, network: str) -> list[Pool]:
    """Turn a GeckoTerminal pools response (with include=base_token) into Pool objects."""
    tokens = {t["id"]: t.get("attributes", {}) for t in payload.get("included", []) if t.get("type") == "token"}
    pools = []
    for item in payload.get("data", []):
        a = item.get("attributes", {})
        rel = item.get("relationships", {})
        token_id = rel.get("base_token", {}).get("data", {}).get("id", "")
        token = tokens.get(token_id, {})
        h1 = a.get("transactions", {}).get("h1", {}) or {}
        change = a.get("price_change_percentage") or {}
        pools.append(Pool(
            network=network,
            pool_address=a.get("address", ""),
            token_address=token.get("address") or token_id.split("_", 1)[-1],
            symbol=token.get("symbol") or a.get("name", "").split(" / ")[0],
            name=token.get("name") or a.get("name", ""),
            dex=rel.get("dex", {}).get("data", {}).get("id", ""),
            created_at=pd.Timestamp(a["pool_created_at"]) if a.get("pool_created_at") else pd.NaT,
            price_usd=_float(a.get("base_token_price_usd")),
            liquidity_usd=_float(a.get("reserve_in_usd")),
            fdv_usd=_float(a.get("fdv_usd")),
            volume_h1=_float((a.get("volume_usd") or {}).get("h1")),
            buys_h1=int(h1.get("buys") or 0),
            sells_h1=int(h1.get("sells") or 0),
            coingecko_id=token.get("coingecko_coin_id"),
            change_h1=_float(change.get("h1")) / 100,
            change_h6=_float(change.get("h6")) / 100,
        ))
    return pools


def fetch_new_pools(client, network: str, pages: int = 2) -> list[Pool]:
    pools = []
    for page in range(1, pages + 1):
        payload = client.get_json(f"{GT_BASE}/networks/{network}/new_pools?include=base_token&page={page}")
        if not payload:
            break
        pools.extend(parse_pools(payload, network))
    return pools


def fetch_trending_pools(client, network: str, pages: int = 1) -> list[Pool]:
    """Actively traded pools (often a few hours old): new_pools alone shows only minutes-old ones."""
    pools = []
    for page in range(1, pages + 1):
        payload = client.get_json(f"{GT_BASE}/networks/{network}/trending_pools?include=base_token&page={page}")
        if not payload:
            break
        pools.extend(parse_pools(payload, network))
    return pools


def fetch_pools(client, network: str, addresses: list[str]) -> dict[str, Pool]:
    """Current state of known pools, up to 30 per request. Missing pools are simply absent."""
    found = {}
    for i in range(0, len(addresses), 30):
        chunk = ",".join(addresses[i:i + 30])
        payload = client.get_json(f"{GT_BASE}/networks/{network}/pools/multi/{chunk}?include=base_token")
        for pool in parse_pools(payload or {}, network):
            found[pool.pool_address] = pool
    return found
