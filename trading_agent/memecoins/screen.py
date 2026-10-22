"""Rules that decide which new tokens are worth a (paper) look. Every failed rule is reported."""

import pandas as pd

from .security import SecurityReport
from .sources import Pool


def market_reasons(pool: Pool, cfg: dict, now: pd.Timestamp, top_symbols: set[str] = frozenset()) -> list[str]:
    """Cheap checks from pool data alone (no extra API calls)."""
    reasons = []
    if not pool.coingecko_id and pool.symbol.strip().lower() in top_symbols:
        reasons.append("copies the ticker of a top-250 coin")
    age_min = (now - pool.created_at).total_seconds() / 60 if pd.notna(pool.created_at) else float("nan")
    if not age_min >= cfg["min_age_minutes"]:
        reasons.append("too new")
    if not age_min <= cfg["max_age_hours"] * 60:
        reasons.append("too old")
    if pool.coingecko_id:
        reasons.append("established token (already on CoinGecko)")
    if not pool.liquidity_usd >= cfg["min_liquidity_usd"]:
        reasons.append("low liquidity")
    if pool.buys_h1 + pool.sells_h1 < cfg["min_trades_h1"]:
        reasons.append("little trading")
    if pool.sells_h1 == 0:
        reasons.append("nobody has sold (possible honeypot)")
    return reasons


def security_reasons(report: SecurityReport, cfg: dict) -> list[str]:
    """Scam checks. Anything unknown counts as a failure."""
    if not report.found:
        return ["security data unavailable"]
    reasons = list(report.dangers)
    if report.honeypot is not False:
        reasons.append("honeypot" if report.honeypot else "honeypot check unknown")
    for name, value, limit in (("buy tax", report.buy_tax, cfg["max_tax"]), ("sell tax", report.sell_tax, cfg["max_tax"])):
        if value is None:
            reasons.append(f"{name} unknown")
        elif value > limit:
            reasons.append(f"{name} {value:.0%}")
    if report.mintable is not False:
        reasons.append("creator can mint more" if report.mintable else "mint authority unknown")
    if report.freezable is not False:
        reasons.append("creator can freeze tokens" if report.freezable else "freeze authority unknown")
    if report.top10_share is not None and report.top10_share > cfg["max_top10_share"]:
        reasons.append(f"top 10 wallets hold {report.top10_share:.0%}")
    if report.lp_locked_share is None:
        reasons.append("liquidity lock unknown")
    elif report.lp_locked_share < cfg["min_lp_locked"]:
        reasons.append(f"only {report.lp_locked_share:.0%} of liquidity locked")
    return list(dict.fromkeys(reasons))  # drop duplicates, keep order
