"""The scanner's journal: every token it looked at, and what happened to the ones it tracks.

    candidates.csv   one row per pool, the first time it is seen
    checks.csv       one row per (tracked pool, checkpoint), e.g. price after 1h / 24h / 7d

Tracked groups:
    pick     passed every rule
    control  a random sample of pools that failed, for comparison ("do the rules help?")
"""

import csv
from pathlib import Path

import pandas as pd

CANDIDATE_COLUMNS = [
    "seen_at", "network", "pool_address", "token_address", "symbol", "name", "dex", "pool_created_at",
    "price_usd", "liquidity_usd", "fdv_usd", "volume_h1", "buys_h1", "sells_h1",
    "passed", "group", "sell_tax", "reasons",
]
CHECK_COLUMNS = [
    "network", "pool_address", "symbol", "group", "checkpoint", "checked_at",
    "price_usd", "liquidity_usd", "gross_return", "net_return", "rugged",
]
CHECKPOINTS = {"1h": pd.Timedelta(hours=1), "24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7)}
RUG_LIQUIDITY_DROP = 0.9  # liquidity down 90%+ from entry = treat as a rug pull (total loss)


def safe_text(value) -> str:
    """Stop text from the internet being run as a formula when the CSV is opened in Excel."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t") else text


def append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: safe_text(row.get(c, "")) if isinstance(row.get(c), str) else row.get(c, "")
                             for c in columns})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"pool_address": str, "token_address": str}) if path.exists() else pd.DataFrame()


def side_cost(liquidity_usd: float, trade_usd: float, cfg: dict, network: str, tax: float = 0.0) -> float:
    """Cost of one buy or one sell, as a fraction of the trade.

    DEX fee + price impact (in a pool holding `liquidity_usd`, buying `trade_usd`
    moves the price by roughly 2 x trade / liquidity) + token tax + network gas.
    MEV bots ("sandwich attacks") can cost more; this is an optimistic estimate.
    """
    if not liquidity_usd > 0:
        return 1.0
    impact = min(2 * trade_usd / liquidity_usd, 1.0)
    gas = cfg["gas_usd"].get(network, 1.0) / trade_usd
    return min(cfg["dex_fee"] + impact + tax + gas, 1.0)


def net_return(entry: dict, price_usd: float, liquidity_usd: float, cfg: dict) -> tuple[float, float, bool]:
    """(gross, net after costs, rugged) for a paper trade of cfg['trade_usd'] bought at `entry`."""
    entry_liq = float(entry["liquidity_usd"])
    entry_price = float(entry["price_usd"])
    if not entry_price > 0:
        return float("nan"), float("nan"), False  # no usable entry price: leave out of the results
    rugged = (not liquidity_usd > 0) or liquidity_usd < (1 - RUG_LIQUIDITY_DROP) * entry_liq or not price_usd > 0
    if rugged:
        return -1.0, -1.0, True
    gross = price_usd / entry_price - 1
    tax = pd.to_numeric(entry.get("sell_tax"), errors="coerce")
    tax = 0.0 if pd.isna(tax) else float(tax)
    buy = side_cost(entry_liq, cfg["trade_usd"], cfg, entry["network"], tax)
    sell = side_cost(liquidity_usd, cfg["trade_usd"], cfg, entry["network"], tax)
    return gross, (1 + gross) * (1 - buy) * (1 - sell) - 1, False


def due_checks(candidates: pd.DataFrame, checks: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Tracked candidates with a checkpoint that is due and not yet recorded."""
    if candidates.empty:
        return pd.DataFrame()
    tracked = candidates[candidates["group"].isin(["pick", "control"])]
    done = set() if checks.empty else set(zip(checks["pool_address"], checks["checkpoint"]))
    rows = []
    for _, c in tracked.iterrows():
        seen = pd.Timestamp(c["seen_at"])
        for name, delay in CHECKPOINTS.items():
            if seen + delay <= now and (c["pool_address"], name) not in done:
                rows.append({**c.to_dict(), "checkpoint": name})
    return pd.DataFrame(rows)


def verdict_lines(table: pd.DataFrame, min_tokens: int = 100) -> list[str]:
    """Plain-English reading of the summary table, one line per checkpoint."""
    lines = []
    for checkpoint in CHECKPOINTS:
        rows = table[table["checkpoint"] == checkpoint].set_index("group") if not table.empty else table
        if "pick" not in rows.index:
            lines.append(f"{checkpoint}: no results yet.")
            continue
        pick = rows.loc["pick"]
        if pick["tokens"] < min_tokens:
            lines.append(f"{checkpoint}: only {int(pick['tokens'])} picks so far - too early to judge (need {min_tokens}+).")
            continue
        money = "MADE money" if pick["mean_net"] > 0 else "LOST money"
        line = (f"{checkpoint}: picks {money} on average ({pick['mean_net']:+.1%} per trade after costs; "
                f"{pick['positive']:.0%} ended up, {pick['rugged']:.0%} rugged).")
        if "control" in rows.index:
            control = rows.loc["control"]
            better = "the rules HELPED" if pick["mean_net"] > control["mean_net"] else "the rules did NOT help"
            line += f" Comparison group: {control['mean_net']:+.1%}, so {better}."
        lines.append(line)
    return lines


def summarize(checks: pd.DataFrame) -> pd.DataFrame:
    """Per group and checkpoint: how many, median/mean result after costs, share positive, share rugged."""
    if checks.empty:
        return pd.DataFrame()
    order = {name: i for i, name in enumerate(CHECKPOINTS)}
    table = checks.groupby(["checkpoint", "group"]).agg(
        tokens=("net_return", "size"),
        median_net=("net_return", "median"),
        mean_net=("net_return", "mean"),
        positive=("net_return", lambda r: (r > 0).mean()),
        rugged=("rugged", "mean"),
    ).reset_index()
    return table.sort_values(["checkpoint", "group"], key=lambda s: s.map(order) if s.name == "checkpoint" else s)
