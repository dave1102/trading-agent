"""The scanner's journal: every entry it judged, and what happened to the tracked ones.

    candidates.csv   one row per ENTRY = (strategy, pool), the first time it is judged
    checks.csv       one row per (tracked entry, checkpoint), e.g. price after 5m ... 7d

Each strategy (trending, early, not_pumped) is an independent paper trader with its
own entry time and price. Tracked groups:
    pick     a strategy's entry that passed every rule
    control  a random sample of all listed pools, for comparison ("do the rules help?")
"""

import csv
from pathlib import Path

import pandas as pd

CANDIDATE_COLUMNS = [
    "entry_id", "strategy", "seen_at", "network", "pool_address", "token_address", "symbol", "name", "dex",
    "pool_created_at", "price_usd", "liquidity_usd", "fdv_usd", "volume_h1", "buys_h1", "sells_h1",
    "change_h1", "change_h6", "passed", "group", "sell_tax", "reasons",
]
CHECK_COLUMNS = [
    "entry_id", "strategy", "network", "pool_address", "symbol", "group", "checkpoint", "checked_at",
    "price_usd", "liquidity_usd", "gross_return", "net_return", "rugged",
]
CHECKPOINTS = {
    "5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1), "24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7),
}
QUICK_EXIT_CHECKPOINTS = ["5m", "15m", "30m", "1h"]  # the last one is the forced exit
RUG_LIQUIDITY_DROP = 0.9  # liquidity down 90%+ from entry = treat as a rug pull (total loss)
# A price up more than 100x is almost always a broken reading (e.g. a pool's first price),
# not a real trade result; such checks are left out of the summaries.
SUSPECT_GROSS = 99.0


def entry_id(strategy: str, pool_address: str) -> str:
    return f"{strategy}:{pool_address}"


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
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"pool_address": str, "token_address": str, "entry_id": str})


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


def label(row) -> str:
    """Results are reported per strategy for picks, and as 'control' for the comparison group."""
    return row["strategy"] if row["group"] == "pick" else "control"


def due_checks(candidates: pd.DataFrame, checks: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Tracked entries with a checkpoint that is due and not yet recorded."""
    if candidates.empty:
        return pd.DataFrame()
    tracked = candidates[candidates["group"].isin(["pick", "control"])]
    done = set() if checks.empty else set(zip(checks["entry_id"], checks["checkpoint"]))
    rows = []
    for _, c in tracked.iterrows():
        seen = pd.Timestamp(c["seen_at"])
        for name, delay in CHECKPOINTS.items():
            if seen + delay <= now and (c["entry_id"], name) not in done:
                rows.append({**c.to_dict(), "checkpoint": name})
    return pd.DataFrame(rows)


def clean(checks: pd.DataFrame) -> pd.DataFrame:
    """Drop broken price readings (impossible jumps)."""
    if checks.empty or "gross_return" not in checks.columns:
        return checks
    return checks[~(checks["gross_return"] > SUSPECT_GROSS)]


def add_liquid_control(checks: pd.DataFrame, candidates: pd.DataFrame, min_liquidity: float) -> pd.DataFrame:
    """Add 'control_liquid' rows: comparison pools at least as liquid as the strictest picks.

    The full comparison group is mostly tiny, inactive pools, which mainly lose to costs;
    pools as liquid as the picks are the fair comparison.
    """
    if checks.empty or candidates.empty:
        return checks
    liquidity = candidates.set_index("entry_id")["liquidity_usd"]
    liquid = checks[(checks["group"] == "control") & (checks["entry_id"].map(liquidity) >= min_liquidity)]
    return pd.concat([checks, liquid.assign(group="control_liquid")], ignore_index=True)


def quick_exit(checks: pd.DataFrame, take_profit: float, stop_loss: float) -> pd.DataFrame:
    """Simulate 'sell at +take_profit or -stop_loss, else at 1 hour' for every tracked entry.

    Uses only the 5m / 15m / 30m / 1h checks, so prices between checks are unseen: an
    approximation (real exits would sometimes be better or worse). Entries without a
    complete set of short checks up to their exit are left out.
    """
    if checks.empty:
        return pd.DataFrame()
    short = checks[checks["checkpoint"].isin(QUICK_EXIT_CHECKPOINTS)]
    order = {name: i for i, name in enumerate(QUICK_EXIT_CHECKPOINTS)}
    rows = []
    for eid, entry in short.groupby("entry_id"):
        entry = entry.assign(_order=entry["checkpoint"].map(order)).sort_values("_order")
        if list(entry["checkpoint"]) != QUICK_EXIT_CHECKPOINTS[: len(entry)]:
            continue  # a short check is missing
        exit_row = None
        for _, check in entry.iterrows():
            if check["rugged"] or check["gross_return"] >= take_profit or check["gross_return"] <= -stop_loss \
                    or check["checkpoint"] == QUICK_EXIT_CHECKPOINTS[-1]:
                exit_row = check
                break
        if exit_row is not None:
            rows.append({**exit_row.drop(labels="_order").to_dict(), "checkpoint": "quick_exit",
                         "exit_at": exit_row["checkpoint"]})
    return pd.DataFrame(rows)


def summarize(checks: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint and group: how many, median/mean result after costs, share positive, share rugged."""
    checks = clean(checks)
    if checks.empty:
        return pd.DataFrame()
    order = {name: i for i, name in enumerate([*CHECKPOINTS, "quick_exit"])}
    table = checks.groupby(["checkpoint", "group"]).agg(
        tokens=("net_return", "size"),
        median_net=("net_return", "median"),
        mean_net=("net_return", "mean"),
        positive=("net_return", lambda r: (r > 0).mean()),
        rugged=("rugged", "mean"),
    ).reset_index()
    return table.sort_values(["checkpoint", "group"], key=lambda s: s.map(order) if s.name == "checkpoint" else s)


def verdict_lines(table: pd.DataFrame, strategies: list[str], min_tokens: int = 100,
                  checkpoints=("quick_exit", "1h", "24h")) -> list[str]:
    """Plain-English reading of the summary table, per strategy and checkpoint.

    Judged by the MEDIAN trade: a few huge winners or broken readings cannot distort it.
    """
    lines = []
    for strategy in strategies:
        for checkpoint in checkpoints:
            rows = table[table["checkpoint"] == checkpoint].set_index("group") if not table.empty else table
            name = f"{strategy} @ {checkpoint}"
            if strategy not in rows.index:
                lines.append(f"{name}: no results yet.")
                continue
            pick = rows.loc[strategy]
            if pick["tokens"] < min_tokens:
                lines.append(f"{name}: only {int(pick['tokens'])} trades so far - too early to judge (need {min_tokens}+).")
                continue
            money = "MADE money" if pick["median_net"] > 0 else "LOST money"
            line = (f"{name}: the typical trade {money} ({pick['median_net']:+.1%} after costs; "
                    f"{pick['positive']:.0%} ended up, {pick['rugged']:.0%} rugged).")
            control_group = "control_liquid" if "control_liquid" in rows.index else "control"
            if control_group in rows.index:
                control = rows.loc[control_group]
                better = "the rules HELPED" if pick["median_net"] > control["median_net"] else "the rules did NOT help"
                line += (f" Equally liquid random pools: {control['median_net']:+.1%}, "
                         f"{control['rugged']:.0%} rugged, so {better}.")
            lines.append(line)
    return lines
