"""New meme coin scanner (paper only): scan, track, report.

Examples:
    python -m trading_agent.memescan scan          # look at new and trending pools once
    python -m trading_agent.memescan track         # record prices for entries that are due (5m ... 7d)
    python -m trading_agent.memescan report        # how has each strategy done vs the comparison group?
    python -m trading_agent.memescan run --every 15   # scan every 15 minutes, track every 5, until stopped
"""

import argparse
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import requests

from .config import PROJECT_ROOT, load_config
from .memecoins import journal
from .memecoins.copycats import load_top_symbols
from .memecoins.http import ApiClient
from .memecoins.screen import NOT_READY, market_reasons, security_reasons
from .memecoins.security import check_token
from .memecoins.sources import fetch_new_pools, fetch_pools, fetch_trending_pools

DATA_DIR = PROJECT_ROOT / "data" / "memecoins"
CANDIDATES = DATA_DIR / "candidates.csv"
CHECKS = DATA_DIR / "checks.csv"
TOP_SYMBOLS = DATA_DIR / "top_symbols.json"
LOCK_FILE = DATA_DIR / "scanner.lock"


def entry_row(strategy, pool, now, group, reasons, sell_tax=None) -> dict:
    return {
        "entry_id": journal.entry_id(strategy, pool.pool_address), "strategy": strategy,
        "seen_at": now.isoformat(), "network": pool.network, "pool_address": pool.pool_address,
        "token_address": pool.token_address, "symbol": pool.symbol, "name": pool.name, "dex": pool.dex,
        "pool_created_at": pool.created_at.isoformat() if pd.notna(pool.created_at) else "",
        "price_usd": pool.price_usd, "liquidity_usd": pool.liquidity_usd, "fdv_usd": pool.fdv_usd,
        "volume_h1": pool.volume_h1, "buys_h1": pool.buys_h1, "sells_h1": pool.sells_h1,
        "change_h1": pool.change_h1, "change_h6": pool.change_h6,
        "passed": group == "pick", "group": group, "sell_tax": sell_tax if sell_tax is not None else "",
        "reasons": "; ".join(reasons),
    }


def scan(client, cfg: dict, rng: random.Random, now=None, candidates_path=CANDIDATES,
         top_symbols: set[str] = frozenset()) -> list[dict]:
    """Judge each listed pool once per strategy (and once for the comparison group).

    Pools that fail only for reasons that can change ("too new", "little trading" ...)
    are not recorded, so they are judged again when they reappear.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    known = journal.read_csv(candidates_path)
    judged = set(known["entry_id"]) if not known.empty else set()
    strategies = cfg["strategies"]
    rows = []
    for network in cfg["networks"]:
        try:
            pools = fetch_new_pools(client, network, cfg["pages"]) + fetch_trending_pools(client, network, cfg["pages"])
        except requests.RequestException as e:
            print(f"  {network}: could not reach GeckoTerminal ({type(e).__name__}); skipped")
            continue
        unique = {p.pool_address: p for p in pools}
        security_cache = {}
        picks = {name: 0 for name in strategies}
        for pool in unique.values():
            for name, rules in strategies.items():
                eid = journal.entry_id(name, pool.pool_address)
                if eid in judged or network not in rules.get("networks", cfg["networks"]):
                    continue
                reasons, sell_tax = market_reasons(pool, rules, now, top_symbols), None
                if not reasons:
                    if pool.token_address not in security_cache:
                        try:
                            security_cache[pool.token_address] = check_token(client, network, pool.token_address)
                        except requests.RequestException:
                            security_cache[pool.token_address] = None
                    report = security_cache[pool.token_address]
                    if report is None:
                        reasons = ["security check failed"]
                    else:
                        reasons, sell_tax = security_reasons(report, {**cfg, **rules}), report.sell_tax
                elif set(reasons) <= NOT_READY and not rules.get("judge_once"):
                    continue  # may still qualify later
                group = "" if reasons else "pick"
                picks[name] += group == "pick"
                rows.append(entry_row(name, pool, now, group, reasons, sell_tax))
                judged.add(eid)
            control_id = journal.entry_id("control", pool.pool_address)
            if control_id not in judged and pool.liquidity_usd >= cfg["control_min_liquidity_usd"]:
                group = "control" if rng.random() < cfg["control_rate"] else ""
                rows.append(entry_row("control", pool, now, group, []))
                judged.add(control_id)
        summary = ", ".join(f"{n} {c}" for n, c in picks.items())
        print(f"  {network:7} {len(unique):3} pools listed; picks: {summary}")
    journal.append_rows(candidates_path, journal.CANDIDATE_COLUMNS, rows)
    for r in rows:
        if r["group"] == "pick":
            print(f"  PICK [{r['strategy']}] {r['network']:7} {r['symbol']:<12} "
                  f"liquidity ${r['liquidity_usd']:,.0f}  price ${r['price_usd']:.10g}")
    return rows


def track(client, cfg: dict, now=None, candidates_path=CANDIDATES, checks_path=CHECKS) -> list[dict]:
    """Record the price of tracked entries whose checkpoint (5m, 15m, 30m, 1h, 24h, 7d) has arrived."""
    now = now or pd.Timestamp.now(tz="UTC")
    due = journal.due_checks(journal.read_csv(candidates_path), journal.read_csv(checks_path), now)
    rows = []
    for network, group in (due.groupby("network") if not due.empty else []):
        try:
            current = fetch_pools(client, network, sorted(set(group["pool_address"])))
        except requests.RequestException as e:
            print(f"  {network}: tracking skipped ({type(e).__name__}); will retry next time")
            continue
        for _, entry in group.iterrows():
            pool = current.get(entry["pool_address"])
            price = pool.price_usd if pool else float("nan")
            liquidity = pool.liquidity_usd if pool else float("nan")
            gross, net, rugged = journal.net_return(entry.to_dict(), price, liquidity, cfg)
            rows.append({
                "entry_id": entry["entry_id"], "strategy": entry["strategy"], "network": network,
                "pool_address": entry["pool_address"], "symbol": entry["symbol"], "group": journal.label(entry),
                "checkpoint": entry["checkpoint"], "checked_at": now.isoformat(),
                "price_usd": price, "liquidity_usd": liquidity,
                "gross_return": gross, "net_return": net, "rugged": rugged,
            })
    journal.append_rows(checks_path, journal.CHECK_COLUMNS, rows)
    if rows:
        print(f"  recorded {len(rows)} checkpoint(s)")
    return rows


def report(candidates_path=CANDIDATES, checks_path=CHECKS, cfg: dict | None = None) -> None:
    cfg = cfg or load_config()["memecoins"]
    strategies = list(cfg["strategies"])
    candidates = journal.read_csv(candidates_path)
    if candidates.empty:
        print("No data yet. Run: python -m trading_agent.memescan run --every 15")
        return
    print(f"From {candidates['seen_at'].min()} to {candidates['seen_at'].max()}")
    picks = candidates[candidates["group"] == "pick"]
    for name in strategies:
        mine = picks[picks["strategy"] == name]
        print(f"  {name:11} picks: {len(mine):5,}   per network: {mine['network'].value_counts().to_dict() or '-'}")
    print(f"  comparison group: {(candidates['group'] == 'control').sum():,} random pools")

    rejected = candidates[(candidates["strategy"] != "control") & (candidates["group"] != "pick")]
    reasons = rejected["reasons"].dropna().str.split("; ").explode()
    print("\nMost common final reasons for rejection (all strategies):")
    print(reasons.value_counts().head(10).to_string())

    checks = journal.read_csv(checks_path)
    if checks.empty:
        print("\nNo checkpoints recorded yet (the first arrive 5 minutes after an entry).")
        return
    suspect = int((checks["gross_return"] > journal.SUSPECT_GROSS).sum())
    print(f"\nLeft out {suspect} checkpoint(s) with an impossible price jump (over 100x: broken readings).")
    checks = journal.add_liquid_control(journal.clean(checks), candidates, cfg["control_liquid_usd"])
    exits = journal.quick_exit(checks, cfg["quick_exit_take_profit"], cfg["quick_exit_stop_loss"])
    table = journal.summarize(pd.concat([checks, exits], ignore_index=True))
    shown = table.copy()
    for col in ("median_net", "mean_net"):
        shown[col] = table[col].map("{:+.1%}".format)
    for col in ("positive", "rugged"):
        shown[col] = table[col].map("{:.0%}".format)
    print(f"\nResults after estimated costs (paper trade of ${cfg['trade_usd']} each):")
    print(shown.to_string(index=False))
    print(f"\nquick_exit = sell at the first check at +{cfg['quick_exit_take_profit']:.0%} or "
          f"-{cfg['quick_exit_stop_loss']:.0%}, otherwise at 1 hour (prices between checks are unseen).")
    print("\nVERDICT")
    for line in journal.verdict_lines(table, strategies):
        print("  " + line)
    print("\n'median_net' is the typical result per paper trade; 'control_liquid' = random pools with at least")
    print(f"${cfg['control_liquid_usd']:,} liquidity (the fair comparison). Paper results with an optimistic")
    print("cost model: real trading would likely do somewhat worse.")


def main() -> None:
    parser = argparse.ArgumentParser(description="New meme coin scanner (paper only, no wallet or keys).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    sub.add_parser("track")
    sub.add_parser("report")
    p_run = sub.add_parser("run", help="scan + track repeatedly")
    p_run.add_argument("--every", type=float, default=15, help="minutes between scans")
    p_run.add_argument("--track-every", type=float, default=5, help="minutes between price checks")
    p_run.add_argument("--log", type=Path, help="write all output to this file (for running without a window)")
    args = parser.parse_args()
    if args.command == "run" and args.log:
        log_to_file(args.log)

    cfg = load_config()["memecoins"]
    client = ApiClient()
    rng = random.Random()
    def top():
        return load_top_symbols(client, TOP_SYMBOLS, pd.Timestamp.now(tz="UTC"))

    if args.command == "scan":
        scan(client, cfg, rng, top_symbols=top())
    elif args.command == "track":
        track(client, cfg)
    elif args.command == "report":
        report()
    else:
        with single_instance(LOCK_FILE):
            print(f"Scanner started: scanning every {args.every:g} minutes, checking prices every "
                  f"{args.track_every:g} (stop with Ctrl+C or by stopping the service).")
            last_scan = 0.0
            while True:
                try:
                    if time.monotonic() - last_scan >= args.every * 60 - 1:
                        last_scan = time.monotonic()
                        print(f"\n[{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC] scan")
                        scan(client, cfg, rng, top_symbols=top())
                    track(client, cfg)
                except Exception:  # running unattended: log it and retry next round
                    traceback.print_exc()
                time.sleep(args.track_every * 60)


def log_to_file(path: Path) -> None:
    """Send all printed output and errors to `path`. Needed under pythonw.exe, which has no window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered: each line is saved at once
    sys.stdout = sys.stderr = log


@contextmanager
def single_instance(path):
    """Refuse to start if another scanner (e.g. the startup one) already holds the lock.

    The operating system releases the lock automatically if the process stops.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit("The scanner is already running (another window or the startup copy). Not starting a second one.")
    try:
        yield
    finally:
        handle.close()


if __name__ == "__main__":
    main()
