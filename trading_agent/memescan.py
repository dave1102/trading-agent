"""New meme coin scanner (paper only): scan, track, report.

Examples:
    python -m trading_agent.memescan scan          # look at new and trending pools once
    python -m trading_agent.memescan track         # record prices for picks that are due (1h, 24h, 7d)
    python -m trading_agent.memescan report        # how have picks done vs the comparison group?
    python -m trading_agent.memescan run --every 15   # scan + track every 15 minutes until Ctrl+C
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
from .memecoins.screen import market_reasons, security_reasons
from .memecoins.security import check_token
from .memecoins.sources import fetch_new_pools, fetch_pools, fetch_trending_pools

DATA_DIR = PROJECT_ROOT / "data" / "memecoins"
CANDIDATES = DATA_DIR / "candidates.csv"
CHECKS = DATA_DIR / "checks.csv"
TOP_SYMBOLS = DATA_DIR / "top_symbols.json"
LOCK_FILE = DATA_DIR / "scanner.lock"
NOT_READY = {"too new", "low liquidity", "little trading", "nobody has sold (possible honeypot)"}


def scan(client, cfg: dict, rng: random.Random, now=None, candidates_path=CANDIDATES,
         top_symbols: set[str] = frozenset()) -> list[dict]:
    """Screen pools not seen before. Returns the rows written to the journal."""
    now = now or pd.Timestamp.now(tz="UTC")
    known = journal.read_csv(candidates_path)
    seen = set(known["pool_address"]) if not known.empty else set()
    rows = []
    for network in cfg["networks"]:
        try:
            pools = fetch_new_pools(client, network, cfg["pages"]) + fetch_trending_pools(client, network, cfg["pages"])
        except requests.RequestException as e:
            print(f"  {network}: could not reach GeckoTerminal ({type(e).__name__}); skipped")
            continue
        unique = {p.pool_address: p for p in pools if p.pool_address not in seen}
        passed = 0
        for pool in unique.values():
            reasons = market_reasons(pool, cfg, now, top_symbols)
            sell_tax = None
            if not reasons:
                try:
                    report = check_token(client, network, pool.token_address)
                    reasons, sell_tax = security_reasons(report, cfg), report.sell_tax
                except requests.RequestException:
                    reasons = ["security check failed"]
            in_control = (bool(reasons) and pool.liquidity_usd >= cfg["control_min_liquidity_usd"]
                          and rng.random() < cfg["control_rate"])
            if reasons and set(reasons) <= NOT_READY and not in_control:
                continue  # not judged yet: may be re-checked when it reappears
            group = "pick" if not reasons else "control" if in_control else ""
            passed += not reasons
            rows.append({
                "seen_at": now.isoformat(), "network": network, "pool_address": pool.pool_address,
                "token_address": pool.token_address, "symbol": pool.symbol, "name": pool.name, "dex": pool.dex,
                "pool_created_at": pool.created_at.isoformat() if pd.notna(pool.created_at) else "",
                "price_usd": pool.price_usd, "liquidity_usd": pool.liquidity_usd, "fdv_usd": pool.fdv_usd,
                "volume_h1": pool.volume_h1, "buys_h1": pool.buys_h1, "sells_h1": pool.sells_h1,
                "passed": not reasons, "group": group, "sell_tax": sell_tax if sell_tax is not None else "",
                "reasons": "; ".join(reasons),
            })
            seen.add(pool.pool_address)
        print(f"  {network:7} {len(pools):3} pools listed, {len(unique):3} new to us, {passed} passed every check")
    journal.append_rows(candidates_path, journal.CANDIDATE_COLUMNS, rows)
    for r in rows:
        if r["passed"]:
            print(f"  PICK {r['network']:7} {r['symbol']:<12} liquidity ${r['liquidity_usd']:,.0f}  "
                  f"price ${r['price_usd']:.10g}")
    return rows


def track(client, cfg: dict, now=None, candidates_path=CANDIDATES, checks_path=CHECKS) -> list[dict]:
    """Record the price of tracked pools whose 1h / 24h / 7d checkpoint has arrived."""
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
                "network": network, "pool_address": entry["pool_address"], "symbol": entry["symbol"],
                "group": entry["group"], "checkpoint": entry["checkpoint"], "checked_at": now.isoformat(),
                "price_usd": price, "liquidity_usd": liquidity,
                "gross_return": gross, "net_return": net, "rugged": rugged,
            })
    journal.append_rows(checks_path, journal.CHECK_COLUMNS, rows)
    if rows:
        print(f"  recorded {len(rows)} checkpoint(s)")
    return rows


def report(candidates_path=CANDIDATES, checks_path=CHECKS) -> None:
    candidates = journal.read_csv(candidates_path)
    if candidates.empty:
        print("No data yet. Run: python -m trading_agent.memescan run --every 15")
        return
    picks = candidates[candidates["group"] == "pick"]
    print(f"Pools judged: {len(candidates):,}   picks: {len(picks):,}   "
          f"comparison group: {(candidates['group'] == 'control').sum():,}")
    print(f"From {candidates['seen_at'].min()} to {candidates['seen_at'].max()}\n")
    print("Picks per network:", picks["network"].value_counts().to_dict() or "none")

    reasons = candidates.loc[~candidates["passed"].astype(bool), "reasons"].dropna().str.split("; ").explode()
    print("\nMost common reasons for rejection:")
    print(reasons.value_counts().head(10).to_string())

    checks = journal.read_csv(checks_path)
    if not checks.empty:
        suspect = int((checks["gross_return"] > journal.SUSPECT_GROSS).sum())
        print(f"\nLeft out {suspect} checkpoint(s) with an impossible price jump (over 100x: broken readings).")
    min_liquidity = load_config()["memecoins"]["min_liquidity_usd"]
    table = journal.summarize(journal.add_liquid_control(checks, candidates, min_liquidity))
    if table.empty:
        print("\nNo checkpoints recorded yet (the first arrive 1 hour after a pick).")
        return
    shown = table.copy()
    for col in ("median_net", "mean_net"):
        shown[col] = table[col].map("{:+.1%}".format)
    for col in ("positive", "rugged"):
        shown[col] = table[col].map("{:.0%}".format)
    print(f"\nResults after estimated costs (paper trade of ${load_config()['memecoins']['trade_usd']} each):")
    print(shown.to_string(index=False))
    print("\nVERDICT")
    for line in journal.verdict_lines(table):
        print("  " + line)
    print("\n'median_net' is the typical result per $100 paper trade; 'control_liquid' = comparison pools at least")
    print("as liquid as picks must be (the fair comparison). Paper results with an optimistic cost model:")
    print("real trading would likely do somewhat worse.")


def main() -> None:
    parser = argparse.ArgumentParser(description="New meme coin scanner (paper only, no wallet or keys).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    sub.add_parser("track")
    sub.add_parser("report")
    p_run = sub.add_parser("run", help="scan + track repeatedly")
    p_run.add_argument("--every", type=float, default=15, help="minutes between rounds")
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
            print(f"Scanner started: scanning every {args.every:g} minutes "
                  "(stop with Ctrl+C, or stop_scanner.cmd when running in the background).")
            while True:
                print(f"\n[{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC] scan")
                try:
                    scan(client, cfg, rng, top_symbols=top())
                    track(client, cfg)
                except Exception:  # running unattended: log it and retry next round
                    traceback.print_exc()
                time.sleep(args.every * 60)


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
