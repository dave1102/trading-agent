"""Meme coin rule lab (paper only): which buy and sell rules would have made money?

Uses the scanner's own records (every tracked pool: the picks and the random comparison
group) plus 1-minute price candles for the 4 hours after each was found, so selling
rules like "take +30% profit or cut at -20%" can be replayed minute by minute.

    python -m trading_agent.memelab fetch      # download candles (slow: ~4 s per pool; resumable)
    python -m trading_agent.memelab analyze    # replay rules; design on week 1, test on later data

Honest-testing rules:
  * Rules are chosen on the DESIGN period (scanner week 1) and judged once on the TEST period.
  * Entry = the price the scanner saw; exits use only candles that started after that minute.
  * If a stop and a target are both hit in the same minute, the stop counts (pessimistic).
  * A rug pull (liquidity gone at a later check) counts as -100% from the previous check on.
  * Costs: DEX fee, price impact on the way in and out, token tax and network fee.
"""

import argparse
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config
from .memecoins import journal
from .memecoins.candles import fetch_candles
from .memecoins.http import ApiClient
from .memescan import log_to_file, single_instance

DATA_DIR = PROJECT_ROOT / "data" / "memecoins"
CANDIDATE_FILES = [DATA_DIR / "week1_candidates.csv", DATA_DIR / "candidates.csv"]
CHECK_FILES = [DATA_DIR / "week1_checks.csv", DATA_DIR / "checks.csv"]
CANDLES = DATA_DIR / "lab_candles.csv"
FETCHED = DATA_DIR / "lab_fetched.csv"
LAB_LOG = DATA_DIR / "lab.log"
WINDOW = pd.Timedelta(hours=4)
KEY = ["network", "pool_address", "seen_at"]


def tracked_entries(paths=CANDIDATE_FILES) -> pd.DataFrame:
    """Every tracked pool (pick or comparison), once per (network, pool, time first seen)."""
    frames = [journal.read_csv(p) for p in paths if Path(p).exists()]
    entries = pd.concat(frames, ignore_index=True)
    entries = entries[entries["group"].isin(["pick", "control"])]
    entries = entries.sort_values("group", ascending=False)  # keep the 'pick' row when both exist
    return entries.drop_duplicates(KEY).sort_values("seen_at").reset_index(drop=True)


def fetch(client, entries: pd.DataFrame, now: pd.Timestamp, candles_path=CANDLES, fetched_path=FETCHED,
          rng: random.Random | None = None) -> int:
    """Download candles for entries whose 4-hour window has ended; skips ones already done.

    Pools are fetched in random order, so a partial download is still a fair sample.
    """
    done = journal.read_csv(fetched_path)
    done_keys = set(map(tuple, done[KEY].astype(str).values)) if not done.empty else set()
    todo = entries[pd.to_datetime(entries["seen_at"], format="ISO8601") + WINDOW < now]
    todo = [r for _, r in todo.iterrows() if (r["network"], r["pool_address"], r["seen_at"]) not in done_keys]
    (rng or random.Random(0)).shuffle(todo)
    print(f"{len(done_keys):,} pools already done, {len(todo):,} to fetch (about {len(todo) * client.min_interval / 3600:.1f} h)")
    for i, r in enumerate(todo, 1):
        seen = pd.Timestamp(r["seen_at"])
        try:
            candles = fetch_candles(client, r["network"], r["pool_address"], seen, seen + WINDOW)
        except Exception as e:  # one bad pool must not stop a 7-hour download
            print(f"  {r['network']} {r['pool_address']}: {type(e).__name__}; will retry on the next run")
            continue
        key = {k: r[k] for k in KEY}
        journal.append_rows(candles_path, [*KEY, "ts", "open", "high", "low", "close", "volume"],
                            [{**key, **c, "ts": c["ts"].isoformat()} for c in candles.to_dict("records")])
        journal.append_rows(fetched_path, [*KEY, "candles"], [{**key, "candles": len(candles)}])
        if i % 100 == 0:
            print(f"[{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC] {i:,}/{len(todo):,} pools")
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay meme coin buy/sell rules on the scanner's data (paper only).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch", help="download 1-minute candles for every tracked pool")
    p_fetch.add_argument("--interval", type=float, default=4.0,
                         help="seconds between requests (leaves room for the scanner under the 30/min limit)")
    p_fetch.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        if args.log:
            log_to_file(args.log)
        with single_instance(DATA_DIR / "lab.lock"):
            try:
                fetch(ApiClient(min_interval=args.interval), tracked_entries(), pd.Timestamp.now(tz="UTC"))
                print("Download finished.")
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
