"""Meme coin rule lab (paper only): which buy and sell rules would have made money?

Uses the scanner's own records (every tracked pool: the picks and the random comparison
group) plus 1-minute price candles for the 4 hours after each was found, so selling
rules like "take +30% profit or cut at -20%" can be replayed minute by minute.

    python -m trading_agent.memelab fetch      # download candles (slow: ~4 s per pool; resumable)
    python -m trading_agent.memelab analyze    # replay rules; design on the earlier 60%, test on the rest

Honest-testing rules:
  * Rules are chosen on the DESIGN period (first 60% of the scanner v2 data by time) and judged
    once on the TEST period (the rest). Week 1 is left out: it was not checked in the first hour.
  * Entry = the price the scanner saw; exits use only candles that started after that minute.
  * If a stop and a target are both hit in the same minute, the stop counts (pessimistic).
  * A rug pull (liquidity gone at a later check) counts as -100% from the previous check on.
  * Costs: DEX fee, price impact on the way in and out, token tax and network fee.
"""

import argparse
import random
import time
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
PRIORITY_LIQUIDITY = 5_000  # smaller pools: even a small trade moves the price a lot


TAKE_PROFITS = [0.1, 0.2, 0.3, 0.5, 1.0, None]   # None = no target
STOP_LOSSES = [0.1, 0.2, 0.3, 0.5, None]          # None = no stop
HOLDS_MIN = [5, 15, 30, 60, 120, 240]             # sell at this age if nothing else triggered
MIN_LIQUIDITY = [0, 5_000, 20_000, 50_000]
MIN_BUY_SHARE = [0.0, 0.5, 0.6]                   # buys / (buys + sells) in the hour before
NETWORKS = ["all", "solana", "base", "bsc", "eth"]
MIN_TRADES = 200                                  # a rule must have this many design trades
LAUNCHPAD_DEXES = {"pump-fun", "meteora-dbc"}


def tracked_entries(paths=CANDIDATE_FILES) -> pd.DataFrame:
    """Every tracked pool (pick or comparison), once per (network, pool, time first seen).

    `period` is 'design' for the first file (scanner week 1) and 'test' for later ones.
    """
    frames = [journal.read_csv(p).assign(period="design" if i == 0 else "test")
              for i, p in enumerate(paths) if Path(p).exists()]
    entries = pd.concat(frames, ignore_index=True)
    entries = entries[entries["group"].isin(["pick", "control"])]
    entries = entries.sort_values("group", ascending=False)  # keep the 'pick' row when both exist
    return entries.drop_duplicates(KEY).sort_values("seen_at").reset_index(drop=True)


def fetch(client, entries: pd.DataFrame, now: pd.Timestamp, candles_path=CANDLES, fetched_path=FETCHED,
          rng: random.Random | None = None, reverse: bool = False) -> int:
    """Download candles for entries whose 4-hour window has ended; skips ones already done.

    Pools with at least PRIORITY_LIQUIDITY come first; within that, the order is random, so
    a partial download is still a fair sample. `reverse` walks each part from the other end,
    so a second machine can download at the same time without repeating the first one's work.
    """
    done = journal.read_csv(fetched_path)
    done_keys = set(map(tuple, done[KEY].astype(str).values)) if not done.empty else set()
    todo = entries[pd.to_datetime(entries["seen_at"], format="ISO8601") + WINDOW < now]
    todo = [r for _, r in todo.iterrows() if (r["network"], r["pool_address"], r["seen_at"]) not in done_keys]
    (rng or random.Random(0)).shuffle(todo)
    if reverse:
        todo.reverse()
    # Strategy picks first (forward tests wait on them), then bigger pools; stable, so random within each.
    todo.sort(key=lambda r: (r.get("group") != "pick", float(r["liquidity_usd"]) < PRIORITY_LIQUIDITY))
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


def rug_times(checks: pd.DataFrame) -> dict[tuple[str, str], pd.Timestamp]:
    """Earliest time each pool may have been rugged: the last check before the first rugged one.

    (The exact moment is unknown; assuming the earliest possible time is the cautious choice.)
    """
    times = {}
    if checks.empty:
        return times
    checks = checks.assign(checked_at=pd.to_datetime(checks["checked_at"], format="ISO8601"))
    for key, g in checks.sort_values("checked_at").groupby(["network", "pool_address"]):
        rugged = g["rugged"].astype(str) == "True"
        if rugged.any():
            first = g.loc[rugged, "checked_at"].iloc[0]
            before = g.loc[(~rugged) & (g["checked_at"] < first), "checked_at"]
            times[key] = before.iloc[-1] if len(before) else first - pd.Timedelta(minutes=5)
    return times


def replay(entry, candles: pd.DataFrame, rug_at, tp, sl, hold_min: float, cost_cfg: dict) -> dict:
    """One paper trade: buy at the price the scanner saw, then sell on target / stop / time.

    `candles` must start after the entry minute. Returns gross and net return and the exit reason.
    """
    p0 = float(entry["price_usd"])
    minutes = candles["minute"].to_numpy()
    live = minutes < hold_min
    o, h, l, c = (candles[k].to_numpy()[live] for k in ("open", "high", "low", "close"))
    hit_sl = l <= p0 * (1 - sl) if sl is not None else np.zeros(len(l), bool)
    hit_tp = h >= p0 * (1 + tp) if tp is not None else np.zeros(len(h), bool)
    hits = np.flatnonzero(hit_sl | hit_tp)
    if len(hits):
        i = hits[0]
        if hit_sl[i]:  # checked first: if both happen in one minute, assume the worse came first
            price, reason = min(p0 * (1 - sl), o[i]), "stop"
        else:
            price, reason = max(p0 * (1 + tp), o[i]), "target"
        exit_min = minutes[live][i] + 1
    else:
        price, reason, exit_min = (c[-1] if len(c) else p0), "time", hold_min
    last_trade = minutes[-1] + 1 if len(minutes) else 0  # trading still happened up to here
    if rug_at is not None and max(rug_at, last_trade) < exit_min:
        return {"gross": -1.0, "net": -1.0, "reason": "rug"}
    gross = price / p0 - 1
    liq0 = float(entry["liquidity_usd"])
    tax = pd.to_numeric(entry.get("sell_tax"), errors="coerce")
    tax = 0.0 if pd.isna(tax) else float(tax)
    buy = journal.side_cost(liq0, cost_cfg["trade_usd"], cost_cfg, entry["network"], tax)
    # In a constant-product pool, the money on each side moves with the square root of the price.
    sell = journal.side_cost(liq0 * np.sqrt(max(price / p0, 1e-9)), cost_cfg["trade_usd"], cost_cfg,
                             entry["network"], tax)
    return {"gross": gross, "net": (1 + gross) * (1 - buy) * (1 - sell) - 1, "reason": reason}


def prepare(entries: pd.DataFrame, candles: pd.DataFrame, checks: pd.DataFrame) -> list[tuple]:
    """(entry, its candles after the entry minute, minutes until possible rug) for each usable entry.

    Entries whose recorded price is far from the traded prices (a broken reading) are left out.
    """
    candles = candles.assign(ts=pd.to_datetime(candles["ts"], format="ISO8601"))
    by_key = {k: g for k, g in candles.groupby(KEY)}
    rugs = rug_times(checks)
    prepared, broken = [], 0
    for _, e in entries.iterrows():
        seen = pd.Timestamp(e["seen_at"])
        g = by_key.get((e["network"], e["pool_address"], e["seen_at"]), candles.iloc[:0])
        p0 = float(e["price_usd"])
        if not p0 > 0:
            broken += 1
            continue
        nearby = g.loc[g["ts"] <= seen.floor("min") + pd.Timedelta(minutes=3), "close"]
        if len(nearby) and not (1 / 3 < nearby.median() / p0 < 3):
            broken += 1
            continue
        after = g[g["ts"] > seen.floor("min")]
        after = after.assign(minute=(after["ts"] - seen) / pd.Timedelta(minutes=1)).reset_index(drop=True)
        rug = rugs.get((e["network"], e["pool_address"]))
        rug_min = None if rug is None else (rug - seen) / pd.Timedelta(minutes=1)
        prepared.append((e, after, rug_min))
    print(f"  {len(prepared):,} usable entries; left out {broken} with a broken entry price")
    return prepared


def features(entries: pd.DataFrame) -> pd.DataFrame:
    """What the scanner knew at the moment it found the pool."""
    buys, sells = entries["buys_h1"].astype(float), entries["sells_h1"].astype(float)
    age = (pd.to_datetime(entries["seen_at"], format="ISO8601")
           - pd.to_datetime(entries["pool_created_at"], format="ISO8601", errors="coerce", utc=True))
    out = pd.DataFrame({
        "log_liquidity": np.log10(entries["liquidity_usd"].astype(float).clip(lower=1)),
        "log_fdv": np.log10(entries["fdv_usd"].astype(float).clip(lower=1)),
        "log_volume_h1": np.log10(entries["volume_h1"].astype(float).clip(lower=0) + 1),
        "log_trades_h1": np.log10(buys + sells + 1),
        "buy_share": buys / (buys + sells).replace(0, np.nan),
        "log_age_min": np.log10((age / pd.Timedelta(minutes=1)).clip(lower=0) + 1),
        "volume_to_liquidity": entries["volume_h1"].astype(float) / entries["liquidity_usd"].astype(float),
        "pump_fun": entries["dex"].astype(str).str.contains("pump").astype(float),
    }, index=entries.index)
    for net in ["solana", "base", "bsc", "eth"]:
        out[f"net_{net}"] = (entries["network"] == net).astype(float)
    return out


def exit_rules():
    for tp in TAKE_PROFITS:
        for sl in STOP_LOSSES:
            for hold in HOLDS_MIN:
                yield tp, sl, hold


def rule_name(tp, sl, hold) -> str:
    return (f"target {'none' if tp is None else f'+{tp:.0%}'} / stop {'none' if sl is None else f'-{sl:.0%}'}"
            f" / sell after {hold if hold < 60 else hold // 60}{'m' if hold < 60 else 'h'}")


def replay_all(prepared: list[tuple], cost_cfg: dict) -> pd.DataFrame:
    """Net result of every entry under every exit rule: rows = entries, columns = exit rules.

    A gain above SUSPECT_GROSS (100x) is almost always a broken reading; like the scanner's
    own report, it is left out (NaN) rather than counted.
    """
    columns = {}
    for tp, sl, hold in exit_rules():
        results = (replay(e, c, r, tp, sl, hold, cost_cfg) for e, c, r in prepared)
        columns[(tp, sl, hold)] = [np.nan if x["gross"] > journal.SUSPECT_GROSS else x["net"] for x in results]
    table = pd.DataFrame(columns, index=[e.name for e, _, _ in prepared])
    table.columns = [rule_name(*k) for k in columns]
    return table


def entry_filters(entries: pd.DataFrame):
    """(name, boolean mask) for each combination of simple buying rules."""
    buys, sells = entries["buys_h1"].astype(float), entries["sells_h1"].astype(float)
    share = (buys / (buys + sells).replace(0, np.nan)).fillna(0)
    liquidity = entries["liquidity_usd"].astype(float)
    for net in NETWORKS:
        on_net = entries["network"] == net if net != "all" else pd.Series(True, index=entries.index)
        for liq in MIN_LIQUIDITY:
            for bs in MIN_BUY_SHARE:
                name = f"{net}, liquidity >= ${liq:,}" + (f", buyers >= {bs:.0%}" if bs else "")
                yield name, on_net & (liquidity >= liq) & (share >= bs)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    """90% range of the average trade, from resampling the trades."""
    if len(values) < 2:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def stats(values: pd.Series) -> dict:
    v = values.dropna()
    return {"trades": len(v), "mean": v.mean(), "median": v.median(), "won": (v > 0).mean()}


def search(entries: pd.DataFrame, nets: pd.DataFrame) -> pd.DataFrame:
    """Every (buying rule x selling rule) scored on the design period, with its test result."""
    rows = []
    design = entries["period"] == "design"
    for fname, mask in entry_filters(entries):
        d, t = nets[mask & design], nets[mask & ~design]
        if len(d) < MIN_TRADES:
            continue
        for rule in nets.columns:
            rows.append({"buy_rule": fname, "sell_rule": rule,
                         "design_trades": len(d), "design_mean": d[rule].mean(),
                         "test_trades": len(t), "test_mean": t[rule].mean(), "test_median": t[rule].median(),
                         "test_won": (t[rule] > 0).mean()})
    return pd.DataFrame(rows).sort_values("design_mean", ascending=False).reset_index(drop=True)


def model_test(entries: pd.DataFrame, nets: pd.Series, rng: np.random.Generator) -> dict:
    """Machine learning: learn on the design period which pools do best, then pick on the test period."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = features(entries)
    known = nets.notna().to_numpy()  # broken readings were left out
    design = (entries["period"] == "design").to_numpy() & known
    testing = (entries["period"] != "design").to_numpy() & known
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=50, random_state=0)
    model.fit(X[design], nets[design].clip(-1, 3))  # clip: one lucky 50x must not steer the model
    predicted = pd.Series(model.predict(X[testing]), index=X.index[testing])
    test = nets[testing]
    picked = test[predicted >= predicted.quantile(0.8)]  # the 20% it likes best
    return {"all": stats(test), "picked": stats(picked), "picked_range": bootstrap_mean(picked.to_numpy(), rng),
            "positive_predictions": int((predicted > 0).sum())}


def load_downloads(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fetched list, candles) from every machine's files (lab_fetched*.csv, lab_candles*.csv).

    A pool downloaded by two machines is kept once (from the first file that has it).
    """
    fetched, candles = [], []
    for f in sorted(data_dir.glob("lab_fetched*.csv")):
        done = journal.read_csv(f)
        seen = set(map(tuple, pd.concat(fetched)[KEY].values)) if fetched else set()
        done = done[[tuple(k) not in seen for k in done[KEY].values]]
        c = journal.read_csv(f.with_name(f.name.replace("lab_fetched", "lab_candles")))
        if not c.empty:
            keys = set(map(tuple, done[KEY].values))
            c = c[[tuple(k) in keys for k in c[KEY].values]]
        fetched.append(done)
        candles.append(c)
    return (pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame(columns=KEY),
            pd.concat(candles, ignore_index=True) if candles else pd.DataFrame())


def split_by_time(entries: pd.DataFrame, design_share: float = 0.6) -> pd.DataFrame:
    """Use only the scanner v2 data (5-minute checks from the start, so early rugs are seen);
    the first `design_share` of it by time is the design period, the rest the test period.

    Week 1 was checked only from 1 hour on, so pools that were rugged within the first hour
    were not flagged there; mixing it in made rules look far better than they were.
    """
    v2 = entries[entries["period"] == "test"].sort_values("seen_at")
    cut = v2["seen_at"].iloc[int(len(v2) * design_share)]
    return v2.assign(period=np.where(v2["seen_at"] < cut, "design", "test"))


def analyze(cost_cfg: dict, top: int = 15) -> None:
    rng = np.random.default_rng(0)
    entries = tracked_entries()
    fetched, candles = load_downloads()
    done = set(map(tuple, fetched[KEY].astype(str).values))
    entries = split_by_time(entries[[tuple(map(str, k)) in done for k in entries[KEY].values]])
    checks = pd.concat([journal.read_csv(p) for p in CHECK_FILES if Path(p).exists()], ignore_index=True)
    print(f"Meme rule lab - paper replay at ${cost_cfg['trade_usd']:,} per trade, after costs")
    print(f"  design period: {(entries['period'] == 'design').sum():,} pools, "
          f"test period: {(entries['period'] == 'test').sum():,} pools")
    prepared = prepare(entries, candles, checks)
    entries = pd.DataFrame([e for e, _, _ in prepared])
    nets = replay_all(prepared, cost_cfg)

    print("\n1) SELLING RULES on all pools (no buying filter): average trade")
    design = entries["period"] == "design"
    overview = pd.DataFrame({"design_mean": nets[design].mean(), "test_mean": nets[~design].mean(),
                             "test_median": nets[~design].median(), "test_won": (nets[~design] > 0).mean()})
    best = overview.sort_values("design_mean", ascending=False)
    print(_fmt(best.head(10)).to_string())
    print(f"  ... {len(overview)} selling rules tried; the best on design, judged on test, is above.")

    print(f"\n2) BUYING x SELLING RULES: best {top} of {len(nets.columns) * sum(1 for _ in entry_filters(entries))} "
          f"by design result (min {MIN_TRADES} design trades)")
    results = search(entries, nets)
    print(_fmt(results.head(top)).to_string(index=False))
    winner = results.iloc[0]
    mask = dict(entry_filters(entries))[winner["buy_rule"]]
    test_values = nets.loc[mask & ~design, winner["sell_rule"]].dropna().to_numpy()
    lo, hi = bootstrap_mean(test_values, rng)
    print(f"\n  Chosen in advance (best on design): {winner['buy_rule']} | {winner['sell_rule']}")
    print(f"  Design: {winner['design_mean']:+.1%} per trade over {winner['design_trades']} trades")
    print(f"  TEST:   {winner['test_mean']:+.1%} per trade over {winner['test_trades']} trades "
          f"(90% range {lo:+.1%} to {hi:+.1%})")
    # Launchpad pools (pump.fun, Meteora DBC) move to a new pool when they "graduate"; the old
    # pool then empties, which this replay counts as a rug. Show the result without them too.
    launchpad = entries["dex"].astype(str).isin(LAUNCHPAD_DEXES)
    plain = nets.loc[mask & ~design & ~launchpad, winner["sell_rule"]].dropna()
    print(f"  Without launchpad pools (possible false rugs): {plain.mean():+.1%} per trade over {len(plain)} trades")
    share_positive = (results.head(100)["test_mean"] > 0).mean()
    print(f"  Of the top 100 design rules, {share_positive:.0%} also made money on test.")

    rule = winner["sell_rule"]
    m = model_test(entries, nets[rule], rng)
    print(f"\n3) MACHINE LEARNING (selling rule: {rule})")
    print(f"  Test period, all pools:          {m['all']['mean']:+.1%} average over {m['all']['trades']} trades")
    print(f"  Test period, model's top 20%:    {m['picked']['mean']:+.1%} average over {m['picked']['trades']} "
          f"trades (90% range {m['picked_range'][0]:+.1%} to {m['picked_range'][1]:+.1%})")
    print(f"  Pools the model expected to make money: {m['positive_predictions']}")

    out = DATA_DIR / "lab_results.csv"
    results.to_csv(out, index=False)
    print(f"\nAll {len(results):,} rule results saved to {out}")
    print("Paper replay only. Prices between trades, bots and failed sells make real results worse.")


def replay_next_minute(entry, candles: pd.DataFrame, rug_at, tp, sl, hold_min: float, cost_cfg: dict) -> dict | None:
    """Like replay(), but buying at the open of the first minute after the scanner saw the pool
    (the soonest a real trade could fill). None if nothing traded within 2 minutes."""
    first = candles[candles["minute"] <= 2]
    if first.empty:
        return None
    start = float(first["minute"].iloc[0])
    entry = entry.copy()
    entry["price_usd"] = float(first["open"].iloc[0])
    later = candles.assign(minute=candles["minute"] - start)
    return replay(entry, later[later["minute"] >= 0], None if rug_at is None else rug_at - start,
                  tp, sl, hold_min, cost_cfg)


def forward(cost_cfg: dict, strategy: str = "bsc_momentum", min_trades: int = 100) -> None:
    """Forward paper test of a strategy fixed in config.toml, replayed on 1-minute candles.

    Pre-registered verdict: PASS only with min_trades+ trades, a positive median trade AND the
    90% range of the average trade above zero. Compared with comparison-group pools that meet
    the same market rules but had no safety check.
    """
    rng = np.random.default_rng(0)
    rules = cost_cfg["strategies"][strategy]
    start = pd.Timestamp(rules["forward_start"])
    candidates = journal.read_csv(CANDIDATE_FILES[-1])
    fetched, candles = load_downloads()
    done = set(map(tuple, fetched[KEY].astype(str).values)) if not fetched.empty else set()
    candidates = candidates[pd.to_datetime(candidates["seen_at"], format="ISO8601") >= start]
    buys, sells = candidates["buys_h1"].astype(float), candidates["sells_h1"].astype(float)
    same_market = ((candidates["network"].isin(rules["networks"]))
                   & (candidates["liquidity_usd"].astype(float) >= rules["min_liquidity_usd"])
                   & ((buys / (buys + sells).replace(0, np.nan)).fillna(0) >= rules["min_buy_share"]))
    groups = {
        "strategy": candidates[(candidates["strategy"] == strategy) & (candidates["group"] == "pick")],
        "comparison": candidates[(candidates["group"] == "control") & same_market],
    }
    checks = journal.read_csv(CHECK_FILES[-1])
    print(f"Forward paper test: {strategy} (since {start:%Y-%m-%d %H:%M} UTC, ${cost_cfg['trade_usd']:,} per trade)")
    print(f"  buy at the next minute's open, sell after {rules['forward_hold_minutes']} min "
          f"or at -{rules['forward_stop_loss']:.0%}; costs included")
    results = {}
    for name, group in groups.items():
        ready = group[[tuple(map(str, k)) in done for k in group[KEY].values]]
        nets = []
        for e, c, rug in prepare(ready, candles, checks) if len(ready) else []:
            r = replay_next_minute(e, c, rug, None, rules["forward_stop_loss"], rules["forward_hold_minutes"], cost_cfg)
            if r is not None and r["gross"] <= journal.SUSPECT_GROSS:
                nets.append(r["net"])
        v = np.array(nets)
        results[name] = v
        waiting = len(group) - len(ready)
        if len(v):
            lo, hi = bootstrap_mean(v, rng)
            print(f"  {name:10} {len(v):4} trades: average {v.mean():+.1%} (90% range {lo:+.1%} to {hi:+.1%}), "
                  f"typical {np.median(v):+.1%}, {np.mean(v > 0):.0%} made money"
                  f"{f'; {waiting} waiting for candles' if waiting else ''}")
        else:
            print(f"  {name:10}    0 trades yet{f'; {waiting} waiting for candles' if waiting else ''}")
    v = results["strategy"]
    if len(v) < min_trades:
        print(f"\n  VERDICT: too early - {len(v)} of {min_trades} trades.")
        return
    lo, _ = bootstrap_mean(v, rng)
    passed = np.median(v) > 0 and lo > 0
    print(f"\n  VERDICT: {'PASS - the edge held up on new data' if passed else 'FAIL - no reliable edge'} "
          f"(needs a positive typical trade and the 90% range above zero).")
    print("  Paper only. Real trading adds bots, delays and failed sells.")


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.endswith(("_mean", "_median")):
            df[col] = df[col].map("{:+.1%}".format)
        elif col.endswith("_won"):
            df[col] = df[col].map("{:.0%}".format)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay meme coin buy/sell rules on the scanner's data (paper only).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch", help="download 1-minute candles for every tracked pool")
    p_fetch.add_argument("--interval", type=float, default=4.0,
                         help="seconds between requests (leaves room for the scanner under the 30/min limit)")
    p_fetch.add_argument("--log", type=Path)
    p_fetch.add_argument("--reverse", action="store_true", help="start from the other end (for a second machine)")
    p_fetch.add_argument("--suffix", default="", help="separate output files, e.g. '_pc' (merged by analyze)")
    p_fetch.add_argument("--every", type=float, help="repeat every this many hours (for the server)")
    p_an = sub.add_parser("analyze", help="replay buying and selling rules on the downloaded candles")
    p_an.add_argument("--trade-usd", type=float, help="paper trade size (default: memecoins.trade_usd)")
    p_fw = sub.add_parser("forward", help="forward paper test of a strategy fixed in config.toml")
    p_fw.add_argument("--strategy", default="bsc_momentum")
    args = parser.parse_args()
    if args.command in ("analyze", "forward"):
        cfg = dict(load_config()["memecoins"])
        if args.command == "forward":
            forward(cfg, args.strategy)
            return
        if args.trade_usd:
            cfg["trade_usd"] = args.trade_usd
        analyze(cfg)
        return
    if args.command == "fetch":
        if args.log:
            log_to_file(args.log)
        with single_instance(DATA_DIR / f"lab{args.suffix}.lock"):
            while True:
                try:
                    fetch(ApiClient(min_interval=args.interval), tracked_entries(), pd.Timestamp.now(tz="UTC"),
                          DATA_DIR / f"lab_candles{args.suffix}.csv", DATA_DIR / f"lab_fetched{args.suffix}.csv",
                          reverse=args.reverse)
                    print(f"[{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC] download finished.")
                except Exception:  # running unattended: log it and retry next time
                    traceback.print_exc()
                if not args.every:
                    break
                time.sleep(args.every * 3600)


if __name__ == "__main__":
    main()
