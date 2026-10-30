"""Meme coin scanner tests, using hand-made API responses shaped like the real ones (no network)."""

import random

import pandas as pd
import pytest

from trading_agent import memescan
from trading_agent.memecoins import journal
from trading_agent.memecoins.copycats import COINGECKO_TOP, load_top_symbols
from trading_agent.memecoins.screen import market_reasons, security_reasons
from trading_agent.memecoins.security import SecurityReport, parse_goplus_evm, parse_solana
from trading_agent.memecoins.sources import parse_pools

NOW = pd.Timestamp("2026-10-23 12:00", tz="UTC")
CFG = {
    "networks": ["base"], "pages": 1, "min_age_minutes": 15, "max_age_hours": 24,
    "min_liquidity_usd": 20_000, "min_trades_h1": 50, "max_tax": 0.05, "max_top10_share": 0.30,
    "min_lp_locked": 0.90, "control_rate": 1.0, "control_min_liquidity_usd": 1_000,
    "trade_usd": 100, "dex_fee": 0.003, "gas_usd": {"base": 0.05, "eth": 3.0, "solana": 0.01},
}


def gt_pool(address, token="0xtoken", age_hours=2.0, liquidity=50_000, buys=60, sells=40, price=0.001, cg=None):
    created = (NOW - pd.Timedelta(hours=age_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "data": [{
            "attributes": {
                "address": address, "name": "MEME / WETH", "pool_created_at": created,
                "base_token_price_usd": str(price), "reserve_in_usd": str(liquidity), "fdv_usd": "500000",
                "volume_usd": {"h1": "12000"}, "transactions": {"h1": {"buys": buys, "sells": sells}},
            },
            "relationships": {
                "base_token": {"data": {"id": f"base_{token}", "type": "token"}},
                "dex": {"data": {"id": "uniswap", "type": "dex"}},
            },
        }],
        "included": [{"id": f"base_{token}", "type": "token",
                      "attributes": {"address": token, "symbol": "MEME", "name": "Meme", "coingecko_coin_id": cg}}],
    }


def goplus_evm(token="0xtoken", **overrides):
    info = {
        "is_honeypot": "0", "cannot_sell_all": "0", "owner_change_balance": "0", "hidden_owner": "0",
        "can_take_back_ownership": "0", "transfer_pausable": "0", "selfdestruct": "0",
        "honeypot_with_same_creator": "0", "buy_tax": "0", "sell_tax": "0.01", "is_mintable": "0",
        "dex": [{"pair": "0xpool"}],
        "holders": [
            {"address": "0xpool", "percent": "0.40", "is_locked": 0},  # the pool itself: ignored
            {"address": "0x000000000000000000000000000000000000dead", "percent": "0.30", "is_locked": 0},
            {"address": "0xwhale", "percent": "0.10", "is_locked": 0},
            {"address": "0xfish", "percent": "0.05", "is_locked": 0},
        ],
        "lp_holders": [
            {"address": "0x000000000000000000000000000000000000dead", "percent": "0.95", "is_locked": 0},
            {"address": "0xdev", "percent": "0.05", "is_locked": 0},
        ],
    }
    info.update(overrides)
    return {"code": 1, "result": {token: info}}


class FakeClient:
    """Answers GeckoTerminal / GoPlus URLs from a dictionary of {url fragment: response}."""

    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def get_json(self, url):
        self.urls.append(url)
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        return {"data": []}


def test_parse_pools_reads_geckoterminal_shape():
    [pool] = parse_pools(gt_pool("0xpool"), "base")
    assert pool.token_address == "0xtoken" and pool.symbol == "MEME"
    assert pool.liquidity_usd == 50_000 and pool.buys_h1 == 60 and pool.sells_h1 == 40
    assert pool.created_at == NOW - pd.Timedelta(hours=2)


def test_market_rules():
    [good] = parse_pools(gt_pool("0xa"), "base")
    assert market_reasons(good, CFG, NOW) == []
    [young] = parse_pools(gt_pool("0xb", age_hours=0.1), "base")
    assert "too new" in market_reasons(young, CFG, NOW)
    [listed] = parse_pools(gt_pool("0xc", cg="pepe"), "base")
    assert any("established" in r for r in market_reasons(listed, CFG, NOW))
    [no_sells] = parse_pools(gt_pool("0xd", sells=0, buys=80), "base")
    assert any("honeypot" in r for r in market_reasons(no_sells, CFG, NOW))


def test_copycat_tickers_are_rejected(tmp_path):
    [fake] = parse_pools(gt_pool("0xa"), "base")
    fake.symbol = "VIRTUAL"
    assert "copies the ticker of a top-250 coin" in market_reasons(fake, CFG, NOW, {"virtual", "btc"})
    [real] = parse_pools(gt_pool("0xb", cg="virtual-protocol"), "base")
    real.symbol = "VIRTUAL"
    assert "copies the ticker of a top-250 coin" not in market_reasons(real, CFG, NOW, {"virtual"})

    cache = tmp_path / "top.json"
    client = FakeClient({"coins/markets": [{"symbol": "BTC"}, {"symbol": "virtual"}]})
    assert load_top_symbols(client, cache, NOW) == {"btc", "virtual"}
    offline = FakeClient({})  # returns a non-list: the cached list is reused
    assert load_top_symbols(offline, cache, NOW + pd.Timedelta(days=2)) == {"btc", "virtual"}
    assert load_top_symbols(offline, cache, NOW + pd.Timedelta(hours=1)) == {"btc", "virtual"}
    assert offline.urls == [COINGECKO_TOP]  # only the stale-cache call; a fresh cache makes no call


def test_goplus_evm_parsing_ignores_pool_and_burn_wallets():
    report = parse_goplus_evm(goplus_evm(), "0xTOKEN")  # address case does not matter
    assert report.found and report.honeypot is False
    assert report.top10_share == pytest.approx(0.15)  # only the whale and the fish count
    assert report.lp_locked_share == pytest.approx(0.95)  # burned liquidity counts as locked
    assert security_reasons(report, CFG) == []


@pytest.mark.parametrize("override, expected", [
    ({"is_honeypot": "1"}, "honeypot"),
    ({"sell_tax": "0.25"}, "sell tax 25%"),
    ({"is_mintable": "1"}, "creator can mint more"),
    ({"owner_change_balance": "1"}, "owner can change balances"),
    ({"lp_holders": [{"address": "0xdev", "percent": "1.0", "is_locked": 0}]}, "only 0% of liquidity locked"),
])
def test_goplus_red_flags_fail(override, expected):
    reasons = security_reasons(parse_goplus_evm(goplus_evm(**override), "0xtoken"), CFG)
    assert expected in reasons


def test_unknown_security_fails():
    assert security_reasons(SecurityReport(), CFG) == ["security data unavailable"]
    report = parse_goplus_evm(goplus_evm(lp_holders=None, sell_tax=""), "0xtoken")
    reasons = security_reasons(report, CFG)
    assert "liquidity lock unknown" in reasons and "sell tax unknown" in reasons


def test_solana_parsing():
    status = lambda s: {"authority": [], "status": s}  # noqa: E731
    goplus = {"result": {"Mint1": {
        "mintable": status("0"), "freezable": status("0"), "balance_mutable_authority": status("0"),
        "transfer_fee_upgradable": status("0"), "transfer_fee": {}, "non_transferable": "0",
    }}}
    clean = parse_solana(goplus, {"risks": [], "lpLockedPct": 100}, "Mint1")
    assert security_reasons(clean, CFG) == []
    risky = parse_solana(goplus, {"risks": [{"name": "Single holder ownership", "level": "danger"}],
                                  "lpLockedPct": 20}, "Mint1")
    reasons = security_reasons(risky, CFG)
    assert "Single holder ownership" in reasons and "only 20% of liquidity locked" in reasons
    assert parse_solana(goplus, None, "Mint1").found  # GoPlus knew it, but RugCheck missing ...
    assert security_reasons(parse_solana(goplus, None, "Mint1"), CFG)  # ... still fails


def test_costs_and_rug_detection():
    entry = {"network": "base", "liquidity_usd": 50_000, "price_usd": 1.0, "sell_tax": ""}
    gross, net, rugged = journal.net_return(entry, 1.0, 50_000, CFG)
    assert gross == 0 and net < -0.01 and not rugged  # flat price still loses to fees and impact
    assert journal.net_return(entry, 2.0, 50_000, CFG)[1] > 0.9
    assert journal.net_return(entry, 1.0, 1_000, CFG) == (-1.0, -1.0, True)  # liquidity pulled
    eth = {**entry, "network": "eth"}
    assert journal.net_return(eth, 1.0, 50_000, CFG)[1] < net  # Ethereum gas makes small trades costly


def test_safe_text_defuses_spreadsheet_formulas():
    assert journal.safe_text("=HYPERLINK(\"x\")").startswith("'=")
    assert journal.safe_text("PEPE") == "PEPE"


def test_verdict_lines():
    def rows(group, n, mean):
        return [{"checkpoint": "24h", "group": group, "net_return": mean, "rugged": False}] * n

    early = journal.summarize(pd.DataFrame(rows("pick", 5, 0.1)))
    assert "too early" in journal.verdict_lines(early)[1]

    checks = pd.DataFrame(rows("pick", 120, -0.05) + rows("control", 120, -0.30))
    checks["gross_return"] = checks["net_return"]
    line = journal.verdict_lines(journal.summarize(checks))[1]
    assert "LOST money" in line and "the rules HELPED" in line
    assert journal.verdict_lines(journal.summarize(checks))[0] == "1h: no results yet."


def test_summary_ignores_broken_prices_and_uses_liquid_control():
    checks = pd.DataFrame([
        {"pool_address": "p1", "checkpoint": "24h", "group": "pick", "net_return": -0.5, "gross_return": -0.5, "rugged": False},
        {"pool_address": "c1", "checkpoint": "24h", "group": "control", "net_return": -1.0, "gross_return": -1.0, "rugged": True},
        {"pool_address": "c2", "checkpoint": "24h", "group": "control", "net_return": -0.1, "gross_return": -0.1, "rugged": False},
        {"pool_address": "c3", "checkpoint": "24h", "group": "control", "net_return": 1.6e7, "gross_return": 1.6e7, "rugged": False},
    ])
    candidates = pd.DataFrame({"pool_address": ["p1", "c1", "c2", "c3"], "liquidity_usd": [50_000, 30_000, 2_000, 40_000]})
    table = journal.summarize(journal.add_liquid_control(checks, candidates, 20_000)).set_index("group")
    assert table.loc["control", "tokens"] == 2  # the 16-million-x reading is left out
    assert table.loc["control_liquid", "tokens"] == 1  # only c1 was liquid (c3 is broken)
    assert table.loc["control_liquid", "rugged"] == 1.0


def test_log_to_file_captures_output_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(memescan.sys, "stdout", memescan.sys.stdout)  # restored after the test
    monkeypatch.setattr(memescan.sys, "stderr", memescan.sys.stderr)
    log = tmp_path / "logs" / "scanner.log"
    memescan.log_to_file(log)
    print("hello from the scanner")
    print("an error", file=memescan.sys.stderr)
    memescan.sys.stdout.close()
    assert log.read_text(encoding="utf-8").splitlines() == ["hello from the scanner", "an error"]


def test_api_client_retries_dropped_connections(monkeypatch):
    import requests

    from trading_agent.memecoins.http import ApiClient

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    class FlakySession:
        headers = {}

        def __init__(self, failures):
            self.failures = failures
            self.calls = 0

        def get(self, url, timeout):
            self.calls += 1
            if self.calls <= self.failures:
                raise requests.ConnectionError("Connection aborted: RemoteDisconnected")
            return Response()

    monkeypatch.setattr("trading_agent.memecoins.http.time.sleep", lambda s: None)
    session = FlakySession(failures=1)
    assert ApiClient(min_interval=0, session=session).get_json("https://x") == {"ok": True}
    assert session.calls == 2
    with pytest.raises(requests.ConnectionError):  # gives up after its retries
        ApiClient(min_interval=0, retries=2, session=FlakySession(failures=5)).get_json("https://x")


def test_only_one_scanner_can_run(tmp_path):
    lock = tmp_path / "scanner.lock"
    with memescan.single_instance(lock):
        with pytest.raises(SystemExit, match="already running"):
            with memescan.single_instance(lock):
                pass
    with memescan.single_instance(lock):  # free again once the first one stopped
        pass


def test_scan_track_report_end_to_end(tmp_path, capsys):
    candidates, checks = tmp_path / "candidates.csv", tmp_path / "checks.csv"
    routes = {
        "new_pools": {"data": []},
        "trending_pools": {
            "data": gt_pool("0xgood", "0xgood_t")["data"] + gt_pool("0xbad", "0xbad_t")["data"]
                    + gt_pool("0xyoung", "0xyoung_t", age_hours=0.05)["data"],
            "included": gt_pool("0xgood", "0xgood_t")["included"] + gt_pool("0xbad", "0xbad_t")["included"]
                        + gt_pool("0xyoung", "0xyoung_t")["included"],
        },
        "contract_addresses=0xgood_t": goplus_evm("0xgood_t"),
        "contract_addresses=0xbad_t": goplus_evm("0xbad_t", is_honeypot="1"),
    }
    cfg = {**CFG, "control_rate": 0.0}  # no random comparison picks in this test
    client = FakeClient(routes)
    rows = memescan.scan(client, cfg, random.Random(1), NOW, candidates)
    assert {r["pool_address"]: r["group"] for r in rows} == {"0xgood": "pick", "0xbad": ""}  # young: not judged
    assert not any("0xyoung_t" in u for u in client.urls)  # no security lookup wasted on it

    # Scanning again does not record the same pools twice.
    assert memescan.scan(FakeClient(routes), cfg, random.Random(1), NOW, candidates) == []

    later = NOW + pd.Timedelta(hours=25)
    price_now = gt_pool("0xgood", "0xgood_t", price=0.002)
    tracked = memescan.track(FakeClient({"pools/multi": price_now}), cfg, later, candidates, checks)
    assert {r["checkpoint"] for r in tracked} == {"1h", "24h"}  # 7d not due yet
    assert all(r["gross_return"] == pytest.approx(1.0) for r in tracked)

    memescan.report(candidates, checks)
    out = capsys.readouterr().out
    assert "picks: 1" in out and "honeypot" in out
