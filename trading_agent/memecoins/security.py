"""Scam checks from GoPlus Security (all chains) and RugCheck (Solana). Public APIs, no keys.

Every field is None when the source could not tell us. The screener treats
"unknown" as a failure: a token we cannot check is a token we do not buy.
"""

from dataclasses import dataclass, field

GOPLUS = "https://api.gopluslabs.io/api/v1"
RUGCHECK = "https://api.rugcheck.xyz/v1"
EVM_CHAIN_IDS = {"eth": 1, "bsc": 56, "base": 8453}
BURN_ADDRESSES = {"0x0000000000000000000000000000000000000000", "0x000000000000000000000000000000000000dead"}


@dataclass
class SecurityReport:
    found: bool = False  # did the security service know this token at all?
    honeypot: bool | None = None  # cannot sell, or the owner can take your tokens
    buy_tax: float | None = None  # fractions: 0.05 = 5%
    sell_tax: float | None = None
    mintable: bool | None = None  # creator can create more tokens
    freezable: bool | None = None  # creator can freeze your tokens
    top10_share: float | None = None  # supply held by the 10 biggest ordinary wallets
    lp_locked_share: float | None = None  # liquidity that is locked or burned, so it cannot be pulled
    dangers: list[str] = field(default_factory=list)  # named red flags


def _flag(value) -> bool | None:
    return None if value in (None, "") else str(value) == "1"


def _tax(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_goplus_evm(payload: dict | None, address: str) -> SecurityReport:
    info = ((payload or {}).get("result") or {}).get(address.lower())
    if not info:
        return SecurityReport()
    report = SecurityReport(found=True)
    for key, label in [
        ("is_honeypot", "honeypot"), ("cannot_sell_all", "cannot sell all"),
        ("owner_change_balance", "owner can change balances"), ("hidden_owner", "hidden owner"),
        ("can_take_back_ownership", "owner can reclaim control"), ("transfer_pausable", "transfers can be paused"),
        ("selfdestruct", "contract can self-destruct"), ("honeypot_with_same_creator", "creator made honeypots before"),
    ]:
        if _flag(info.get(key)):
            report.dangers.append(label)
    report.honeypot = _flag(info.get("is_honeypot")) or _flag(info.get("cannot_sell_all")) or False
    report.buy_tax = _tax(info.get("buy_tax"))
    report.sell_tax = _tax(info.get("sell_tax"))
    report.mintable = _flag(info.get("is_mintable"))
    report.freezable = False  # EVM tokens have no freeze authority; pausing is covered above

    pairs = {d.get("pair", "").lower() for d in info.get("dex") or []}
    ordinary = [
        h for h in info.get("holders") or []
        if not _flag(h.get("is_locked")) and h.get("address", "").lower() not in BURN_ADDRESSES | pairs
    ]
    if info.get("holders"):
        report.top10_share = sum(_tax(h.get("percent")) or 0 for h in ordinary[:10])
    lp = info.get("lp_holders")
    if lp:
        report.lp_locked_share = sum(
            _tax(h.get("percent")) or 0 for h in lp
            if _flag(h.get("is_locked")) or h.get("address", "").lower() in BURN_ADDRESSES
        )
    return report


def parse_solana(goplus: dict | None, rugcheck: dict | None, address: str) -> SecurityReport:
    info = ((goplus or {}).get("result") or {}).get(address)
    if not info or not rugcheck:
        return SecurityReport(found=bool(info or rugcheck))
    report = SecurityReport(found=True)

    def status(key):
        return _flag((info.get(key) or {}).get("status"))

    report.mintable = status("mintable")
    report.freezable = status("freezable")
    if status("balance_mutable_authority"):
        report.dangers.append("owner can change balances")
    if _flag(info.get("non_transferable")):
        report.dangers.append("token cannot be transferred")
    fee = _tax(((info.get("transfer_fee") or {}).get("current_fee_rate") or {}).get("fee_rate")) or 0.0
    if fee > 0 or status("transfer_fee_upgradable"):
        report.dangers.append("transfer fee (or creator can add one)")
    report.buy_tax = report.sell_tax = fee
    report.dangers += [r.get("name", "?") for r in rugcheck.get("risks") or [] if r.get("level") == "danger"]
    report.honeypot = any("cannot" in d or "honeypot" in d.lower() for d in report.dangers)
    locked = rugcheck.get("lpLockedPct")
    report.lp_locked_share = None if locked is None else float(locked) / 100
    return report


def check_token(client, network: str, address: str) -> SecurityReport:
    if network == "solana":
        goplus = client.get_json(f"{GOPLUS}/solana/token_security?contract_addresses={address}")
        rugcheck = client.get_json(f"{RUGCHECK}/tokens/{address}/report/summary")
        return parse_solana(goplus, rugcheck, address)
    chain_id = EVM_CHAIN_IDS[network]
    payload = client.get_json(f"{GOPLUS}/token_security/{chain_id}?contract_addresses={address}")
    return parse_goplus_evm(payload, address)
