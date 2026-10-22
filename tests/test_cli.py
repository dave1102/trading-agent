"""End-to-end checks of the command-line tools, using fake data and temporary folders."""

import sys
from dataclasses import fields

import pandas as pd
import pytest

from trading_agent import download, evaluate, experiment, paper_trade
from trading_agent.backtest.engine import Costs
from trading_agent.config import load_config
from trading_agent.market_data.storage import data_path, load_ohlcv
from trading_agent.market_data.synthetic import random_walk_ohlcv
from tests.test_crypto import FakeExchange


def run_cli(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()


def test_config_file_is_valid():
    cfg = load_config()
    cost_fields = {f.name for f in fields(Costs)}
    for market in ("crypto", "forex"):
        assert cfg[market]["symbols"], f"{market} needs at least one symbol"
        assert set(cfg["costs"][market]) == cost_fields
    assert 0 < cfg["risk"]["position_fraction"] <= 1
    assert 0 < cfg["paper"]["position_fraction"] <= 1
    assert 0.5 <= cfg["model"]["threshold"] < 1
    assert 0 < cfg["model"]["test_fraction"] < 1


def test_download_crypto_saves_and_resumes(monkeypatch, tmp_path, capsys):
    fake = FakeExchange(300)
    monkeypatch.setattr(download.crypto, "make_exchange", lambda exchange_id: fake)
    monkeypatch.setattr(download, "data_path", lambda m, s, t: data_path(m, s, t, base_dir=tmp_path))
    monkeypatch.setattr(download.crypto.time, "time", lambda: 1_704_067_200 + 300 * 3600)  # after the last candle

    run_cli(monkeypatch, download, "crypto", "--symbols", "BTC/USDT", "--start", "2024-01-01")
    saved = load_ohlcv(tmp_path / "crypto" / "BTC-USDT_1h.csv")
    assert len(saved) == 300

    run_cli(monkeypatch, download, "crypto", "--symbols", "BTC/USDT")  # resume: no duplicates
    assert len(load_ohlcv(tmp_path / "crypto" / "BTC-USDT_1h.csv")) == 300
    assert "300 candles" in capsys.readouterr().out


def test_download_forex_file_import(monkeypatch, tmp_path):
    csv = tmp_path / "eurusd.csv"
    lines = [f"20240102 {h:02d}{m:02d}00;1.1;1.1002;1.0998;1.1001;0" for h in range(17, 20) for m in range(60)]
    csv.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(download, "data_path", lambda m, s, t: data_path(m, s, t, base_dir=tmp_path))

    run_cli(monkeypatch, download, "forex", "--symbol", "EURUSD", "--file", str(csv))
    hourly = load_ohlcv(tmp_path / "forex" / "EURUSD_1h.csv")
    assert len(hourly) == 3  # three hours of 1-minute data -> three hourly candles


def test_experiment_demo_prints_comparison(monkeypatch, capsys):
    run_cli(monkeypatch, experiment, "--demo")
    out = capsys.readouterr().out
    for strategy in ("ml_baseline", "sma_crossover", "buy_and_hold"):
        assert strategy in out


def test_evaluate_prints_both_models(monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "load_ohlcv", lambda path: random_walk_ohlcv(n=3000, seed=13))
    run_cli(monkeypatch, evaluate, "--market", "crypto", "--symbol", "BTC/USDT", "--folds", "3")
    out = capsys.readouterr().out
    assert "Averages per model" in out and "logistic" in out and "gboost" in out


@pytest.fixture
def paper_env(monkeypatch, tmp_path):
    data = {"AAA": random_walk_ohlcv(n=3000, seed=14), "BBB": random_walk_ohlcv(n=3000, seed=15)}
    monkeypatch.setattr(paper_trade, "ACCOUNTS_DIR", tmp_path)
    monkeypatch.setattr(paper_trade, "data_path", lambda market, symbol, tf: symbol)
    monkeypatch.setattr(paper_trade, "load_ohlcv", lambda symbol: data[symbol])
    return tmp_path


def test_paper_trade_runs_and_resumes(monkeypatch, capsys, paper_env):
    args = ["--market", "crypto", "--symbols", "AAA", "BBB", "--strategy", "sma", "--start", "2024-02-01"]
    run_cli(monkeypatch, paper_trade, *args, "--end", "2024-03-01")
    first = capsys.readouterr().out
    run_cli(monkeypatch, paper_trade, *args)
    second = capsys.readouterr().out

    first_count = int(first.split("Processed ")[1].split(" ")[0].replace(",", ""))
    second_count = int(second.split("Processed ")[1].split(" ")[0].replace(",", ""))
    assert first_count > 0 and second_count > 0
    trades = pd.read_csv(paper_env / "crypto_sma" / "trades.csv")
    assert {"open", "close"} <= set(trades["action"])


def test_paper_trade_ml_strategy(monkeypatch, capsys, paper_env):
    run_cli(monkeypatch, paper_trade, "--market", "crypto", "--symbols", "AAA", "--start", "2024-03-01")
    assert "Equity:" in capsys.readouterr().out


def test_paper_trade_refuses_mismatched_resume(monkeypatch, paper_env):
    run_cli(monkeypatch, paper_trade, "--market", "crypto", "--symbols", "AAA", "--strategy", "sma",
            "--start", "2024-02-01", "--account", "demo")
    with pytest.raises(SystemExit, match="was started with"):
        run_cli(monkeypatch, paper_trade, "--market", "crypto", "--symbols", "AAA", "BBB", "--strategy", "sma",
                "--start", "2024-02-01", "--account", "demo")
    # --reset starts over instead of refusing
    run_cli(monkeypatch, paper_trade, "--market", "crypto", "--symbols", "AAA", "BBB", "--strategy", "sma",
            "--start", "2024-02-01", "--account", "demo", "--reset")
