# Trading Agent (research + paper trading only)

A learning project: collect historical crypto and forex data, train ML models,
backtest them realistically, and paper trade with virtual money.

**Safety:** no real-money trades, no live order execution, no API keys. The
crypto client is created without credentials, so it can only read public prices.

## Setup (Windows)

1. Install Python 3.12 from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
2. Install Git from https://git-scm.com/download/win.
3. In a new terminal:

```powershell
cd $HOME\trading-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Stage 1: collecting data

```powershell
# Crypto: downloads every symbol listed in config.toml (re-run later to top up)
python -m trading_agent.download crypto

# Forex: download free M1 "Generic ASCII" files from https://www.histdata.com,
# unzip them, then import:
python -m trading_agent.download forex --symbol EURUSD --file path\to\DAT_ASCII_EURUSD_M1_2024.csv
```

Files are saved in `data/raw/<market>/`, for example `data/raw/crypto/BTC-USDT_1h.csv`.

## Stage 2: baseline model and backtesting

```powershell
# Try it without any downloaded data (random prices, so expect no edge):
python -m trading_agent.experiment --demo

# On data you have collected:
python -m trading_agent.experiment --market crypto --symbol BTC/USDT
python -m trading_agent.experiment --market forex --symbol EURUSD
```

This trains a logistic-regression model on the oldest 70% of the data, then
compares it on the unseen newest 30% with an SMA-crossover rule and buy-and-hold.
Costs and risk rules are set in `config.toml`.

(This README gets a full walkthrough in Stage 7.)
