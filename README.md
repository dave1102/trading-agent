# Trading Agent: research and paper trading for crypto and forex

A learning project that builds a machine-learning trading agent the careful way:
collect real historical prices, train simple models, test them with realistic
costs on data they have never seen, and paper trade with virtual money.

> **Safety: this project cannot trade real money.**
> - It never places orders. The paper-trading code has no network access at all
>   (a test, `tests/test_paper.py`, fails if anyone adds it).
> - It uses no API keys or passwords. The crypto client is created without
>   credentials, so it can only read public prices (also checked by a test).
> - All balances are simulated numbers in files on your computer.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Setup on Windows](#2-setup-on-windows)
3. [Project layout](#3-project-layout)
4. [Step-by-step walkthrough](#4-step-by-step-walkthrough)
5. [Reading the results](#5-reading-the-results)
6. [Settings (config.toml)](#6-settings-configtoml)
7. [What we have found so far](#7-what-we-have-found-so-far)
8. [Google Colab](#8-google-colab)
9. [Troubleshooting](#9-troubleshooting)
10. [Running the tests](#10-running-the-tests)
11. [Three ideas that keep backtests honest](#11-three-ideas-that-keep-backtests-honest)

---

## 1. How it works

```
 Download prices  ->  Clean & store  ->  Features  ->  Model  ->  Signal  ->  Backtest / Paper trade
 (CCXT, HistData)     (data/raw/)        (past only)   (sklearn)   (1/0/-1)    (costs, risk rules, logs)
```

- **Candles**: each row of price data is one *candle*: the open, high, low and close
  price over a period (1 hour, 4 hours, 1 day) plus the volume traded.
- **Features**: numbers computed from past candles only, such as recent returns,
  volatility, RSI and distance from moving averages.
- **Model**: predicts the chance that the price rises over the next few candles.
- **Signal**: if the model is confident enough, go long (1); otherwise stay out (0).
  Shorting (-1) is optional.
- **Backtest**: replays history candle by candle. A decision made at a candle's close
  is filled at the **next** candle's open, and every fill pays fees, spread and slippage.

## 2. Setup on Windows

You need **Python 3.12** and **Git**.

1. Install Python from <https://www.python.org/downloads/> and tick
   **"Add python.exe to PATH"** on the first installer screen.
2. Install Git from <https://git-scm.com/download/win>.
3. Open a **new** terminal (PowerShell) and run:

```powershell
cd $HOME\trading-agent
python -m venv .venv                    # a private folder for this project's libraries
.\.venv\Scripts\Activate.ps1            # switch this terminal to use it
pip install -r requirements.txt
pytest                                  # should end with "passed"
```

Each time you open a new terminal, run `.\.venv\Scripts\Activate.ps1` again. You
will see `(.venv)` at the start of the prompt when it is active.

If PowerShell says *"running scripts is disabled on this system"*, run this once
and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In VS Code, press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one
in `.venv`, so the editor sees the installed libraries too.

## 3. Project layout

```
trading-agent/
├── config.toml              Settings: symbols, costs, risk limits, model options
├── requirements.txt         Python libraries
├── trading_agent/
│   ├── download.py          Command: get historical data
│   ├── experiment.py        Command: one train/test comparison
│   ├── evaluate.py          Command: walk-forward model comparison
│   ├── sweep.py             Command: try many settings + locked-away final test
│   ├── paper_trade.py       Command: paper trade on replayed candles
│   ├── market_data/         Downloading (CCXT, HistData), importing, cleaning, storage
│   ├── features.py          Model inputs (past data only)
│   ├── models/baseline.py   Logistic regression and gradient boosting
│   ├── strategies.py        Simple benchmark strategies
│   ├── evaluation.py        Walk-forward, holdout test, 200-day trend filter
│   ├── backtest/            Backtester and performance metrics
│   └── paper/               Virtual account, risk limits, trade logs
├── notebooks/colab_training.ipynb   Google Colab notebook
├── tests/                   Automated tests
├── data/                    Downloaded prices and results (not in Git)
└── paper_accounts/          Virtual accounts and trade logs (not in Git)
```

## 4. Step-by-step walkthrough

Run every command from the `trading-agent` folder with the `.venv` active.

### Step 1: Get historical data

**Forex** (EUR/USD, GBP/USD, USD/JPY), free 1-minute data from HistData.com,
converted to hourly candles:

```powershell
python -m trading_agent.download histdata --years 2019-2025
```

Years already downloaded are skipped. HistData only publishes the current year
month by month, so ask for finished years only.

**Crypto** (BTC, ETH, SOL; add more in `config.toml`):

```powershell
python -m trading_agent.download crypto                         # exchange from config.toml (MEXC)
python -m trading_agent.download crypto --symbols XRP/USDT ADA/USDT --start 2022-01-01
python -m trading_agent.download crypto --exchange binanceus    # another exchange
```

Run it again later to add only the newest candles. If your network blocks
exchanges, download in Colab instead (section 8) and copy the CSV files into
`data\raw\crypto\`.

**Your own CSV file** (any file with timestamp, open, high, low, close columns):

```powershell
python -m trading_agent.download forex --symbol EURUSD --file path\to\file.csv --format generic
```

Every download prints a summary such as
`EURUSD: 42,778 candles ... 1061 gaps`. Gaps at weekends are normal for forex;
many gaps in crypto (which trades 24/7) would mean a data problem.

### Step 2: A first experiment

```powershell
python -m trading_agent.experiment --demo                              # random prices, no data needed
python -m trading_agent.experiment --market forex --symbol EURUSD
```

This trains on the oldest 70% of the data and compares three strategies on the
newest 30%: the ML model, a moving-average crossover rule, and buy-and-hold.

The `--demo` run uses random prices, so the model *should* show no edge. If it
ever looked brilliant on random data, something would be wrong.

### Step 3: Walk-forward testing (more trustworthy)

```powershell
python -m trading_agent.evaluate --market crypto --symbol BTC/USDT
```

This trains on the past and tests on the next slice of time, five times in a row,
for both models. A model has to do well across most slices, not just one.

### Step 4: Sweep settings, then a final test on hidden data

```powershell
python -m trading_agent.sweep --market crypto     # hides 2026
python -m trading_agent.sweep --market forex      # hides 2025
```

This tries 28 combinations of candle size, prediction horizon, confidence
threshold and a 200-day trend filter on the older data. It picks the single best
combination across all symbols, then tests it **once** on the hidden recent
period. That last table, the **VERDICT**, is the result to believe. Full results
are saved to `data\results\sweep_<market>.csv` (opens in Excel).

### Step 4b: Does extra information improve the confident trades?

```powershell
python -m trading_agent.download funding                      # futures funding rates (Deribit, BTC and ETH)
python -m trading_agent.confidence                            # 1-hour candles, 1 hour ahead
python -m trading_agent.confidence --timeframe 1d --horizon 5
```

This trains on data before 2025 and compares three input sets on the unseen
period after it: price only, plus market inputs (what BTC is doing, time of day,
day of week), plus funding rates. For each confidence level it pools the trades of
all coins and shows win rate, profit per trade and profit factor.

### Step 4c: Multi-coin trend following (the first strategy that passed)

```powershell
python -m trading_agent.download crypto --start 2022-01-01      # all coins in [portfolio] need data
python -m trading_agent.trend
```

Instead of predicting the next hour, this holds coins that have been rising and
steps aside from coins that have been falling, across 13 coins at once:

- **Trend score** per coin: the share of its 20-, 60- and 120-day returns that are
  positive (0, ⅓, ⅔ or 1).
- **Sizing**: calmer coins get bigger positions; at most 25% in one coin and 100% in total.
- **Weekly rebalance** (Sunday close, traded at Monday's open), with full costs.

All settings are common textbook values fixed before testing. The pass rule was
also fixed in advance: beat an equal-weight basket of all coins on Sharpe ratio
**and** have a worst drawdown at least a third smaller, in both periods and in
every coin group (the original 3, the 10 new coins, all 13).

### Step 4d: New meme coin scanner (Solana, Base, BNB Chain, Ethereum)

```powershell
python -m trading_agent.memescan run --every 15   # leave running; Ctrl+C to stop
python -m trading_agent.memescan report           # check progress any time (in another terminal)
```

Paper only: it reads public data and never holds a wallet or keys. Each round it:

1. **Finds** new and trending pools on each chain (GeckoTerminal).
2. **Screens** them. Market rules: 15 minutes to 24 hours old, at least $20,000
   liquidity, 50+ trades in the last hour, real sellers, not an established coin,
   not copying a top-250 coin's ticker. Scam checks (GoPlus Security, plus RugCheck
   on Solana): no honeypot, taxes at most 5%, creator cannot mint more or freeze
   tokens, top 10 wallets hold at most 30%, at least 90% of liquidity locked or
   burned. **Anything that cannot be checked fails.**
3. **Journals** every pick, plus a random 20% of rejected pools as a comparison group.
4. **Tracks** each one 1 hour, 1 day and 7 days later, after estimated costs for a
   $100 paper trade (DEX fee, price impact, token tax, gas). If liquidity falls 90%
   or the pool disappears, it counts as a rug pull: -100%.

Files are in `data\memecoins\`: `candidates.csv` (every pool judged, with reasons)
and `checks.csv` (results). All rules are in `[memecoins]` in `config.toml`.

**Version 2 (from 30 October 2026): three entry strategies.** Each is an independent
paper trader with its own entry time and price, and all share the same safety checks:

| Strategy | Buys |
|---|---|
| `trending` | the week-1 rules: 15 min to 24 h old, $20k+ liquidity, 50+ trades in the last hour |
| `early` | 15–60 min old, $10k+ liquidity, 30+ trades in the last hour |
| `not_pumped` | like `trending`, but rose less than 50% in the last hour |

Entries are checked after 5 m, 15 m, 30 m, 1 h, 24 h and 7 d. The report also tests
a **quick exit** for each strategy: sell at the first check at +30% or -20%,
otherwise at 1 hour. The comparison group is a random 20% of all listed pools with
at least $1,000 liquidity; `control_liquid` is the part with $20k+, the fair
comparison. Week-1 results are kept as `week1_*.csv`.

**Where it runs:** on the Hostinger server, in a Docker container called `memescan`
(folder `/opt/trading-agent`, data in `/opt/trading-agent/data/memecoins`). It runs
24/7, restarts by itself, and is capped at 300 MB of memory and half a CPU core. Control
it from the PC by double-clicking these (they use the `vps` SSH shortcut):

| File | What it does |
|---|---|
| `scanner_status.cmd` | Says whether it is running and shows the latest log lines |
| `stop_scanner.cmd` / `start_scanner.cmd` | Stops / starts it |
| `weekly_report.cmd` | Fetches the report into `data\memecoins\report_YYYY-MM-DD.txt` |

To update the server after a code change: commit, then
`git archive --format=tar.gz -o src.tar.gz HEAD`, copy `src.tar.gz` and
`deploy\vps_deploy.sh` to `/opt/trading-agent/` on the server, and run
`bash /opt/trading-agent/vps_deploy.sh` there. The data folder is kept.

To run it on the PC instead: `python -m trading_agent.memescan run --every 15`.

**Weekly report:** `weekly_report.cmd` saves the report to
`data\memecoins\report_YYYY-MM-DD.txt` and opens it in Notepad. On this PC a Windows
scheduled task named **"Meme scanner weekly report"** runs it every Friday at 12:00
(or at the next sign-in if the PC was off). Manage or delete it in **Task Scheduler**
(Start menu → type "Task Scheduler" → Task Scheduler Library). The report ends with a
plain-English **VERDICT**.

Only one scanner can run at a time, so starting a
second copy by hand simply exits. Your PC must be awake for it to scan (set sleep to
"Never" while collecting data).

The report answers two questions: do picks make money after costs, and do they do
better than the comparison group (that is, do the rules help)? Wait for 100+ picks
per checkpoint before drawing conclusions.

**Forward paper test of the trend strategy** (from 1 November 2026): on the server, a
second container `trendpaper` replays three virtual $10,000 portfolios from the fixed
start date, using only completed days of MEXC prices: `trend` (exactly the tested
settings), `basket_hold` (all 13 coins equally) and `btc_hold`. First decision at the
close of Sunday 1 November (UTC), first trade at Monday's open, then weekly. Files are
in `/opt/trading-agent/data/trend/` on the server: `equity_log.csv` (one row per day,
only ever appended), `trades.csv` and `holdings.csv`. Its results are at the top of
the Friday report. On the PC: `python -m trading_agent.trendpaper update` then
`python -m trading_agent.trendpaper report`.

### Step 5: Paper trade

```powershell
python -m trading_agent.paper_trade --market forex --start 2025-01-01
python -m trading_agent.paper_trade --market crypto --strategy sma --account sma_test
```

This replays candles one at a time into a virtual account shared by all symbols,
applying the risk limits from `[paper]` in `config.toml`. The model is trained
only on data from before `--start`.

Files are written to `paper_accounts\<account name>\`:

| File | Contents |
|---|---|
| `trades.csv` | Every simulated fill (`open`/`close`), with price, fee, profit or loss and reason; plus orders the risk rules `rejected` |
| `equity.csv` | Account value after each candle |
| `state.json` | Cash, open positions and risk flags, so the run can resume |

Run the same command again to **resume**: only new candles are processed. Use
`--reset` to start the account over, or `--account <new name>` for a separate one.
Changing the symbols or strategy of an existing account is refused, so results
never mix.

Exit reasons you may see in `trades.csv`:
`signal` (the model changed its mind), `stop_loss`, `take_profit`,
`daily_loss_limit` (lost too much today, paused until tomorrow), and
`drawdown_halt` (fell too far from the peak, stopped for good).

## 5. Reading the results

| Number | Meaning | Rule of thumb |
|---|---|---|
| `total_return` / `return` | Profit or loss over the test period, after costs | Compare with buy-and-hold over the same period |
| `cagr` | Return per year, compounded | |
| `sharpe` | Return divided by how bumpy it was, per year | Below 0: losing. Above 1 on unseen data: unusually good. Treat claims above 2 with suspicion |
| `max_drawdown` | Worst fall from a previous peak | -0.30 means at one point you were 30% below your best |
| `trades` | Completed round trips | Each one pays costs twice |
| `win_rate` | Share of trades that made money | Can be low and still profitable if wins are big |
| `profit_factor` | Money won ÷ money lost | Above 1 makes money; `inf` means no losing trades |
| `exposure` | Share of time holding a position | Low exposure can look "safe" just by doing nothing |
| `accuracy` vs `up_share` | How often the model was right, vs how often prices rose | If accuracy is not clearly above the larger of `up_share` and `1 - up_share`, the model knows little |
| `folds_profitable` | In walk-forward tests, how many slices made money | Look for most of them, not one lucky one |

## 6. Settings (config.toml)

| Section | What it controls |
|---|---|
| `[crypto]`, `[forex]` | Symbols, candle size, exchange, start date |
| `[costs.crypto]`, `[costs.forex]` | `fee_rate`, `spread`, `slippage` as fractions of price (0.001 = 0.1%). Each fill pays the fee, half the spread and the slippage |
| `[risk]` | Backtest risk rules: `position_fraction`, `stop_loss`, `take_profit`, `max_drawdown_halt` |
| `[model]` | `horizon` (candles ahead), `threshold` (confidence needed to trade), `test_fraction`, `allow_short` |
| `[paper]` | Paper-trading limits: starting cash, size per position, maximum open positions, total exposure, stop-loss, daily loss limit, drawdown halt |

For risk settings, `0` means "switched off". Keep costs pessimistic: if a strategy
only works with tiny costs, it does not work. After editing, run `pytest`, which
includes a check that the file is still valid.

## 7. What we have found so far

These are research results, not advice, and they will change as data grows.

- **Hourly crypto trading loses to costs.** With 1-hour candles the model traded
  about 230 times per test slice. Each round trip costs about 0.32% of the
  position, which turned a tiny edge (+1.8% with zero costs) into about -29%.
- **Trading less removes the cost problem, but no edge appeared.** The best sweep
  setting (daily candles, 5 days ahead) averaged +1.1% per slice, while
  buy-and-hold averaged +54%. On hidden 2026 data it returned -10.8% (BTC),
  -11.3% (ETH) and +2.6% (SOL), with accuracy below 50%. It lost less than holding
  in a falling market mostly because it was often out of the market.
- **Forex: no edge either.** The best setting hardly traded on hidden 2025 data.
- **A high win rate is easy and means little.** Buying BTC daily with a +0.5% target
  and -10% stop won 95% of trades in 2025–2026, yet lost 0.38% per trade after costs.
  Judge strategies by profit per trade and profit factor.
- **Extra information helped only a little.** With BTC/market and calendar inputs,
  the model's 65%-confidence trades on 1-hour candles improved from -0.24% to -0.02%
  per trade (win rate 45% → 52%). Only the 70% level made money per trade
  (about +0.25%, profit factor about 1.6), but with just 14–16 trades in 22 months,
  far too few to trust. Funding rates added nothing clear. On daily candles every
  input set lost money.
- **That 70% result was luck.** Tested unchanged on 10 coins it had never seen
  (XRP, ADA, DOGE, AVAX, LINK, DOT, LTC, BNB, TRX, ATOM), the same setup made
  93 trades, won 55% of them, and lost 0.25% per trade (profit factor 0.73).
  Testing a promising setup on fresh symbols is a fast way to catch a fluke.
- **Multi-coin trend following passed its pre-set test.** On all 13 coins:
  May 2022–2024 +112% (Sharpe 1.13, worst drawdown -25%) vs the basket's +89%
  (Sharpe 0.69, -59%); 2025 to now -9% (Sharpe -0.14, -23%) vs the basket's -51%
  (Sharpe -0.35, -67%). It also passed on the original 3 and on the 10 new coins
  separately, and doubling costs changed little. Against a basket scaled down to
  the same ~40% invested, it still did better (+112% vs +48%; -9% vs -18%), so the
  edge comes from timing, not just from holding less. Momentum rotation failed.
- **Limits of that result:** it is about 4.5 years of one market; it still lost
  money in 2025–2026 (just much less than holding); it won only about half of all
  weeks; and the 13 coins are ones that still exist today, which flatters
  every strategy a little (coins that collapsed, such as LUNA, are missing).
- **A more flexible model did worse.** Gradient boosting traded more, learned noise
  and lost more than logistic regression.
- **Data note:** HistData's 2023 files are missing many whole hours from February
  to July. Other years are clean.

Simple price-pattern models usually find no reliable edge in liquid markets. The
value of this project is a testing setup that tells you so honestly, before any
money is at risk.

## 8. Google Colab

Colab gives you free cloud computing and a network that can reach crypto
exchanges, which some home networks block.

1. Package the project (re-run after every code change):
   ```powershell
   git archive --format=zip -o trading-agent.zip HEAD
   ```
2. Upload `trading-agent.zip` to the top level of **My Drive**.
3. At <https://colab.research.google.com>, choose **File → Upload notebook** and pick
   `notebooks\colab_training.ipynb`.
4. Choose **Runtime → Run all**, and allow Google Drive access when asked.

Results appear under each cell. Data and results are saved in
**My Drive → trading-agent-data**, so they survive Colab restarts. To use the data on
your PC, download the CSV files from `trading-agent-data/raw/crypto/` and put them in
`data\raw\crypto\`.

Colab runs on US servers, so the notebook uses Binance.US. If that fails, change
`CRYPTO_EXCHANGE` in the notebook to `"coinbase"` or `"kraken"`.

## 9. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `could not reach binance: NetworkError` | Your network or router blocks the exchange. Download in Colab (section 8), or try a different network |
| `ExchangeNotAvailable` / HTTP 451 from Binance.US | Binance.US only serves the US. Use Colab, or a different exchange |
| `No full-year download for ... (is the year finished?)` | HistData only has full-year files for finished years |
| `FileNotFoundError: ... data\raw\...csv` | You have not downloaded that symbol yet (Step 1) |
| `Account ... was started with ...` | You changed settings for an existing paper account. Use `--reset` or a new `--account` name |
| `Not enough data for this many folds` | Too little data for walk-forward. Download more years or use `--folds 3` |
| `running scripts is disabled on this system` | See the PowerShell note in section 2 |
| `python` opens the Microsoft Store | Python is not installed or not on PATH. Reinstall and tick "Add python.exe to PATH" |
| VS Code says packages are "not installed" | VS Code is using a different Python. Select the `.venv` interpreter (section 2) |

## 10. Running the tests

```powershell
pytest            # everything (about 30 seconds)
pytest -q tests\test_backtest.py     # one file
```

The tests need no internet: they use fake exchanges, fake websites and generated
prices. Among other things they check that:

- features never use future prices;
- trades fill at the next candle's open, with costs on every fill;
- stop-loss, daily-loss and drawdown rules fire when they should;
- a paper account stopped halfway and resumed ends exactly where an uninterrupted run would;
- walk-forward and holdout tests never train on their test data;
- the paper-trading code cannot reach the network, and the crypto client has no keys.

Run `pytest` after any change. If something fails, the message names the test and
the line that broke.

## 11. Three ideas that keep backtests honest

1. **No looking ahead.** A decision at time *t* may only use information available
   at *t*. Here, features use past candles only, and trades fill at the next open.
2. **Pay realistic costs.** Fees, spread and slippage turn many "winning" strategies
   into losers, especially ones that trade often.
3. **Test on unseen data, and only once.** Every extra setting you try on the same
   data makes a lucky result more likely. Walk-forward testing plus a locked-away
   final period is the defence.
