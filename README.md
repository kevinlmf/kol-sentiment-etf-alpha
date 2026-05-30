# KOL-Sentiment-Signal-Alpha

KOL tweet sentiment signals for sector ETF returns, with lagged backtests.

Full write-up: **[project_overview.pdf](project_overview.pdf)**

---

## Idea

**Question:** Can discussions from tech / crypto KOLs on X predict short-horizon moves in **sector proxy ETFs**?

**Hypothesis:** When **more KOLs** talk about an industry and sentiment is **bearish**, the market is often in a **crowded panic** state; the next-day ETF move is more likely to **revert** (contrarian logic).

**Approach:** An interpretable pipeline—not an end-to-end black box:

```text
tweets → NLP sentiment → post-level factors → daily aggregation → IC scan → lag1 long/short → backtest
```

**Primary signal:** `sig_kol_breadth_contrarian` (participating KOL count × sentiment reversal × influence)

**Evaluation:** Signal on day T → PnL on day T+1 (`ind_ret_1d`), 5bp one-way turnover cost, no same-day leakage.

---

## Pipeline

```text
users.csv (119 KOLs)
    │
    ▼
data_mining.py      industry tags, Twikit fetch, kol_master
    │
    ▼
data_cleaning.py    dedupe / bot filter, align ETF returns (5m/1h/1d/5d)
    │
    ▼
nlp.py              lexicon + SnowNLP + optional Transformer scores
    │
    ▼
signal_testing.py   40+ candidate factors, IC scan, multi-horizon matrix, ML baseline
    │
    ▼
backtesting.py      lag1 long/short, equity / Sharpe / drawdown, benchmarks
```

| Stage | Script | Key outputs |
|-------|--------|-------------|
| Mining | `data_mining.py` | `history_tweets.csv`, `kol_ranked.csv` |
| Cleaning | `data_cleaning.py` | `tweet_events_clean.csv` |
| NLP | `nlp.py` | `sentiment_score_nlp` |
| Signals | `signal_testing.py` | `signal_best_*.csv`, multi-horizon IC matrix |
| Backtest | `backtesting.py` | equity curves, benchmark comparison |

---

## Headline results (1d lag1, exploratory sample)

| Metric | All-industry primary | Per-industry 1d routing |
|--------|----------------------|-------------------------|
| Combo return | +6.2% | +11.8% |
| Sharpe | 2.55 | 7.33* |
| IC | 0.255 | 0.263 |

\* ~51 combo days; Sharpe / return are POC-only, not production stats.

Best per industry (1d): `ai_tech` → unanimity contrarian; `crypto` → bear_crowd_strict; `ai_tools` → rank1 momentum.

---

## Quick start

```bash
git clone https://github.com/kevinlmf/kol-sentiment-etf-alpha
pip install -r requirements.txt
bash run_pipeline.sh
```

**Common commands:**

```bash
python3 signal_testing.py --scan --ret-col ind_ret_1d --lag 1
python3 signal_testing.py --scan-multi-horizon
python3 backtesting.py --compare-benchmarks --all --mode lag1
python3 backtesting.py --backtest-multi-horizon
python3 backtesting.py --all --signal sig_kol_breadth_contrarian --mode lag1
```

---

## Data notes

- KOL list: `users.csv` (configure Twikit cookies to fetch tweets)
- `data/import/twikit_cookies.json` is gitignored—do not commit
