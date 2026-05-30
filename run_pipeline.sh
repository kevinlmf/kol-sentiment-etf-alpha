#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 data_mining.py --pipeline --top-k 5 --days 30 --sleep-user 8
python3 data_cleaning.py --write-master
python3 nlp.py
python3 data_mining.py --master --from-clean
python3 signal_testing.py --ml-score --backend tfidf --target ind_ret_1d --pca-dim 32
python3 signal_testing.py --ml-updown --ret-col ind_ret_1d --train-ratio 0.7
python3 signal_testing.py --scan --ret-col ind_ret_1d --lag 1
python3 signal_testing.py --scan-multi-horizon
python3 backtesting.py --compare-benchmarks --all --ret-col ind_ret_1d --mode lag1
python3 backtesting.py --backtest-multi-horizon
python3 backtesting.py --all --signal sig_kol_breadth_contrarian --mode lag1
python3 backtesting.py --all --signal sig_ml_up --mode lag1
