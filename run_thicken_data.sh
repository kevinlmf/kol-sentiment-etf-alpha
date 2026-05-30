#!/usr/bin/env bash
# 加厚数据与标签：时间 × KOL × ind_ret_1d
# 用法:
#   ./run_thicken_data.sh              # 修复时间 + 清洗 + 全量抓取(119人×90天) + 标签
#   SKIP_FETCH=1 ./run_thicken_data.sh # 只修复/清洗/补收益，不抓推文（快）
set -euo pipefail
cd "$(dirname "$0")"

DAYS="${DAYS:-90}"
SLEEP_USER="${SLEEP_USER:-8}"
MAX_PER_USER="${MAX_PER_USER:-80}"
MAX_PAGES="${MAX_PAGES:-2}"

echo "=============================================="
echo "  数据加厚流水线"
echo "  KOL: 119 | 窗口: ${DAYS} 天 | 每人≤${MAX_PER_USER} 条 | 翻页 ${MAX_PAGES}"
echo "=============================================="

echo ""
echo ">>> [1/6] 修复 history tweet_time (Snowflake)"
python3 data_mining.py --repair-history

echo ""
echo ">>> [2/6] 行业分类写回 users.csv"
python3 data_mining.py --classify-only --top-k 5

if [[ "${SKIP_FETCH:-0}" != "1" ]]; then
  echo ""
  echo ">>> [3/6] 抓取全部 KOL（加深翻页，merge 去重）"
  echo "    预计 1~3 小时，可 Ctrl+C 后 --resume 续跑"
  python3 data_mining.py --fetch-twikit --all-kols \
    --days "$DAYS" --resume --refetch-users --retry-failed \
    --max-per-user "$MAX_PER_USER" --max-pages "$MAX_PAGES" \
    --sleep-user "$SLEEP_USER" --sleep-page 1.5
else
  echo ""
  echo ">>> [3/6] 跳过抓取 (SKIP_FETCH=1)"
fi

echo ""
echo ">>> [4/6] 清洗 + 收益对齐"
python3 data_cleaning.py

echo ""
echo ">>> [5/6] NLP 情绪"
python3 nlp.py

echo ""
echo ">>> [6/6] 补全 ind_ret_*（含 pending 落地）"
python3 data_cleaning.py --returns-only

echo ""
python3 - <<'PY'
import pandas as pd
from pathlib import Path

def leg_count(ev):
    v = pd.to_numeric(ev["ind_ret_1d"], errors="coerce")
    ok = ev[v.notna()].copy()
    ok["d"] = pd.to_datetime(ok["tweet_time"], utc=True, errors="coerce").dt.date
    return len(ok.groupby(["d", "industry"]))

root = Path(".")
u = pd.read_csv("users.csv")
h = pd.read_csv("data/import/history_tweets.csv")
c = pd.read_csv("data/import/history_tweets_clean.csv")
ev = pd.read_csv("data/outputs/tweet_events_clean.csv")
t = pd.to_datetime(h["tweet_time"], utc=True, errors="coerce")
v = pd.to_numeric(ev["ind_ret_1d"], errors="coerce")
print("=== 数据加厚结果 ===")
print(f"  种子 KOL:        {len(u)}")
print(f"  抓取推文:        {len(h)} ({h['username'].nunique()} 人)")
print(f"  tweet_time 有效: {t.notna().sum()}/{len(h)}")
print(f"  时间跨度:        {t.min()} .. {t.max()}")
print(f"  清洗事件:        {len(ev)} ({ev['username'].nunique()} 人)")
print(f"  ind_ret_1d 有效: {v.notna().sum()}/{len(ev)} ({100*v.notna().mean():.1f}%)")
print(f"  IC 可用腿(日×行业): {leg_count(ev)}")
pend = ev.get("align_note", pd.Series(dtype=str)).astype(str).str.contains("ret_1d_pending").sum()
print(f"  仍 pending:      {pend}")
print("")
print("下一步: python3 signal_testing.py --scan --ret-col ind_ret_1d --lag 1")
PY

echo ""
echo "完成。"
