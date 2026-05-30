#!/usr/bin/env python3
"""
Step 3 — NLP 情感信号（中文推文 → 可交易情绪分）

输入:  data/import/history_tweets_clean.csv
输出:  回写 clean + tweet_events_clean（新增 sentiment_nlp / sentiment_score_nlp）

用法:
  python3 nlp.py
  python3 nlp.py --transformers
  python3 nlp.py --then-mining
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

from data_cleaning import (
    EVENTS_CLEAN_CSV,
    HISTORY_CLEAN_CSV,
    repair_tweet_times,
    select_event_rows,
)

# 中文金融 / 社媒情绪词（比 data_mining 词典更全）
_POS_ZH = (
    "利好", "看涨", "上涨", "暴涨", "牛市", "牛", "超预期", "beat", "strong",
    "买入", "加仓", "抄底", "起飞", "破新高", "moon", "bull", "bullish", "gem",
    "赚", "吃肉", "封神", "无敌", "起飞", "冲", "稳了", "win",
)
_NEG_ZH = (
    "利空", "看跌", "下跌", "暴跌", "熊市", "熊", "不及预期", "miss", "weak",
    "卖出", "减仓", "崩盘", "腰斩", "爆仓", "crash", "bear", "bearish", "rug",
    "亏", "割肉", "凉凉", "崩", "完犊子", "危险", "跑路", "panic", "fear",
)
_CASHTAG = re.compile(r"\$[A-Za-z]{1,6}\b")
_URL = re.compile(r"https?://\S+")


def _prep(text: str) -> str:
    t = str(text or "")
    t = _URL.sub("", t)
    t = _CASHTAG.sub("", t)
    return t.strip()


def lexicon_score(text: str) -> tuple[str, float]:
    t = _prep(text).lower()
    if not t:
        return "neutral", 0.0
    p = sum(1 for w in _POS_ZH if w in t)
    n = sum(1 for w in _NEG_ZH if w in t)
    if p == n == 0:
        return "neutral", 0.0
    score = (p - n) / max(p + n, 1)
    score = max(-1.0, min(1.0, score))
    if score > 0.08:
        return "positive", round(score, 4)
    if score < -0.08:
        return "negative", round(score, 4)
    return "neutral", round(score, 4)


def snownlp_score(text: str) -> float | None:
    try:
        from snownlp import SnowNLP
    except ImportError:
        return None
    t = _prep(text)
    if len(t) < 4:
        return None
    try:
        prob = float(SnowNLP(t).sentiments)
        return round((prob - 0.5) * 2.0, 4)
    except Exception:
        return None


def _load_transformer_pipeline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None,
        device=-1,
    )


def transformer_scores(texts: list[str], pipe) -> list[float]:
    cleaned = [_prep(t) or " " for t in texts]
    label_map = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0,
        "pos": 1.0,
        "neg": -1.0,
    }
    out: list[float] = []
    batch = 32
    for i in range(0, len(cleaned), batch):
        chunk = cleaned[i : i + batch]
        try:
            preds = pipe(chunk, truncation=True, max_length=256)
        except Exception:
            preds = [[] for _ in chunk]
        for pred in preds:
            if not pred:
                out.append(0.0)
                continue
            if isinstance(pred, dict):
                pred = [pred]
            score = 0.0
            for item in pred:
                lab = str(item.get("label", "")).lower()
                w = float(item.get("score", 0))
                score += label_map.get(lab, 0.0) * w
            out.append(round(max(-1.0, min(1.0, score)), 4))
    return out


def composite_scores(
    texts: list[str],
    *,
    use_transformers: bool = False,
) -> tuple[list[str], list[float]]:
    pipe = None
    tfm: list[float] | None = None
    if use_transformers:
        try:
            pipe = _load_transformer_pipeline()
            tfm = transformer_scores(texts, pipe)
            print(f"  transformers: {pipe.model.config.name_or_path}")
        except Exception as exc:
            print(f"  transformers 不可用，回退词典+SnowNLP: {exc}")
            tfm = None

    labels: list[str] = []
    scores: list[float] = []
    for i, text in enumerate(texts):
        lex_l, lex_s = lexicon_score(text)
        snow_s = snownlp_score(text)
        parts: list[tuple[float, float]] = [(lex_s, 0.45)]
        if snow_s is not None:
            parts.append((snow_s, 0.25))
        if tfm is not None:
            parts.append((tfm[i], 0.30))
        else:
            parts.append((lex_s, 0.30))

        wsum = sum(w for _, w in parts)
        score = sum(s * w for s, w in parts) / wsum if wsum else 0.0
        score = round(max(-1.0, min(1.0, score)), 4)
        if score > 0.08:
            lab = "positive"
        elif score < -0.08:
            lab = "negative"
        else:
            lab = "neutral"
        labels.append(lab)
        scores.append(score)
    return labels, scores


def enrich_dataframe(df: pd.DataFrame, *, use_transformers: bool = False) -> pd.DataFrame:
    col = "text_clean" if "text_clean" in df.columns else "text"
    texts = df[col].fillna("").astype(str).tolist()
    labels, scores = composite_scores(texts, use_transformers=use_transformers)
    out = df.copy()
    out["sentiment_nlp"] = labels
    out["sentiment_score_nlp"] = scores
    return out


def run_nlp(
    *,
    use_transformers: bool = False,
    then_mining: bool = False,
) -> pd.DataFrame:
    if not HISTORY_CLEAN_CSV.exists():
        raise SystemExit(f"缺少 {HISTORY_CLEAN_CSV}，请先 python3 data_cleaning.py")

    df = repair_tweet_times(pd.read_csv(HISTORY_CLEAN_CSV))
    print(f"NLP: {len(df)} 条 <- {HISTORY_CLEAN_CSV}")
    enriched = enrich_dataframe(df, use_transformers=use_transformers)
    enriched.to_csv(HISTORY_CLEAN_CSV, index=False, encoding="utf-8-sig")

    events = select_event_rows(enriched, drop_bots=True)
    events.to_csv(EVENTS_CLEAN_CSV, index=False, encoding="utf-8-sig")

    std = enriched["sentiment_score_nlp"].std()
    nz = (enriched["sentiment_score_nlp"] != 0).sum()
    print(f"  sentiment_score_nlp: 非零 {nz}/{len(enriched)} | std={std:.4f}")
    print(f"  -> {HISTORY_CLEAN_CSV}")
    print(f"  事件子集: {len(events)} 条 -> {EVENTS_CLEAN_CSV}")

    if then_mining:
        from data_mining import build_daily_factor, build_master_from_clean, run_backtest

        master = build_master_from_clean()
        build_daily_factor(master)
        run_backtest(master)

    return enriched


def main() -> None:
    p = argparse.ArgumentParser(description="Step3: NLP 情感打分")
    p.add_argument(
        "--transformers",
        action="store_true",
        help="叠加 HuggingFace 多语言情感模型（需 pip install transformers torch）",
    )
    p.add_argument(
        "--then-mining",
        action="store_true",
        help="完成后自动 python3 data_mining.py --master --from-clean",
    )
    args = p.parse_args()
    run_nlp(use_transformers=args.transformers, then_mining=args.then_mining)


if __name__ == "__main__":
    main()
