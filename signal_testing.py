#!/usr/bin/env python3
"""
信号测试 — IC 扫描 / 分行业选优 / ML score / 因子诊断

用法:
  python3 signal_testing.py --scan --ret-col ind_ret_1d --lag 1
  python3 signal_testing.py --scan-multi-horizon
  python3 signal_testing.py --ml-updown --ret-col ind_ret_1d
  python3 signal_testing.py --ml-score --backend tfidf --target ind_ret_1d
  python3 signal_testing.py --diagnose --signal sig_bear_only
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from data_cleaning import EVENTS_CLEAN_CSV, OUTPUTS_DIR, repair_tweet_times
from data_mining import INDUSTRY_ETF_MAP, KOL_RANKED_CSV
from nlp import enrich_dataframe

Backend = Literal["auto", "bert", "tfidf"]


def _pick_sentiment(df: pd.DataFrame) -> str:
    if "sentiment_score_nlp" in df.columns and df["sentiment_score_nlp"].std(skipna=True) > 0:
        return "sentiment_score_nlp"
    return "sentiment_score"


def load_events() -> pd.DataFrame:
    if not EVENTS_CLEAN_CSV.exists():
        raise SystemExit(f"缺少 {EVENTS_CLEAN_CSV}，请先 data_cleaning.py + nlp.py")
    df = repair_tweet_times(pd.read_csv(EVENTS_CLEAN_CSV))
    if "sentiment_score_nlp" not in df.columns:
        df = enrich_dataframe(df, use_transformers=False)
    return df


def attach_influence_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if KOL_RANKED_CSV.exists():
        rk = pd.read_csv(KOL_RANKED_CSV)[["username", "influence_score"]]
        out = out.merge(rk, on="username", how="left", suffixes=("", "_rk"))
        if "influence_score_rk" in out.columns:
            out["influence_score"] = out["influence_score"].fillna(out["influence_score_rk"])
            out.drop(columns=["influence_score_rk"], inplace=True)
    out["views_seed"] = pd.to_numeric(out.get("views_seed"), errors="coerce")
    out["influence"] = pd.to_numeric(out.get("influence_score"), errors="coerce")
    out["influence"] = out["influence"].fillna(out["views_seed"])
    out["influence"] = out["influence"].fillna(out["influence"].median())
    out["influence"] = np.log1p(out["influence"].clip(lower=0))
    return out


def _safe_ic(x: pd.Series, y: pd.Series, min_n: int = 5) -> float | None:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < min_n or s["x"].std() == 0 or s["y"].std() == 0:
        return None
    v = s["x"].corr(s["y"])
    return round(float(v), 4) if pd.notna(v) else None

LEADERBOARD_CSV = OUTPUTS_DIR / "signal_leaderboard.csv"
LEADERBOARD_BY_INDUSTRY_CSV = OUTPUTS_DIR / "signal_leaderboard_by_industry.csv"
BEST_PER_INDUSTRY_CSV = OUTPUTS_DIR / "signal_best_per_industry.csv"
BEST_ALL_CSV = OUTPUTS_DIR / "signal_best_all.csv"
SCAN_SUMMARY_CSV = OUTPUTS_DIR / "signal_scan_summary.csv"
MULTI_HORIZON_LEADERBOARD_CSV = OUTPUTS_DIR / "signal_leaderboard_multi_horizon.csv"
MULTI_HORIZON_BEST_CSV = OUTPUTS_DIR / "signal_best_per_industry_horizon.csv"
MULTI_HORIZON_MATRIX_CSV = OUTPUTS_DIR / "signal_matrix_frequency_industry.csv"
SIGNAL_VARIANTS_CSV = OUTPUTS_DIR / "tweet_signals_variants.csv"

# 多频率扫描：每个 horizon 默认 lag（短频 contemporaneous，日频及以上 lag1）
DEFAULT_HORIZONS: tuple[dict[str, str | int], ...] = (
    {"ret_col": "ind_ret_5m", "horizon": "5m", "lag": 0, "min_days": 5, "label": "5分钟"},
    {"ret_col": "ind_ret_1h", "horizon": "1h", "lag": 0, "min_days": 5, "label": "1小时"},
    {"ret_col": "ind_ret_1d", "horizon": "1d", "lag": 1, "min_days": 5, "label": "1日"},
    {"ret_col": "ind_ret_5d", "horizon": "5d", "lag": 1, "min_days": 3, "label": "5日"},
    {"ret_col": "ind_ret_20d", "horizon": "20d", "lag": 1, "min_days": 3, "label": "20日"},
)

# 日频衍生信号（在 enrich_daily_signals 中生成）
DAILY_EXTRA_SIGNALS: dict[str, str] = {
    "sig_sentiment_delta": "情绪相对5日均值突变 × 当日影响力",
    "sig_sentiment_delta_contra": "情绪突变反转 × 影响力",
    "sig_contrarian_top3": "仅Top3 KOL的反转信号（日频 sum）",
    "sig_consensus_contrarian": "行业共识情绪反转（|均值|×影响力）",
    "sig_post_burst_contrarian": "发帖密集日 × 情绪反转",
    "sig_dispersion_contrarian": "高分歧日 × 情绪反转",
    "sig_bear_crowd": "看空帖占比 × 情绪反转（恐慌反弹）",
    "sig_bull_crowd": "看多帖占比 × 情绪（拥挤做多，对照）",
    "sig_net_crowd_contrarian": "(看空占比−看多占比) × 情绪反转",
    "sig_unanimity_contrarian": "低分歧时加强反转（一致预期）",
    "sig_bear_crowd_strict": "严格恐慌：bear_ratio>0.5 才 bear_crowd",
    "sig_kol_breadth_contrarian": "参与 KOL 数 × 情绪反转",
    "sig_momentum_daily": "跟随行业情绪动量（非反转）",
}


def _sent_series(df: pd.DataFrame) -> pd.Series:
    col = _pick_sentiment(df)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def build_signal_variants(df: pd.DataFrame) -> pd.DataFrame:
    """帖级多信号（均为可解释标量）。"""
    work = attach_influence_score(df)
    work["tweet_time"] = pd.to_datetime(work["tweet_time"], utc=True, errors="coerce")
    work["event_date"] = work["tweet_time"].dt.date
    s = _sent_series(work)
    work["sentiment"] = s
    inf = work["influence"]
    rank = pd.to_numeric(work.get("kol_rank_in_industry"), errors="coerce").fillna(5)
    rank_w = (6 - rank.clip(1, 5)) / 5.0

    likes = pd.to_numeric(work.get("likes_num", work.get("likes")), errors="coerce").fillna(0)
    views = pd.to_numeric(work.get("views", work.get("views_seed")), errors="coerce").fillna(0)
    eng = np.log1p(likes.clip(lower=0))
    user_med_likes = (
        work.groupby("username")["likes_num"]
        .transform(lambda x: x.median() if len(x) else 1.0)
        .replace(0, 1.0)
    )
    surprise_eng = eng / np.log1p(user_med_likes)

    is_rt = work.get("is_rt", False)
    if not isinstance(is_rt, pd.Series):
        is_rt = pd.Series(is_rt, index=work.index)
    orig = ~is_rt.astype(bool)

    # --- 信号族 ---
    work["sig_baseline"] = s * inf
    work["sig_contrarian"] = (-s) * inf
    work["sig_rank_weighted"] = s * inf * rank_w
    work["sig_top_rank_only"] = np.where(rank <= 3, s * inf, 0.0)
    work["sig_bull_only"] = np.maximum(s, 0) * inf
    work["sig_bear_only"] = np.minimum(s, 0) * inf
    work["sig_engagement"] = s * surprise_eng
    work["sig_attention"] = inf * np.sign(s.replace(0, np.nan).fillna(0))
    work["sig_attention_raw"] = inf  # 纯关注度，不看情绪
    work["sig_original_post"] = np.where(orig, s * inf, 0.0)
    work["sig_sentiment_x_likes"] = s * eng
    work["sig_contrarian_top3"] = np.where(rank <= 3, (-s) * inf, 0.0)

    # --- 新增：更有经济含义的候选信号 ---
    work = work.sort_values(["username", "tweet_time"])
    user_roll = work.groupby("username")["sentiment"].transform(
        lambda x: x.shift(1).rolling(15, min_periods=3).mean()
    )
    surprise_s = s - user_roll.fillna(0)
    work["sig_surprise_contrarian"] = (-surprise_s) * inf

    viral_thr = float(surprise_eng.quantile(0.65)) if len(surprise_eng) > 5 else 1.0
    viral = surprise_eng >= max(viral_thr, 1.05)
    work["sig_viral_contrarian"] = np.where(viral, (-s) * inf, 0.0)
    work["sig_contrarian_x_viral"] = (-s) * inf * np.clip(surprise_eng, 0.5, 3.0)

    extreme = s.abs() >= 0.2
    work["sig_extreme_contrarian"] = np.where(extreme, (-s) * inf, 0.0)
    work["sig_contrarian_orig"] = np.where(orig, (-s) * inf, 0.0)
    work["sig_bull_fade"] = np.where(s > 0.15, (-s) * inf, 0.0)
    work["sig_bear_bounce"] = np.where(s < -0.1, (-s) * inf, 0.0)

    if "sentiment_score" in work.columns:
        lex = pd.to_numeric(work["sentiment_score"], errors="coerce").fillna(0)
        disagree = (np.sign(lex) != np.sign(s)) & (lex.abs() > 0.08) & (s.abs() > 0.08)
        work["sig_nlp_lex_disagree"] = np.where(disagree, (-s) * inf, 0.0)

    # --- 扩展信号库（拥挤/广度/动量/失衡）---
    bear_mask = s < -0.1
    bull_mask = s > 0.1
    work["sig_bear_x_eng"] = np.where(bear_mask, (-s) * surprise_eng * inf, 0.0)
    work["sig_bull_x_eng"] = np.where(bull_mask, s * surprise_eng * inf, 0.0)
    work["sig_imbalance_contrarian"] = (-s) * inf * np.sign(s).replace(0, np.nan).fillna(0)
    work["sig_momentum"] = s * inf  # 跟随情绪（对照）
    work["sig_rank1_contrarian"] = np.where(rank <= 1, (-s) * inf, 0.0)
    work["sig_rank1_momentum"] = np.where(rank <= 1, s * inf, 0.0)
    work["sig_views_weighted"] = s * np.log1p(views.clip(lower=0))
    work["sig_contrarian_views"] = (-s) * np.log1p(views.clip(lower=0))
    work["sig_high_kol_count"] = inf * np.where(s.abs() >= 0.15, np.sign(s), 0.0)
    nlp_lex_gap = s
    if "sentiment_score" in work.columns:
        nlp_lex_gap = s - pd.to_numeric(work["sentiment_score"], errors="coerce").fillna(0)
    work["sig_nlp_minus_lex"] = (-nlp_lex_gap) * inf
    work["sig_strong_bear_only"] = np.where(s < -0.2, (-s) * inf, 0.0)
    work["sig_strong_bull_fade"] = np.where(s > 0.2, (-s) * inf, 0.0)
    work["sig_rt_contrarian"] = np.where(~orig, (-s) * inf, 0.0)
    work["sig_dual_confirm_contra"] = np.where(bear_mask & (surprise_eng >= 1.0), (-s) * inf, 0.0)

    # Transformer 监督 score（先跑 transformer_signal.py）
    if "transformer_score" in work.columns:
        ts = pd.to_numeric(work["transformer_score"], errors="coerce").fillna(0)
        work["sig_transformer"] = ts * inf
        work["sig_transformer_raw"] = ts

    return work


SIGNAL_DEFS: dict[str, str] = {
    "sig_baseline": "Sentiment × Influence（原版）",
    "sig_contrarian": "−Sentiment × Influence（反转）",
    "sig_rank_weighted": "Sentiment × Influence × Top排名权重",
    "sig_top_rank_only": "仅行业排名前3 KOL",
    "sig_bull_only": "仅看多情绪 × Influence",
    "sig_bear_only": "仅看空情绪 × Influence",
    "sig_engagement": "情绪 × 相对自身互动惊喜度",
    "sig_attention": "Influence × sign(Sentiment)",
    "sig_attention_raw": "纯 Influence（无情绪）",
    "sig_original_post": "非转发帖 × Sentiment × Influence",
    "sig_sentiment_x_likes": "Sentiment × log(1+likes)",
    "sig_contrarian_top3": "仅 Top3 KOL：−Sentiment × Influence",
    "sig_surprise_contrarian": "相对该 KOL 历史情绪的意外 × 反转",
    "sig_viral_contrarian": "高互动惊喜帖才做情绪反转",
    "sig_contrarian_x_viral": "反转 × 互动惊喜度（连续加权）",
    "sig_extreme_contrarian": "仅极端情绪帖（|s|≥0.2）反转",
    "sig_contrarian_orig": "仅原创帖（非 RT）反转",
    "sig_bull_fade": "仅明显看多帖反转（拥挤做多）",
    "sig_bear_bounce": "仅明显看空帖反转（恐慌反弹）",
    "sig_nlp_lex_disagree": "NLP 与词典情绪不一致时反转",
    "sig_bear_x_eng": "看空帖 × 互动惊喜 × 反转",
    "sig_bull_x_eng": "看多帖 × 互动惊喜（对照）",
    "sig_imbalance_contrarian": "情绪方向 × 影响力反转",
    "sig_momentum": "跟随情绪 × 影响力（非反转）",
    "sig_rank1_contrarian": "仅行业 #1 KOL 反转",
    "sig_rank1_momentum": "仅行业 #1 KOL 跟随",
    "sig_views_weighted": "情绪 × log(views)",
    "sig_contrarian_views": "反转 × log(views)",
    "sig_nlp_minus_lex": "NLP 与词典差 × 反转",
    "sig_strong_bear_only": "强看空 (s<-0.2) 反转",
    "sig_strong_bull_fade": "强看多 (s>0.2) 反转",
    "sig_rt_contrarian": "仅转发帖反转",
    "sig_dual_confirm_contra": "看空 + 高互动双重确认反转",
    "sig_transformer": "BERT→Ridge 预测收益 × Influence",
    "sig_transformer_raw": "BERT→Ridge 预测收益（纯 score）",
    "sig_ml_up": "行业日 Logistic：NLP+因子 → P(次日涨)−0.5",
    **DAILY_EXTRA_SIGNALS,
}


def aggregate_daily_signals(
    tweets: pd.DataFrame,
    signal_cols: list[str],
    *,
    weight_col: str = "influence",
) -> pd.DataFrame:
    gcols = ["event_date", "industry", "industry_label"]
    agg: dict = {
        "n_posts": ("tweet_id", "count"),
        "n_kol": ("username", "nunique"),
    }
    if weight_col in tweets.columns:
        agg["influence_sum"] = (weight_col, "sum")
    if "sentiment" in tweets.columns:
        agg["sentiment_mean"] = ("sentiment", "mean")
        agg["sentiment_sum"] = ("sentiment", "sum")
        agg["bear_posts"] = ("sentiment", lambda x: int((x < -0.1).sum()))
        agg["bull_posts"] = ("sentiment", lambda x: int((x > 0.1).sum()))

    for c in signal_cols:
        agg[f"{c}_sum"] = (c, "sum")
        agg[f"{c}_mean"] = (c, "mean")

    for rc in ("ind_ret_5m", "ind_ret_1h", "ind_ret_1d", "ind_ret_5d", "ind_ret_20d"):
        if rc in tweets.columns:
            agg[rc] = (rc, "mean")
    daily = tweets.groupby(gcols, as_index=False).agg(**agg)

    # 影响力加权平均（比简单 sum 更抗「刷屏」）
    if weight_col in tweets.columns:
        for c in signal_cols:
            tmp = tweets[gcols + [c, weight_col]].copy()
            tmp["_wx"] = tmp[c] * tmp[weight_col].clip(lower=1e-9)
            wm = tmp.groupby(gcols, as_index=False).agg(_wx=("_wx", "sum"), _den=(weight_col, "sum"))
            wm[f"{c}_wmean"] = wm["_wx"] / wm["_den"].replace(0, np.nan)
            daily = daily.merge(wm[gcols + [f"{c}_wmean"]], on=gcols, how="left")

    if "sentiment" in tweets.columns:
        disp = tweets.groupby(gcols)["sentiment"].std().rename("sentiment_dispersion")
        daily = daily.merge(disp.reset_index(), on=gcols, how="left")
        daily["sig_disagreement_sum"] = daily["sentiment_dispersion"].fillna(0)
    return daily.sort_values(gcols)


def enrich_daily_signals(daily: pd.DataFrame, *, momentum_window: int = 5) -> pd.DataFrame:
    """日频衍生：情绪突变、Top3 反转日频列等。"""
    d = daily.sort_values(["industry", "event_date"]).copy()
    if "sentiment_mean" in d.columns and "influence_sum" in d.columns:
        roll = d.groupby("industry")["sentiment_mean"].transform(
            lambda s: s.shift(1).rolling(momentum_window, min_periods=2).mean()
        )
        delta = d["sentiment_mean"] - roll
        inf = d["influence_sum"].fillna(1.0)
        d["sig_sentiment_delta_sum"] = delta * inf
        d["sig_sentiment_delta_contra_sum"] = (-delta) * inf
        sm = d["sentiment_mean"].fillna(0)
        inf = d["influence_sum"].fillna(1.0)
        d["sig_consensus_contrarian_sum"] = (-sm) * inf * sm.abs()
        d["sig_post_burst_contrarian_sum"] = (-sm) * np.log1p(d["n_posts"].fillna(0)) * inf
        if "sentiment_dispersion" in d.columns:
            disp = d["sentiment_dispersion"].fillna(0)
            d["sig_dispersion_contrarian_sum"] = (-sm) * disp * inf
        if "bear_posts" in d.columns and "n_posts" in d.columns:
            n = d["n_posts"].clip(lower=1)
            bear_r = d["bear_posts"] / n
            bull_r = d["bull_posts"] / n if "bull_posts" in d.columns else 0.0
            d["sig_bear_crowd_sum"] = bear_r * (-sm) * inf
            d["sig_bull_crowd_sum"] = bull_r * sm * inf
            d["sig_net_crowd_contrarian_sum"] = (bear_r - bull_r) * (-sm) * inf
            d["sig_bear_crowd_strict_sum"] = np.where(
                bear_r > 0.5, bear_r * (-sm) * inf, 0.0
            )
        if "sentiment_dispersion" in d.columns:
            disp = d["sentiment_dispersion"].fillna(0)
            d["sig_unanimity_contrarian_sum"] = (-sm) * inf / (disp + 0.15)
        if "n_kol" in d.columns:
            d["sig_kol_breadth_contrarian_sum"] = (
                d["n_kol"].fillna(0) * (-sm) * inf / d["n_posts"].clip(lower=1)
            )
        d["sig_momentum_daily_sum"] = sm * inf
    return d


def list_signal_value_cols(daily: pd.DataFrame) -> list[str]:
    """可用于 IC 评估的日频信号列。"""
    cols: list[str] = []
    for c in daily.columns:
        if not c.startswith("sig_"):
            continue
        if c.endswith("_sum") or c.endswith("_wmean"):
            cols.append(c)
    return sorted(set(cols))


def apply_filters(
    daily: pd.DataFrame,
    *,
    industry: str | None,
    max_rank: int | None,
    tweets: pd.DataFrame | None,
) -> pd.DataFrame:
    out = daily.copy()
    if industry:
        out = out[out["industry"] == industry]
    if max_rank and tweets is not None:
        top_users = tweets.loc[
            pd.to_numeric(tweets["kol_rank_in_industry"], errors="coerce") <= max_rank,
            "username",
        ].unique()
        t = tweets[tweets["username"].isin(top_users)].copy()
        return aggregate_daily_signals(t, [c for c in SIGNAL_DEFS if c in t.columns])
    return out


def apply_lag(daily: pd.DataFrame, signal_sum_col: str, lag: int) -> pd.DataFrame:
    if lag <= 0:
        return daily
    out = daily.sort_values(["industry", "event_date"]).copy()
    out[signal_sum_col] = out.groupby("industry")[signal_sum_col].shift(lag)
    return out.dropna(subset=[signal_sum_col])


def evaluate_signals(
    daily: pd.DataFrame,
    ret_col: str,
    *,
    lag: int = 0,
    min_days: int = 8,
) -> pd.DataFrame:
    rows: list[dict] = []
    value_cols = list_signal_value_cols(daily)

    for col in value_cols:
        d = apply_lag(daily, col, lag)
        sub = d.dropna(subset=[ret_col, col])
        ic = _safe_ic(sub[col], sub[ret_col], min_n=min_days)
        ls = cum = None
        if len(sub) >= 10 and sub[col].std() > 0:
            q, b = sub[col].quantile(0.7), sub[col].quantile(0.3)
            long = sub.loc[sub[col] >= q, ret_col].mean()
            short = sub.loc[sub[col] <= b, ret_col].mean()
            if pd.notna(long) and pd.notna(short):
                ls = round(float(long - short), 6)
            cum = round(float((np.sign(sub[col]) * sub[ret_col]).sum()), 6)
        name = col.replace("_sum", "").replace("_wmean", "")
        agg_type = "wmean" if col.endswith("_wmean") else "sum"
        rows.append(
            {
                "signal": name,
                "agg": agg_type,
                "signal_col": col,
                "description": SIGNAL_DEFS.get(name, name),
                "ret_col": ret_col,
                "lag_days": lag,
                "n_days": len(sub),
                "ic": ic,
                "ls_spread": ls,
                "cum_sign_ret": cum,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("ic", ascending=False, na_position="last")
    return out


def pick_best_universe(
    board: pd.DataFrame,
    *,
    ret_col: str,
    lag: int,
    min_days: int = 8,
    prefer_positive_ic: bool = True,
) -> pd.DataFrame:
    """全行业 pooled（日×行业腿）IC 最优信号。"""
    sub = board[
        (board["ret_col"] == ret_col)
        & (board["lag_days"] == lag)
        & (board.get("industry_filter", "all").astype(str) == "all")
    ].copy()
    sub = sub[sub["n_days"] >= min_days].dropna(subset=["ic"])
    if sub.empty:
        return pd.DataFrame()
    if prefer_positive_ic and (sub["ic"] > 0).any():
        pick = sub.loc[sub["ic"].idxmax()]
    else:
        pick = sub.loc[sub["ic"].abs().idxmax()]
    row = pick.to_dict()
    row["scope"] = "all_industries"
    return pd.DataFrame([row])


def pick_best_per_industry(
    board: pd.DataFrame,
    *,
    ret_col: str,
    lag: int,
    min_days: int = 5,
    prefer_positive_ic: bool = True,
) -> pd.DataFrame:
    """每个行业选 IC 最优的一条（样本不足则跳过）。"""
    sub = board[(board["ret_col"] == ret_col) & (board["lag_days"] == lag)].copy()
    sub = sub[sub["n_days"] >= min_days]
    rows: list[dict] = []
    for ind, g in sub.groupby("industry"):
        g = g.dropna(subset=["ic"])
        if g.empty:
            continue
        if prefer_positive_ic and (g["ic"] > 0).any():
            pick = g.loc[g["ic"].idxmax()]
        else:
            pick = g.loc[g["ic"].abs().idxmax()]
        r = pick.to_dict()
        r["scope"] = "per_industry"
        rows.append(r)
    best = pd.DataFrame(rows)
    if not best.empty:
        best = best.sort_values("ic", ascending=False)
    return best


def run_signal_scan(
    *,
    ret_cols: list[str],
    max_rank: int | None,
    lag: int,
    min_days: int = 5,
) -> pd.DataFrame:
    """先全行业选优，再分行业选优，写出汇总表。"""
    print("\n" + "=" * 60)
    print("  Step 1 / 2 — 全行业（所有日×行业腿 pooled）")
    print("=" * 60)
    board_all = run_lab(ret_cols=ret_cols, industry=None, max_rank=max_rank, lag=lag)

    all_best_parts: list[pd.DataFrame] = []
    for ret in ret_cols:
        b = pick_best_universe(board_all, ret_col=ret, lag=lag, min_days=max(min_days, 8))
        if not b.empty:
            all_best_parts.append(b)
    best_all = pd.concat(all_best_parts, ignore_index=True) if all_best_parts else pd.DataFrame()
    if not best_all.empty:
        best_all.to_csv(BEST_ALL_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  全行业最优 -> {BEST_ALL_CSV}")
        print(
            best_all[
                ["ret_col", "signal", "agg", "ic", "n_days", "ls_spread", "description"]
            ].to_string(index=False)
        )

    print("\n" + "=" * 60)
    print("  Step 2 / 2 — 分行业（各行业单独 IC 选优）")
    print("=" * 60)
    _, best_ind = run_by_industry(
        ret_cols=ret_cols, max_rank=max_rank, lag=lag, min_days=min_days
    )

    summary_parts: list[pd.DataFrame] = []
    if not best_all.empty:
        summary_parts.append(best_all)
    if not best_ind.empty:
        summary_parts.append(best_ind)
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(SCAN_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  汇总表 -> {SCAN_SUMMARY_CSV}")
    return summary


def build_daily_panel_per_industry(
    tweets: pd.DataFrame,
    mapping: pd.DataFrame,
    ret_col: str,
) -> pd.DataFrame:
    """按 mapping 表（signal_best_per_industry）为每个行业选用不同日频信号列。"""
    sig_cols = [c for c in SIGNAL_DEFS if c in tweets.columns and "delta" not in c[:20]]
    parts: list[pd.DataFrame] = []
    m = mapping[mapping["ret_col"] == ret_col] if "ret_col" in mapping.columns else mapping

    for _, row in m.iterrows():
        ind = row["industry"]
        sub = tweets[tweets["industry"] == ind].copy()
        if sub.empty:
            continue
        daily = enrich_daily_signals(aggregate_daily_signals(sub, sig_cols))
        col = row.get("signal_col") or f"{row['signal']}_sum"
        if col not in daily.columns:
            col = f"{row['signal']}_sum"
        if col not in daily.columns:
            continue

        block = daily[
            ["event_date", "industry", "industry_label", "n_posts", "n_kol", col, ret_col]
        ].copy()
        from data_mining import INDUSTRY_ETF_MAP

        block["etf"] = block["industry"].map(INDUSTRY_ETF_MAP)
        block["signal_raw"] = block[col]
        block["signal_lag1"] = block.groupby("industry")["signal_raw"].shift(1)
        block["fwd_ret"] = block[ret_col]
        block["position_lag1"] = np.sign(block["signal_lag1"]).replace(0, np.nan).fillna(0)
        block["position_contemp"] = np.sign(block["signal_raw"]).replace(0, np.nan).fillna(0)
        block["pnl_lag1"] = block["position_lag1"] * block["fwd_ret"]
        block["pnl_contemp"] = block["position_contemp"] * block["fwd_ret"]
        block["signal_used"] = row["signal"]
        block["signal_agg"] = row.get("agg", "sum")
        parts.append(block)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["event_date", "industry"])


def _filter_tweets_for_scan(tweets: pd.DataFrame, max_rank: int | None) -> pd.DataFrame:
    out = tweets.copy()
    if max_rank:
        rank_ok = pd.to_numeric(out["kol_rank_in_industry"], errors="coerce") <= max_rank
        out = out[rank_ok].copy()
    return out


def scan_by_industry_from_tweets(
    tweets: pd.DataFrame,
    *,
    ret_cols: list[str],
    lag: int,
    min_days: int = 5,
    max_rank: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分行业 IC 扫描（不写文件）。"""
    tweets_f = _filter_tweets_for_scan(tweets, max_rank)
    sig_cols = [c for c in SIGNAL_DEFS if c in tweets_f.columns and not c.startswith("sig_sentiment_delta")]
    industries = sorted(tweets_f["industry"].dropna().unique())

    all_board: list[pd.DataFrame] = []
    for ind in industries:
        sub_t = tweets_f[tweets_f["industry"] == ind].copy()
        if sub_t.empty:
            continue
        daily = enrich_daily_signals(aggregate_daily_signals(sub_t, sig_cols))
        label = sub_t["industry_label"].iloc[0] if "industry_label" in sub_t.columns else ind
        for ret in ret_cols:
            if ret not in daily.columns:
                continue
            ev = evaluate_signals(daily, ret, lag=lag, min_days=min_days)
            if ev.empty:
                continue
            ev = ev.assign(
                industry=ind,
                industry_label=label,
                max_rank=max_rank or "all",
            )
            all_board.append(ev)

    full = pd.concat(all_board, ignore_index=True) if all_board else pd.DataFrame()
    best_parts: list[pd.DataFrame] = []
    for ret in ret_cols:
        b = pick_best_per_industry(full, ret_col=ret, lag=lag, min_days=min_days)
        if not b.empty:
            best_parts.append(b)
    best = pd.concat(best_parts, ignore_index=True) if best_parts else pd.DataFrame()
    return full, best


def scan_universe_from_tweets(
    tweets: pd.DataFrame,
    *,
    ret_cols: list[str],
    lag: int,
    min_days: int = 5,
    max_rank: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全行业 pooled IC 扫描（不写文件）。"""
    tweets_f = _filter_tweets_for_scan(tweets, max_rank)
    sig_cols = [c for c in SIGNAL_DEFS if c in tweets_f.columns]
    daily = enrich_daily_signals(aggregate_daily_signals(tweets_f, sig_cols))
    parts: list[pd.DataFrame] = []
    for ret in ret_cols:
        if ret not in daily.columns:
            continue
        ev = evaluate_signals(daily, ret, lag=lag, min_days=max(min_days, 8))
        if ev.empty:
            continue
        parts.append(
            ev.assign(
                industry_filter="all",
                max_rank=max_rank or "all",
            )
        )
    board = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    best_parts: list[pd.DataFrame] = []
    for ret in ret_cols:
        b = pick_best_universe(board, ret_col=ret, lag=lag, min_days=max(min_days, 8))
        if not b.empty:
            best_parts.append(b)
    best = pd.concat(best_parts, ignore_index=True) if best_parts else pd.DataFrame()
    return board, best


def run_by_industry(
    *,
    ret_cols: list[str],
    max_rank: int | None,
    lag: int,
    min_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分行业扫描全部信号变体，输出排行榜 + 各行业最优信号表。"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_events()
    tweets = build_signal_variants(raw)
    full, best = scan_by_industry_from_tweets(
        tweets, ret_cols=ret_cols, lag=lag, min_days=min_days, max_rank=max_rank
    )
    if not full.empty:
        full.to_csv(LEADERBOARD_BY_INDUSTRY_CSV, index=False, encoding="utf-8-sig")
    if not best.empty:
        best.to_csv(BEST_PER_INDUSTRY_CSV, index=False, encoding="utf-8-sig")

    industries = sorted(tweets["industry"].dropna().unique())
    print(f"\n=== 分行业信号扫描 ({len(tweets)} 帖 | {len(industries)} 行业 | lag={lag}) ===")
    print(f"  明细 -> {LEADERBOARD_BY_INDUSTRY_CSV}")
    print(f"  各行业最优 -> {BEST_PER_INDUSTRY_CSV}\n")

    for ret in ret_cols:
        b = best[best["ret_col"] == ret] if not best.empty else pd.DataFrame()
        if b.empty:
            continue
        print(f"--- 推荐主信号 ({ret} | lag={lag}) ---")
        show = b[
            [
                "industry_label",
                "signal",
                "agg",
                "ic",
                "n_days",
                "ls_spread",
                "cum_sign_ret",
                "description",
            ]
        ]
        print(show.to_string(index=False))
        print()

    return full, best


def run_multi_horizon_scan(
    *,
    horizons: tuple[dict[str, str | int], ...] | None = None,
    max_rank: int | None = None,
    min_days: int = 5,
) -> pd.DataFrame:
    """多频率 × 分行业 signal 扫描，输出矩阵与明细。"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = list(horizons or DEFAULT_HORIZONS)
    raw = load_events()
    tweets = build_signal_variants(raw)

    label_counts: dict[str, int] = {}
    for spec in specs:
        rc = str(spec["ret_col"])
        if rc in tweets.columns:
            label_counts[rc] = int(tweets[rc].notna().sum())

    board_parts: list[pd.DataFrame] = []
    best_parts: list[pd.DataFrame] = []
    matrix_rows: list[dict] = []

    print("\n" + "=" * 60)
    print("  多频率 signal 扫描（频率 × 行业）")
    print("=" * 60)
    print("  帖级标签覆盖:", ", ".join(f"{k}={v}" for k, v in label_counts.items()))

    for spec in specs:
        ret_col = str(spec["ret_col"])
        horizon = str(spec["horizon"])
        lag = int(spec["lag"])
        hmin = max(int(spec.get("min_days", min_days)), min_days)
        hlabel = str(spec.get("label", horizon))
        n_labels = label_counts.get(ret_col, 0)
        if n_labels < 10:
            print(f"\n  [跳过] {hlabel} ({ret_col}): 有效标签仅 {n_labels} 条")
            continue

        print(f"\n--- {hlabel} | {ret_col} | lag={lag} | min_days={hmin} ---")
        full, best_ind = scan_by_industry_from_tweets(
            tweets, ret_cols=[ret_col], lag=lag, min_days=hmin, max_rank=max_rank
        )
        _, best_all = scan_universe_from_tweets(
            tweets, ret_cols=[ret_col], lag=lag, min_days=hmin, max_rank=max_rank
        )

        meta = {"horizon": horizon, "horizon_label": hlabel, "label_obs_tweets": n_labels}
        if not full.empty:
            full = full.assign(**meta)
            board_parts.append(full)
        if not best_ind.empty:
            best_ind = best_ind.assign(**meta, scope="per_industry")
            best_parts.append(best_ind)
            for _, row in best_ind.iterrows():
                matrix_rows.append(
                    {
                        "industry": row.get("industry"),
                        "industry_label": row.get("industry_label"),
                        "horizon": horizon,
                        "horizon_label": hlabel,
                        "ret_col": ret_col,
                        "lag_days": lag,
                        "best_signal": row.get("signal"),
                        "signal_agg": row.get("agg"),
                        "signal_col": row.get("signal_col"),
                        "ic": row.get("ic"),
                        "n_days": row.get("n_days"),
                        "ls_spread": row.get("ls_spread"),
                        "cum_sign_ret": row.get("cum_sign_ret"),
                        "description": row.get("description"),
                        "scope": "per_industry",
                        "label_obs_tweets": n_labels,
                    }
                )
            show = best_ind[
                ["industry_label", "signal", "ic", "n_days", "ls_spread", "description"]
            ]
            print(show.to_string(index=False))

        if not best_all.empty:
            r = best_all.iloc[0].to_dict()
            matrix_rows.append(
                {
                    "industry": "all",
                    "industry_label": "全行业 pooled",
                    "horizon": horizon,
                    "horizon_label": hlabel,
                    "ret_col": ret_col,
                    "lag_days": lag,
                    "best_signal": r.get("signal"),
                    "signal_agg": r.get("agg"),
                    "signal_col": r.get("signal_col"),
                    "ic": r.get("ic"),
                    "n_days": r.get("n_days"),
                    "ls_spread": r.get("ls_spread"),
                    "cum_sign_ret": r.get("cum_sign_ret"),
                    "description": r.get("description"),
                    "scope": "all_industries",
                    "label_obs_tweets": n_labels,
                }
            )
            print(
                f"  全行业最优: {r.get('signal')} | IC={r.get('ic')} | n={r.get('n_days')}"
            )

    board = pd.concat(board_parts, ignore_index=True) if board_parts else pd.DataFrame()
    best = pd.concat(best_parts, ignore_index=True) if best_parts else pd.DataFrame()
    matrix = pd.DataFrame(matrix_rows)

    if not board.empty:
        board.to_csv(MULTI_HORIZON_LEADERBOARD_CSV, index=False, encoding="utf-8-sig")
    if not best.empty:
        best.to_csv(MULTI_HORIZON_BEST_CSV, index=False, encoding="utf-8-sig")
    if not matrix.empty:
        matrix = matrix.sort_values(["horizon", "industry"], na_position="last")
        matrix.to_csv(MULTI_HORIZON_MATRIX_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("  输出文件")
    print("=" * 60)
    print(f"  明细排行榜 -> {MULTI_HORIZON_LEADERBOARD_CSV}")
    print(f"  分行业最优 -> {MULTI_HORIZON_BEST_CSV}")
    print(f"  频率×行业矩阵 -> {MULTI_HORIZON_MATRIX_CSV}")

    if not matrix.empty:
        print("\n  【矩阵摘要】各行业 × 频率 最优 IC:")
        pivot = matrix[matrix["scope"] == "per_industry"].pivot_table(
            index="industry_label",
            columns="horizon_label",
            values="ic",
            aggfunc="first",
        )
        print(pivot.to_string())

    return matrix


def run_lab(
    *,
    ret_cols: list[str],
    industry: str | None,
    max_rank: int | None,
    lag: int,
) -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_events()
    tweets = build_signal_variants(raw)
    sig_cols = [c for c in SIGNAL_DEFS if c in tweets.columns]

    if max_rank:
        rank_ok = pd.to_numeric(tweets["kol_rank_in_industry"], errors="coerce") <= max_rank
        tweets_f = tweets[rank_ok].copy()
    else:
        tweets_f = tweets

    if industry:
        tweets_f = tweets_f[tweets_f["industry"] == industry].copy()

    save_cols = [
        "tweet_id",
        "username",
        "event_date",
        "industry",
        "industry_label",
        "kol_rank_in_industry",
        "sentiment",
        "influence",
        *sig_cols,
        "ind_ret_1h",
        "ind_ret_1d",
    ]
    save_cols = [c for c in save_cols if c in tweets_f.columns]
    tweets_f[save_cols].to_csv(SIGNAL_VARIANTS_CSV, index=False, encoding="utf-8-sig")

    daily = aggregate_daily_signals(tweets_f, sig_cols)
    daily = enrich_daily_signals(daily)
    parts: list[pd.DataFrame] = []
    for ret in ret_cols:
        if ret not in daily.columns:
            continue
        parts.append(
            evaluate_signals(daily, ret, lag=lag).assign(
                industry_filter=industry or "all",
                max_rank=max_rank or "all",
            )
        )
    board = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    board.to_csv(LEADERBOARD_CSV, index=False, encoding="utf-8-sig")

    print(f"\n=== 信号实验 ({len(tweets_f)} 帖 | filter industry={industry or 'all'} rank<={max_rank or 'all'}) ===")
    print(f"  变体列 -> {SIGNAL_VARIANTS_CSV}")
    print(f"  排行榜 -> {LEADERBOARD_CSV}\n")
    for ret in ret_cols:
        sub = board[board["ret_col"] == ret]
        if sub.empty:
            continue
        print(f"--- {ret} | lag={lag} ---")
        print(
            sub.head(8)[
                ["signal", "ic", "n_days", "ls_spread", "cum_sign_ret", "description"]
            ].to_string(index=False)
        )
        print()
    return board




# ========== Text encoder ==========

Backend = Literal["auto", "bert", "tfidf"]


class FrozenChineseEncoder:
    """bert-base-chinese [CLS] embedding，不 fine-tune。"""

    name = "bert-base-chinese"

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
            from transformers import BertModel, BertTokenizer

        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._model = BertModel.from_pretrained(self.model_name)
        self._model.eval()

    @property
    def dim(self) -> int:
        return 768

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        self._lazy_load()
        import torch

        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = [str(t or " ").strip() or " " for t in texts[i : i + batch_size]]
            enc = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self._model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            vecs.append(cls)
        return np.vstack(vecs)


class TfidfEmbeddingEncoder:
    """TF-IDF → TruncatedSVD，得到固定维稠密向量（非 Transformer，但流程一致）。"""

    name = "tfidf-svd"

    def __init__(self, n_components: int = 128, max_features: int = 800):
        self.n_components = n_components
        self.max_features = max_features
        self._tfidf = None
        self._svd = None
        self._fitted = False

    @property
    def dim(self) -> int:
        return self.n_components

    def fit(self, texts: list[str]) -> TfidfEmbeddingEncoder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        clean = [str(t or " ").strip() or " " for t in texts]
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
        )
        X = self._tfidf.fit_transform(clean)
        k = min(self.n_components, max(1, X.shape[1] - 1), max(1, X.shape[0] - 1))
        self.n_components = k
        self._svd = TruncatedSVD(n_components=k, random_state=42)
        self._svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        assert self._tfidf is not None and self._svd is not None
        clean = [str(t or " ").strip() or " " for t in texts]
        X = self._tfidf.transform(clean)
        return self._svd.transform(X).astype(np.float32)


def bert_available() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import BertModel  # noqa: F401

        return True
    except Exception:
        return False


def get_text_encoder(
    backend: Backend = "auto",
    *,
    model_name: str = "bert-base-chinese",
    tfidf_dim: int = 128,
) -> FrozenChineseEncoder | TfidfEmbeddingEncoder:
    if backend == "tfidf":
        return TfidfEmbeddingEncoder(n_components=tfidf_dim)
    if backend == "bert" or (backend == "auto" and bert_available()):
        try:
            return FrozenChineseEncoder(model_name=model_name)
        except Exception:
            if backend == "bert":
                raise
    return TfidfEmbeddingEncoder(n_components=tfidf_dim)


TRADING_DAYS_PER_YEAR = 252

# ========== Panel (shared with backtesting) ==========

def filter_tweets(
    tweets: pd.DataFrame,
    *,
    industry: str | None,
    max_rank: int | None,
) -> pd.DataFrame:
    out = tweets.copy()
    if industry and industry not in ("all", "*"):
        out = out[out["industry"] == industry]
    if max_rank is not None:
        r = pd.to_numeric(out["kol_rank_in_industry"], errors="coerce")
        out = out[r <= max_rank]
    return out


def build_daily_panel(
    tweets: pd.DataFrame,
    signal_col: str,
    ret_col: str = "ind_ret_1d",
    *,
    value_col: str | None = None,
) -> pd.DataFrame:
    """日×行业面板，含 lag 信号与 forward 收益。"""
    post_cols = [c for c in SIGNAL_DEFS if c in tweets.columns]
    daily = enrich_daily_signals(aggregate_daily_signals(tweets, post_cols))
    if value_col and value_col in daily.columns:
        sum_col = value_col
    else:
        sum_col = f"{signal_col}_sum"
        if sum_col not in daily.columns:
            wmean_col = f"{signal_col}_wmean"
            if wmean_col in daily.columns:
                sum_col = wmean_col
            else:
                raise ValueError(f"缺少 {sum_col}，signal={signal_col}（日频衍生信号需 enrich_daily_signals）")
    if ret_col not in daily.columns:
        raise ValueError(f"缺少 {ret_col}")

    daily = daily.sort_values(["industry", "event_date"]).reset_index(drop=True)
    daily["event_date"] = pd.to_datetime(daily["event_date"])
    daily["etf"] = daily["industry"].map(INDUSTRY_ETF_MAP)
    daily["signal_raw"] = daily[sum_col]
    daily["signal_lag1"] = daily.groupby("industry")["signal_raw"].shift(1)
    daily["fwd_ret"] = daily[ret_col]
    daily["position_lag1"] = np.sign(daily["signal_lag1"]).replace(0, np.nan).fillna(0)
    daily["position_contemp"] = np.sign(daily["signal_raw"]).replace(0, np.nan).fillna(0)
    daily["pnl_lag1"] = daily["position_lag1"] * daily["fwd_ret"]
    daily["pnl_contemp"] = daily["position_contemp"] * daily["fwd_ret"]
    return daily


ML_PANEL_CSV = OUTPUTS_DIR / "ml_industry_daily_panel.csv"
ML_METRICS_JSON = OUTPUTS_DIR / "ml_updown_metrics.json"

# 行业日特征：NLP 聚合 + 结构化因子（不含帖级 sig_* 列）
INDUSTRY_ML_FEATURE_COLS = (
    "sentiment_mean",
    "sentiment_sum",
    "sentiment_dispersion",
    "n_kol",
    "n_posts",
    "influence_sum",
    "bear_posts",
    "bull_posts",
    "bear_ratio",
    "bull_ratio",
    "kol_breadth_x_bear",
)


def build_industry_daily_dataset(
    tweets: pd.DataFrame,
    ret_col: str = "ind_ret_1d",
) -> pd.DataFrame:
    """(event_date, industry) 一行：X=日频 NLP/因子，y=次日涨跌。"""
    work = build_signal_variants(tweets)
    daily = aggregate_daily_signals(work, signal_cols=[])
    daily = daily.dropna(subset=[ret_col]).copy()
    if daily.empty:
        return daily

    n = daily["n_posts"].clip(lower=1)
    daily["bear_ratio"] = daily["bear_posts"].fillna(0) / n
    daily["bull_ratio"] = daily["bull_posts"].fillna(0) / n
    sm = daily["sentiment_mean"].fillna(0)
    daily["kol_breadth_x_bear"] = daily["n_kol"].fillna(0) * (-sm) * daily["influence_sum"].fillna(1.0) / n
    daily["y_up"] = (pd.to_numeric(daily[ret_col], errors="coerce") > 0).astype(int)
    daily["event_date"] = pd.to_datetime(daily["event_date"])
    return daily.sort_values(["event_date", "industry"]).reset_index(drop=True)


def split_daily_by_time(
    daily: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = daily["event_date"].drop_duplicates().sort_values()
    if len(dates) < 3:
        cut = dates.iloc[-1] if len(dates) else pd.Timestamp("1970-01-01", tz="UTC")
        return daily.iloc[:0], daily, cut
    cut_idx = max(1, min(int(len(dates) * train_ratio), len(dates) - 1))
    cut_date = dates.iloc[cut_idx]
    train = daily[daily["event_date"] < cut_date].copy()
    test = daily[daily["event_date"] >= cut_date].copy()
    return train, test, cut_date


def _build_industry_ml_matrix(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    use_industry: bool,
    fit_encoder: OneHotEncoder | None = None,
) -> tuple[np.ndarray, OneHotEncoder | None]:
    num = df[list(feature_cols)].astype(float).values
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    if not use_industry:
        return num, None
    enc = fit_encoder or OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat = enc.fit_transform(df[["industry"]]) if fit_encoder is None else enc.transform(df[["industry"]])
    return np.hstack([num, cat]), enc


def fit_ml_updown_model(
    train: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = INDUSTRY_ML_FEATURE_COLS,
    use_industry: bool = True,
    C: float = 1.0,
) -> Pipeline:
    X, enc = _build_industry_ml_matrix(
        train, feature_cols=feature_cols, use_industry=use_industry
    )
    y = train["y_up"].values
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    pipe.industry_encoder_ = enc  # type: ignore[attr-defined]
    pipe.feature_cols_ = feature_cols  # type: ignore[attr-defined]
    pipe.use_industry_ = use_industry  # type: ignore[attr-defined]
    return pipe


def predict_ml_updown(
    model: Pipeline,
    df: pd.DataFrame,
) -> np.ndarray:
    enc = getattr(model, "industry_encoder_", None)
    cols = getattr(model, "feature_cols_", INDUSTRY_ML_FEATURE_COLS)
    use_ind = getattr(model, "use_industry_", True)
    X, _ = _build_industry_ml_matrix(
        df, feature_cols=cols, use_industry=use_ind, fit_encoder=enc
    )
    return model.predict_proba(X)[:, 1]


def _ml_cls_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    fwd_ret: np.ndarray | None = None,
) -> dict:
    pred = (prob >= 0.5).astype(int)
    out: dict = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4) if len(y_true) else None,
    }
    if len(np.unique(y_true)) >= 2 and len(np.unique(prob)) >= 2:
        try:
            out["auc"] = round(float(roc_auc_score(y_true, prob)), 4)
        except ValueError:
            out["auc"] = None
    else:
        out["auc"] = None
    if fwd_ret is not None:
        out["ic_prob_fwd_ret"] = _safe_ic(pd.Series(prob), pd.Series(fwd_ret), min_n=5)
    return out


def run_ml_updown(
    *,
    ret_col: str = "ind_ret_1d",
    train_ratio: float = 0.7,
    use_industry: bool = True,
    C: float = 1.0,
    industry: str | None = None,
    max_rank: int | None = None,
) -> dict:
    """行业日二分类：预测次日涨/跌概率，写出 ml_industry_daily_panel.csv。"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_events()
    tweets = filter_tweets(raw, industry=industry, max_rank=max_rank)
    daily = build_industry_daily_dataset(tweets, ret_col=ret_col)
    if len(daily) < 15:
        raise SystemExit(f"有效行业日样本 {len(daily)} < 15，请先增厚数据")

    train, test, cut_date = split_daily_by_time(daily, train_ratio)
    if train.empty or test.empty:
        raise SystemExit("时间切分后训练或测试为空，请调整 --train-ratio 或补数据")

    model = fit_ml_updown_model(train, use_industry=use_industry, C=C)
    daily = daily.copy()
    daily["ml_prob_up"] = predict_ml_updown(model, daily)
    daily["ml_split"] = np.where(daily["event_date"] >= cut_date, "test", "train")

    ret_v = daily[ret_col].astype(float).values
    m_train = _ml_cls_metrics(
        train["y_up"].values,
        predict_ml_updown(model, train),
        train[ret_col].astype(float).values,
    )
    m_test = _ml_cls_metrics(
        test["y_up"].values,
        predict_ml_updown(model, test),
        test[ret_col].astype(float).values,
    )
    m_full = _ml_cls_metrics(
        daily["y_up"].values,
        daily["ml_prob_up"].values,
        ret_v,
    )

    # 信号：概率相对 0.5 的偏离（连续，便于 IC）
    daily["sig_ml_up_sum"] = daily["ml_prob_up"] - 0.5
    daily.to_csv(ML_PANEL_CSV, index=False, encoding="utf-8-sig")

    metrics = {
        "ret_col": ret_col,
        "train_ratio": train_ratio,
        "cut_date": str(cut_date.date()),
        "n_daily": int(len(daily)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": list(INDUSTRY_ML_FEATURE_COLS),
        "use_industry_onehot": use_industry,
        "C": C,
        "train": m_train,
        "test": m_test,
        "full": m_full,
        "y_up_rate": round(float(daily["y_up"].mean()), 4),
    }
    with open(ML_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== 行业日涨跌二分类 (Logistic) ===")
    print(f"  样本: {len(daily)} 行业日 | 训练 {len(train)} | 测试 {len(test)} | 切分日 {cut_date.date()}")
    print(f"  特征: {', '.join(INDUSTRY_ML_FEATURE_COLS)}")
    print(f"  训练 AUC={m_train.get('auc')} acc={m_train.get('accuracy')} | 测试 AUC={m_test.get('auc')} acc={m_test.get('accuracy')}")
    print(f"  测试 IC(prob vs {ret_col})={m_test.get('ic_prob_fwd_ret')}")
    print(f"  -> {ML_PANEL_CSV}")
    print(f"  -> {ML_METRICS_JSON}")
    print("  回测: python3 backtesting.py --all --signal sig_ml_up --mode lag1")
    return metrics


def build_ml_daily_panel(
    tweets: pd.DataFrame,
    ret_col: str = "ind_ret_1d",
    *,
    train_ratio: float = 0.7,
    use_industry: bool = True,
    C: float = 1.0,
) -> pd.DataFrame:
    """训练 Logistic 并生成与 build_daily_panel 相同结构的 lag1 面板。"""
    daily = build_industry_daily_dataset(tweets, ret_col=ret_col)
    if len(daily) < 10:
        raise ValueError(f"行业日样本不足: {len(daily)}")

    train, _, _ = split_daily_by_time(daily, train_ratio)
    if train.empty:
        raise ValueError("训练集为空，无法拟合 sig_ml_up")

    model = fit_ml_updown_model(train, use_industry=use_industry, C=C)
    daily = daily.sort_values(["industry", "event_date"]).reset_index(drop=True)
    daily["ml_prob_up"] = predict_ml_updown(model, daily)
    daily["etf"] = daily["industry"].map(INDUSTRY_ETF_MAP)
    daily["signal_raw"] = daily["ml_prob_up"] - 0.5
    daily["signal_lag1"] = daily.groupby("industry")["signal_raw"].shift(1)
    daily["fwd_ret"] = daily[ret_col]
    daily["position_lag1"] = np.sign(daily["signal_lag1"]).replace(0, np.nan).fillna(0)
    daily["position_contemp"] = np.sign(daily["signal_raw"]).replace(0, np.nan).fillna(0)
    daily["pnl_lag1"] = daily["position_lag1"] * daily["fwd_ret"]
    daily["pnl_contemp"] = daily["position_contemp"] * daily["fwd_ret"]
    daily.attrs["ml_model"] = "logistic_updown"
    daily.attrs["ml_train_ratio"] = train_ratio
    return daily


def apply_costs(pnl: pd.Series, position: pd.Series, cost_bps: float) -> pd.Series:
    """换手时扣费（单边 cost_bps basis points）。"""
    if cost_bps <= 0:
        return pnl
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 10_000.0)
    return pnl - cost


def equity_curve(pnl: pd.Series) -> pd.Series:
    pnl = pnl.fillna(0)
    return (1 + pnl).cumprod()


def performance_stats(
    panel: pd.DataFrame,
    pnl_col: str,
    position_col: str,
    *,
    split: str,
    net_pnl: bool = False,
) -> dict:
    if net_pnl:
        sub = panel.dropna(subset=["pnl_net"]).copy()
    else:
        sub = panel.dropna(subset=[pnl_col, "fwd_ret"]).copy()
    if sub.empty:
        return {"split": split, "n_days": 0}

    if net_pnl:
        pnl = sub["pnl_net"].fillna(0)
    else:
        pnl = apply_costs(sub[pnl_col], sub[position_col], cost_bps=sub.attrs.get("cost_bps", 0))
    eq = equity_curve(pnl)
    total_ret = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    ann_factor = TRADING_DAYS_PER_YEAR / max(len(pnl), 1)
    ann_ret = (1 + total_ret) ** ann_factor - 1 if len(pnl) else 0.0
    vol = float(pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if pnl.std() > 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else None
    dd = (eq / eq.cummax() - 1) if len(eq) else pd.Series([0])
    max_dd = float(dd.min()) if len(dd) else 0.0
    win_rate = float((pnl > 0).mean()) if len(pnl) else None
    ic = None
    if "fwd_ret" in sub.columns and "signal_lag1" in sub.columns:
        sig_for_ic = sub["signal_lag1"] if "lag1" in pnl_col or net_pnl else sub["signal_raw"]
        ic = _safe_ic(sig_for_ic, sub["fwd_ret"], min_n=5)

    active = sub[position_col] != 0 if position_col in sub.columns else pnl != 0
    return {
        "split": split,
        "n_days": int(len(sub)),
        "n_active": int(active.sum()),
        "total_return": round(total_ret, 6),
        "ann_return": round(ann_ret, 6),
        "ann_vol": round(vol, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "ic_signal_fwd_ret": ic,
        "avg_daily_pnl": round(float(pnl.mean()), 6),
    }


def finalize_panel(
    panel: pd.DataFrame,
    pnl_col: str,
    pos_col: str,
    cost_bps: float,
    mode: str,
    signal_col: str,
) -> pd.DataFrame:
    """按行业分别扣费、分别累计净值；并生成等权组合日收益。"""
    out = panel.dropna(subset=["fwd_ret"]).copy()
    out.attrs["cost_bps"] = cost_bps
    out["pnl_gross"] = out[pnl_col]
    out["pnl_net"] = np.nan
    out["equity_industry"] = np.nan

    for _ind, idx in out.groupby("industry").groups.items():
        g = out.loc[idx].sort_values("event_date")
        net = apply_costs(g[pnl_col], g[pos_col], cost_bps)
        out.loc[g.index, "pnl_net"] = net.values
        out.loc[g.index, "equity_industry"] = equity_curve(net).values

    out["strategy_mode"] = mode
    out["signal_name"] = signal_col
    return out


def build_combined_portfolio(panel: pd.DataFrame) -> pd.DataFrame:
    """每个交易日：各行业等权平均净收益 → 一条组合净值。"""
    comb = (
        panel.groupby("event_date", as_index=False)
        .agg(
            pnl_net=("pnl_net", "mean"),
            pnl_gross=("pnl_gross", "mean"),
            n_industries=("industry", "nunique"),
            n_posts=("n_posts", "sum"),
        )
        .sort_values("event_date")
    )
    comb["equity"] = equity_curve(comb["pnl_net"])
    comb["split"] = "combined_equal_weight"
    return comb


def performance_stats_combined(combined: pd.DataFrame, *, split: str) -> dict:
    sub = combined.dropna(subset=["pnl_net"]).copy()
    if sub.empty:
        return {"split": split, "n_days": 0}
    pnl = sub["pnl_net"].fillna(0)
    eq = equity_curve(pnl)
    total_ret = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    ann_factor = TRADING_DAYS_PER_YEAR / max(len(pnl), 1)
    ann_ret = (1 + total_ret) ** ann_factor - 1 if len(pnl) else 0.0
    vol = float(pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if pnl.std() > 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else None
    dd = (eq / eq.cummax() - 1) if len(eq) else pd.Series([0])
    max_dd = float(dd.min()) if len(dd) else 0.0
    win_rate = float((pnl > 0).mean()) if len(pnl) else None
    return {
        "split": split,
        "n_days": int(len(sub)),
        "n_active": int((pnl != 0).sum()),
        "total_return": round(total_ret, 6),
        "ann_return": round(ann_ret, 6),
        "ann_vol": round(vol, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "ic_signal_fwd_ret": None,
        "avg_daily_pnl": round(float(pnl.mean()), 6),
        "note": "equal_weight_across_industries",
    }


def backtest_by_industry(
    panel: pd.DataFrame,
    pnl_col: str,
    pos_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (ind, label), sub in panel.groupby(["industry", "industry_label"]):
        st = performance_stats(sub, pnl_col, pos_col, split="full", net_pnl=True)
        st["industry"] = ind
        st["industry_label"] = label
        st["etf"] = INDUSTRY_ETF_MAP.get(ind)
        rows.append(st)
    return pd.DataFrame(rows)


def time_split(panel: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(panel["event_date"].unique())
    if len(dates) < 5:
        return panel, panel.iloc[0:0]
    cut = int(len(dates) * train_ratio)
    cut_date = dates[cut]
    train = panel[panel["event_date"] < cut_date]
    test = panel[panel["event_date"] >= cut_date]
    return train, test




# ========== ML score ==========

TABULAR_COLS = (
    "sentiment_score_nlp",
    "sentiment_score",
    "kol_rank_in_industry",
    "views_seed",
    "likes_num",
    "log_views",
    "is_rt",
    "hour_utc",
    "dow_utc",
)


def prepare_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    if KOL_RANKED_CSV.exists() and "influence_score" not in out.columns:
        rk = pd.read_csv(KOL_RANKED_CSV)[["username", "influence_score"]]
        out = out.merge(rk, on="username", how="left")
    out["text"] = out.get("text_clean", out.get("text", "")).fillna("").astype(str)
    out["y"] = pd.to_numeric(out[target], errors="coerce")
    out = out.dropna(subset=["y"])
    out["sentiment_score_nlp"] = pd.to_numeric(
        out.get("sentiment_score_nlp", 0), errors="coerce"
    ).fillna(0)
    out["sentiment_score"] = pd.to_numeric(out.get("sentiment_score", 0), errors="coerce").fillna(0)
    out["kol_rank_in_industry"] = pd.to_numeric(
        out.get("kol_rank_in_industry"), errors="coerce"
    ).fillna(5)
    out["views_seed"] = pd.to_numeric(out.get("views_seed"), errors="coerce")
    if out["views_seed"].isna().all() and "influence_score" in out.columns:
        out["views_seed"] = out["influence_score"]
    out["views_seed"] = out["views_seed"].fillna(out["views_seed"].median())
    out["likes_num"] = pd.to_numeric(out.get("likes_num", out.get("likes")), errors="coerce").fillna(0)
    out["log_views"] = np.log1p(out["views_seed"].clip(lower=0))
    out["is_rt"] = out.get("is_rt", False).astype(int)
    out["hour_utc"] = out["tweet_time"].dt.hour
    out["dow_utc"] = out["tweet_time"].dt.dayofweek
    out["industry"] = out.get("industry", "unclassified").fillna("unclassified").astype(str)
    return out.sort_values("tweet_time").reset_index(drop=True)


def ml_time_split_indices(df: pd.DataFrame, test_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    if n < 20:
        idx = np.arange(n)
        return train_test_split(idx, test_size=max(0.2, test_ratio), shuffle=False)
    cut = int(n * (1 - test_ratio))
    return np.arange(cut), np.arange(cut, n)


def build_tabular_matrix(df: pd.DataFrame, fit_encoder: OneHotEncoder | None = None):
    num = df[list(TABULAR_COLS)].astype(float).values
    enc = fit_encoder or OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat = enc.fit_transform(df[["industry"]]) if fit_encoder is None else enc.transform(df[["industry"]])
    return np.hstack([num, cat]), enc

EMB_NPZ = OUTPUTS_DIR / "tweet_embeddings.npz"
SCORES_CSV = OUTPUTS_DIR / "tweet_transformer_scores.csv"
META_JSON = OUTPUTS_DIR / "transformer_signal_meta.json"


def _prep_text(s: str) -> str:
    return str(s or "").strip() or " "


def encode_all(
    df: pd.DataFrame,
    *,
    backend: str,
    model_name: str,
    batch_size: int,
    tfidf_dim: int,
) -> tuple[np.ndarray, str]:
    texts = df.get("text_clean", df.get("text", "")).fillna("").map(_prep_text).tolist()
    enc = get_text_encoder(backend, model_name=model_name, tfidf_dim=tfidf_dim)  # type: ignore[arg-type]
    if hasattr(enc, "fit") and enc.name.startswith("tfidf"):
        enc.fit(texts)
    emb = enc.encode(texts, batch_size=batch_size)
    return emb, enc.name


def fit_score_ridge(
    train_df: pd.DataFrame,
    emb_train: np.ndarray,
    *,
    alpha: float,
    pca: PCA | None,
) -> Pipeline:
    Xe = pca.transform(emb_train) if pca is not None else emb_train
    X_tab, enc = build_tabular_matrix(train_df)
    X = np.nan_to_num(np.hstack([Xe, X_tab]), nan=0.0, posinf=0.0, neginf=0.0)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipe.fit(X, train_df["y"].values)
    pipe.tabular_encoder_ = enc  # type: ignore[attr-defined]
    pipe.pca_ = pca  # type: ignore[attr-defined]
    return pipe


def predict_scores(
    model: Pipeline,
    df: pd.DataFrame,
    emb: np.ndarray,
) -> np.ndarray:
    pca: PCA | None = getattr(model, "pca_", None)
    enc = getattr(model, "tabular_encoder_", None)
    Xe = pca.transform(emb) if pca is not None else emb
    X_tab, _ = build_tabular_matrix(df, enc)
    X = np.nan_to_num(np.hstack([Xe, X_tab]), nan=0.0, posinf=0.0, neginf=0.0)
    y = model.predict(X)
    clip = np.nanpercentile(np.abs(df["y"].values), 99) if "y" in df.columns and df["y"].notna().any() else 0.2
    clip = max(float(clip) * 2, 0.05)
    return np.clip(y, -clip, clip)


def run_ml_score(
    *,
    target: str = "ind_ret_1d",
    test_ratio: float = 0.2,
    alpha: float = 1.0,
    model_name: str = "bert-base-chinese",
    backend: str = "auto",
    batch_size: int = 16,
    pca_dim: int | None = 64,
    tfidf_dim: int = 128,
    encode_only: bool = False,
    write_events: bool = True,
) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_events()
    raw["tweet_time"] = pd.to_datetime(raw["tweet_time"], utc=True, errors="coerce")
    work = prepare_frame(raw, target)
    if len(work) < 20:
        raise SystemExit(f"有效样本 {len(work)} < 20，无法训练 score（先补 {target}）")

    print(f"编码 {len(work)} 条推文 (backend={backend}) …")
    emb, encoder_name = encode_all(
        work,
        backend=backend,
        model_name=model_name,
        batch_size=batch_size,
        tfidf_dim=tfidf_dim,
    )
    tweet_ids = work["tweet_id"].astype(str).values

    np.savez_compressed(
        EMB_NPZ,
        embeddings=emb.astype(np.float32),
        tweet_ids=tweet_ids,
        encoder=encoder_name,
    )
    print(f"  向量 -> {EMB_NPZ}  shape={emb.shape}  encoder={encoder_name}")

    if encode_only:
        return {"n": len(work), "emb_dim": emb.shape[1]}

    train_idx, test_idx = ml_time_split_indices(work, test_ratio)
    train_df = work.iloc[train_idx]
    test_df = work.iloc[test_idx]

    pca = None
    emb_train = emb[train_idx]
    if pca_dim and pca_dim < emb.shape[1]:
        pca = PCA(n_components=min(pca_dim, emb_train.shape[0] - 1, emb_train.shape[1]), random_state=42)
        pca.fit(emb_train)
        print(f"  PCA {emb.shape[1]} → {pca.n_components_} 维（仅训练集 fit）")

    model = fit_score_ridge(train_df, emb_train, alpha=alpha, pca=pca)

    pred_train = predict_scores(model, train_df, emb[train_idx])
    pred_test = predict_scores(model, test_df, emb[test_idx])
    pred_all = predict_scores(model, work, emb)

    def _ic(y_true, y_pred) -> float | None:
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        if m.sum() < 5 or np.std(y_pred[m]) == 0:
            return None
        return round(float(np.corrcoef(y_true[m], y_pred[m])[0, 1]), 4)

    metrics = {
        "target": target,
        "n_total": int(len(work)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "emb_dim": int(emb.shape[1]),
        "pca_dim": int(pca.n_components_) if pca else None,
        "ic_train": _ic(train_df["y"].values, pred_train),
        "ic_test": _ic(test_df["y"].values, pred_test),
        "encoder": encoder_name,
        "model": model_name,
    }

    out = work[
        ["tweet_id", "username", "industry", "industry_label", "tweet_time", target]
    ].copy()
    out["event_date"] = pd.to_datetime(out["tweet_time"], utc=True, errors="coerce").dt.date
    out["transformer_score"] = pred_all
    out["transformer_score_train_only"] = np.nan
    out.loc[work.index[train_idx], "transformer_score_train_only"] = pred_train
    out.loc[work.index[test_idx], "transformer_score_train_only"] = pred_test

    inf_df = attach_influence_score(work)
    out["influence"] = inf_df["influence"].values
    out["sig_transformer"] = out["transformer_score"] * out["influence"]

    out.to_csv(SCORES_CSV, index=False, encoding="utf-8-sig")
    print(f"  score -> {SCORES_CSV}")

    if write_events and EVENTS_CLEAN_CSV.exists():
        ev = pd.read_csv(EVENTS_CLEAN_CSV)
        ev["tweet_id"] = ev["tweet_id"].astype(str)
        merge_cols = out[["tweet_id", "transformer_score", "sig_transformer"]].copy()
        merge_cols["tweet_id"] = merge_cols["tweet_id"].astype(str)
        cols = ["transformer_score", "sig_transformer"]
        ev = ev.drop(columns=[c for c in cols if c in ev.columns], errors="ignore")
        ev = ev.merge(merge_cols, on="tweet_id", how="left")
        ev.to_csv(EVENTS_CLEAN_CSV, index=False, encoding="utf-8-sig")
        print(f"  回写 -> {EVENTS_CLEAN_CSV}")

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n=== Transformer score ({target}) ===")
    print(f"  训练 IC: {metrics['ic_train']} | 测试 IC: {metrics['ic_test']}")
    print(f"  meta -> {META_JSON}")
    return metrics




# ========== Diagnostics ==========

REPORT_CSV = OUTPUTS_DIR / "signal_diagnostics_report.csv"
REPORT_JSON = OUTPUTS_DIR / "signal_diagnostics_report.json"
ROLLING_IC_CSV = OUTPUTS_DIR / "signal_rolling_ic.csv"


def _time_split_dates(dates: list, train_ratio: float = 0.7) -> pd.Timestamp:
    dates = sorted(pd.to_datetime(dates).unique())
    if len(dates) < 4:
        return dates[0] if dates else pd.Timestamp("2000-01-01")
    cut = int(len(dates) * train_ratio)
    return dates[min(cut, len(dates) - 1)]


def bootstrap_ic(
    signal: np.ndarray,
    ret: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    mask = np.isfinite(signal) & np.isfinite(ret)
    s, r = signal[mask], ret[mask]
    n = len(s)
    if n < 8 or s.std() == 0 or r.std() == 0:
        return {"ic_obs": None, "ic_pvalue_two_sided": None, "ic_ci_low": None, "ic_ci_high": None}
    ic_obs = float(np.corrcoef(s, r)[0, 1])
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs, br = s[idx], r[idx]
        if bs.std() == 0 or br.std() == 0:
            continue
        boots.append(float(np.corrcoef(bs, br)[0, 1]))
    if not boots:
        return {"ic_obs": round(ic_obs, 4), "ic_pvalue_two_sided": None, "ic_ci_low": None, "ic_ci_high": None}
    boots_arr = np.array(boots)
    pval = float(2 * min((boots_arr >= 0).mean(), (boots_arr <= 0).mean()))
    return {
        "ic_obs": round(ic_obs, 4),
        "ic_pvalue_two_sided": round(pval, 4),
        "ic_ci_low": round(float(np.percentile(boots_arr, 2.5)), 4),
        "ic_ci_high": round(float(np.percentile(boots_arr, 97.5)), 4),
    }


def rolling_ic_series(
    panel: pd.DataFrame,
    *,
    window: int = 7,
    min_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for ind, sub in panel.groupby("industry"):
        sub = sub.sort_values("event_date").dropna(subset=["signal_lag1", "fwd_ret"])
        for i in range(len(sub)):
            win = sub.iloc[max(0, i - window + 1) : i + 1]
            if len(win) < min_n:
                continue
            ic = _safe_ic(win["signal_lag1"], win["fwd_ret"], min_n=min_n)
            rows.append(
                {
                    "industry": ind,
                    "event_date": sub.iloc[i]["event_date"],
                    "rolling_ic": ic,
                    "window": window,
                    "n_in_window": len(win),
                }
            )
    return pd.DataFrame(rows)


def leg_total_return(panel: pd.DataFrame) -> float:
    pnl = panel["pnl_net"].fillna(0) if "pnl_net" in panel.columns else panel["pnl_lag1"].fillna(0)
    return float((1 + pnl).prod() - 1)


def combined_equal_weight_return(panel: pd.DataFrame) -> float:
    daily = panel.groupby("event_date")["pnl_net"].mean()
    return float((1 + daily.fillna(0)).prod() - 1)


def run_diagnostics(
    *,
    signal_col: str = "sig_contrarian",
    ret_col: str = "ind_ret_1d",
    lag: int = 1,
    cost_bps: float = 5.0,
    train_ratio: float = 0.7,
    n_boot: int = 2000,
) -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tweets = build_signal_variants(load_events())
    panel = build_daily_panel(tweets, signal_col, ret_col=ret_col)
    panel = finalize_panel(panel, "pnl_lag1", "position_lag1", cost_bps, "lag1", signal_col)
    panel = panel.dropna(subset=["fwd_ret", "signal_lag1"]).copy()
    panel["event_date"] = pd.to_datetime(panel["event_date"])

    cut = _time_split_dates(panel["event_date"].tolist(), train_ratio)
    train = panel[panel["event_date"] < cut]
    test = panel[panel["event_date"] >= cut]

    rows: list[dict] = []

    def add_row(metric: str, scope: str, **kwargs) -> None:
        rows.append({"metric": metric, "scope": scope, "signal": signal_col, "ret_col": ret_col, **kwargs})

    n_dates = panel["event_date"].nunique()
    n_legs = len(panel)
    add_row(
        "sample_size",
        "all",
        n_obs=n_legs,
        n_calendar_days=int(n_dates),
        n_industries=int(panel["industry"].nunique()),
        value_json=json.dumps(panel.groupby("industry").size().astype(int).to_dict()),
    )

    ic_all = _safe_ic(panel["signal_lag1"], panel["fwd_ret"], min_n=8)
    ic_train = _safe_ic(train["signal_lag1"], train["fwd_ret"], min_n=5)
    ic_test = _safe_ic(test["signal_lag1"], test["fwd_ret"], min_n=3)
    add_row("ic_lag1", "all", value=ic_all, n_obs=n_legs)
    add_row("ic_lag1", "train", value=ic_train, n_obs=len(train))
    add_row("ic_lag1", "test", value=ic_test, n_obs=len(test))

    boot = bootstrap_ic(panel["signal_lag1"].values, panel["fwd_ret"].values, n_boot=n_boot)
    add_row("ic_bootstrap", "all", value=boot.get("ic_obs"), n_obs=n_boot, value_json=json.dumps(boot))

    comb_ret = combined_equal_weight_return(panel)
    comb_ex_crypto = combined_equal_weight_return(panel[panel["industry"] != "crypto"])
    add_row("portfolio_return_net", "all_equal_weight", value=round(comb_ret, 6), n_obs=int(n_dates))
    add_row("portfolio_return_net", "ex_crypto_equal_weight", value=round(comb_ex_crypto, 6), n_obs=int(n_dates))

    for ind, sub in panel.groupby("industry"):
        label = sub["industry_label"].iloc[0] if "industry_label" in sub.columns else ind
        ic_i = _safe_ic(sub["signal_lag1"], sub["fwd_ret"], min_n=3)
        tr = leg_total_return(sub)
        sign_hit = float((np.sign(sub["signal_lag1"]) * sub["fwd_ret"] > 0).mean())
        add_row(
            "ic_vs_leg_pnl",
            ind,
            industry_label=label,
            ic=ic_i,
            leg_total_return=round(tr, 6),
            sign_win_rate=round(sign_hit, 4),
            n_obs=len(sub),
        )

    roll = rolling_ic_series(panel, window=7)
    if not roll.empty:
        roll.to_csv(ROLLING_IC_CSV, index=False, encoding="utf-8-sig")
        for ind in roll["industry"].unique():
            r = roll[roll["industry"] == ind]["rolling_ic"].dropna()
            if len(r) == 0:
                continue
            pos_frac = float((r > 0).mean())
            add_row(
                "rolling_ic_7d",
                ind,
                value=round(float(r.mean()), 4),
                ic_positive_frac=round(pos_frac, 4),
                n_obs=len(r),
            )

    meta_path = OUTPUTS_DIR / "transformer_signal_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        add_row("ml_transformer_ic_train", "all", value=meta.get("ic_train"), n_obs=meta.get("n_train"))
        add_row("ml_transformer_ic_test", "all", value=meta.get("ic_test"), n_obs=meta.get("n_test"))

    daily_b = aggregate_daily_signals(tweets, ["sig_baseline", "sig_contrarian"])
    daily_b = enrich_daily_signals(daily_b)
    for col, name in [("sig_baseline_sum", "baseline"), ("sig_contrarian_sum", "contrarian")]:
        if col not in daily_b.columns:
            continue
        d = daily_b.sort_values(["industry", "event_date"]).copy()
        d["sig"] = d.groupby("industry")[col].shift(lag)
        ic_b = _safe_ic(d["sig"], d[ret_col], min_n=8)
        add_row("ic_lag1", f"signal_{name}", value=ic_b, n_obs=int(d["sig"].notna().sum()))

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return report


def print_summary(report: pd.DataFrame) -> None:
    print("\n=== 因子诊断（局限性量化） ===\n")
    r = report.set_index(["metric", "scope"], drop=False)

    def g(metric: str, scope: str = "all", field: str = "value"):
        try:
            row = report[(report["metric"] == metric) & (report["scope"] == scope)].iloc[0]
            return row.get(field, row.get("value"))
        except Exception:
            return None

    print(f"  样本: {g('sample_size', 'all', 'n_obs')} 条日×行业腿 | {g('sample_size', 'all', 'n_calendar_days')} 个组合日")
    print(f"  IC (lag1): 全样本 {g('ic_lag1')} | 训练 {g('ic_lag1','train')} | 测试 {g('ic_lag1','test')}")
    boot = report[report["metric"] == "ic_bootstrap"].iloc[0] if len(report[report["metric"] == "ic_bootstrap"]) else None
    if boot is not None and boot.get("value_json"):
        b = json.loads(boot["value_json"])
        print(f"  自助法 IC: {b.get('ic_obs')} | 双侧 p≈{b.get('ic_pvalue_two_sided')} | 95% CI [{b.get('ic_ci_low')}, {b.get('ic_ci_high')}]")
    print(f"  等权组合收益: 全行业 {g('portfolio_return_net','all_equal_weight'):.2%} | 去掉 crypto {g('portfolio_return_net','ex_crypto_equal_weight'):.2%}")

    print("\n  分行业 IC vs 单腿收益:")
    sub = report[report["metric"] == "ic_vs_leg_pnl"]
    if not sub.empty:
        print(
            sub[["scope", "industry_label", "ic", "leg_total_return", "sign_win_rate", "n_obs"]]
            .rename(columns={"scope": "industry"})
            .to_string(index=False)
        )

    print(f"\n  -> {REPORT_CSV}")
    print(f"  -> {ROLLING_IC_CSV} (若已生成)")



def _one_ret_col(ret_col: str | list[str]) -> str:
    """CLI --ret-col 允许 nargs='+', 诊断/回测等只需单列。"""
    if isinstance(ret_col, list):
        return ret_col[0] if ret_col else "ind_ret_1d"
    return ret_col or "ind_ret_1d"


def main() -> None:
    p = argparse.ArgumentParser(description="信号测试")
    p.add_argument("--scan", action="store_true", help="全行业+分行业 IC（推荐）")
    p.add_argument(
        "--scan-multi-horizon",
        action="store_true",
        help="多频率×分行业 IC 矩阵（5m/1h/1d/5d/20d，各 horizon 自带 lag）",
    )
    p.add_argument("--by-industry", action="store_true")
    p.add_argument("--ml-score", action="store_true", help="帖级文本向量+Ridge score")
    p.add_argument(
        "--ml-updown",
        action="store_true",
        help="行业日二分类：NLP+因子 Logistic 预测次日涨跌",
    )
    p.add_argument("--train-ratio", type=float, default=0.7, help="--ml-updown 时间切分比例")
    p.add_argument("--ml-C", type=float, default=1.0, dest="ml_C", help="Logistic 正则 1/C")
    p.add_argument("--diagnose", action="store_true", help="因子局限性诊断")
    p.add_argument("--ret-col", nargs="+", default=["ind_ret_1d"])
    p.add_argument("--industry", default=None)
    p.add_argument("--max-rank", type=int, default=None)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--min-days", type=int, default=5)
    p.add_argument("--signal", default="sig_contrarian")
    p.add_argument("--target", default="ind_ret_1d")
    p.add_argument("--backend", choices=("auto", "bert", "tfidf"), default="tfidf")
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--tfidf-dim", type=int, default=128)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--cost-bps", type=float, default=5.0)
    args = p.parse_args()

    if args.ml_updown:
        run_ml_updown(
            ret_col=_one_ret_col(args.ret_col),
            train_ratio=args.train_ratio,
            C=args.ml_C,
            industry=args.industry,
            max_rank=args.max_rank,
        )
        return
    if args.ml_score:
        run_ml_score(
            target=args.target,
            backend=args.backend,
            pca_dim=args.pca_dim if args.pca_dim > 0 else None,
            tfidf_dim=args.tfidf_dim,
            encode_only=args.encode_only,
        )
        return
    if args.diagnose:
        print_summary(
            run_diagnostics(
                signal_col=args.signal,
                ret_col=_one_ret_col(args.ret_col),
                cost_bps=args.cost_bps,
            )
        )
        return
    if args.scan_multi_horizon:
        run_multi_horizon_scan(max_rank=args.max_rank, min_days=args.min_days)
        return
    if args.scan:
        run_signal_scan(ret_cols=args.ret_col, max_rank=args.max_rank, lag=args.lag, min_days=args.min_days)
        return
    if args.by_industry:
        run_by_industry(ret_cols=args.ret_col, max_rank=args.max_rank, lag=args.lag, min_days=args.min_days)
        return
    run_lab(ret_cols=args.ret_col, industry=args.industry, max_rank=args.max_rank, lag=args.lag)


if __name__ == "__main__":
    main()
