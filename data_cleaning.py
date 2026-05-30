#!/usr/bin/env python3
"""
Step 2 — KOL 推文清洗、机器人过滤、标的解析与 yfinance 收益对齐。

读取 data/import/history_tweets.csv → 写出:
  - data/import/history_tweets_clean.csv
  - data/outputs/tweet_events_clean.csv

用法:
  python3 data_cleaning.py
  python3 data_cleaning.py --returns-only
  python3 data_cleaning.py --skip-returns
  python3 data_cleaning.py --write-master
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import emoji
import numpy as np
import pandas as pd

from data_mining import (
    HISTORY_CSV,
    IMPORT_DIR,
    INDUSTRY_ETF_MAP,
    KOL_RANKED_CSV,
    MASTER_CSV,
    OUTPUTS_DIR,
    USERS_CSV,
    USERNAME_RE,
    classify_text,
    score_sentiment,
)

# -----------------------------------------------------------------------------
# Paths (clean artifacts)
# -----------------------------------------------------------------------------

HISTORY_CLEAN_CSV = IMPORT_DIR / "history_tweets_clean.csv"
EVENTS_CLEAN_CSV = OUTPUTS_DIR / "tweet_events_clean.csv"

# Cashtag symbols that confuse or break yfinance — fall back to industry ETF.
CASHTAG_SKIP_YF: frozenset[str] = frozenset(
    {
        "HYPE",
        "PUMP",
        "FARM",
        "WOJAK",
        "NPC",
        "X",  # yfinance 无有效标的，回退行业 ETF
        "BNB",
        "WLFI",
        "CHIP",
        "CTUSD",
        "ESC",
        "OKB",
        "POLY",
        "SIVE",
        "SOI",
        "TAO",
        "K",   # 单字母 cashtag，yfinance 无效
        "POD",
    }
)

TWITTER_EPOCH_MS = 1288834974657

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_RT_RE = re.compile(r"^\s*RT\s+@", re.IGNORECASE)

# Heuristic thresholds (documented in 项目介绍)
_MAX_POSTS_PER_DAY_BOT = 12
_MIN_LIKES_LOW = 3
_MIN_TEXT_LEN_EVENT = 8
_INTRADAY_LOOKBACK_DAYS = 59
_YF_SLEEP_SEC = 0.12


# -----------------------------------------------------------------------------
# Time repair (Snowflake)
# -----------------------------------------------------------------------------


def snowflake_to_utc(tweet_id: Any) -> pd.Timestamp | None:
    """Decode tweet snowflake id → UTC timestamp (works for numeric Twitter/X ids)."""
    try:
        tid = int(str(tweet_id).strip())
    except (TypeError, ValueError):
        return None
    if tid <= 0:
        return None
    ms = (tid >> 22) + TWITTER_EPOCH_MS
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def repair_tweet_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every row has a parsable UTC tweet_time.
    Priority: existing CSV value → snowflake(tweet_id) → ingested_at.
    Sets `time_source` to csv | snowflake | ingested | missing.
    """
    out = df.copy()
    if "tweet_time" not in out.columns:
        out["tweet_time"] = pd.NaT

    times: list[pd.Timestamp | Any] = []
    sources: list[str] = []

    for i in range(len(out)):
        raw_tt = out["tweet_time"].iloc[i]
        tid = out["tweet_id"].iloc[i]
        ts = pd.to_datetime(raw_tt, utc=True, errors="coerce")
        if pd.notna(ts):
            times.append(ts)
            sources.append("csv")
            continue

        sf = snowflake_to_utc(tid)
        if sf is not None:
            times.append(sf)
            sources.append("snowflake")
            continue

        if "ingested_at" in out.columns:
            ing = pd.to_datetime(out["ingested_at"].iloc[i], utc=True, errors="coerce")
            if pd.notna(ing):
                times.append(ing)
                sources.append("ingested")
                continue

        times.append(pd.NaT)
        sources.append("missing")

    out["tweet_time"] = times
    out["time_source"] = sources
    return out


# -----------------------------------------------------------------------------
# Load / merge KOL metadata
# -----------------------------------------------------------------------------


def _read_ranked_meta() -> pd.DataFrame | None:
    if not KOL_RANKED_CSV.exists():
        return None
    rk = pd.read_csv(KOL_RANKED_CSV)
    cols = [c for c in ("username", "nickname", "industry", "industry_label", "rank_in_industry", "views_seed", "influence_score") if c in rk.columns]
    if "username" not in cols:
        return None
    m = rk[cols].copy()
    if "rank_in_industry" in m.columns:
        m = m.rename(columns={"rank_in_industry": "kol_rank_in_industry"})
    return m


def _views_seed_from_users() -> pd.Series:
    """Fallback mapping username(str.lower) -> views_seed when kol_ranked missing."""
    if not USERS_CSV.exists():
        return pd.Series(dtype=float)
    u = pd.read_csv(USERS_CSV)
    if "首页地址" not in u.columns or "阅读数" not in u.columns:
        return pd.Series(dtype=float)
    u["_u"] = u["首页地址"].astype(str).str.extract(USERNAME_RE, expand=False).str.lower()
    u["_v"] = pd.to_numeric(u["阅读数"], errors="coerce").fillna(0)
    return u.drop_duplicates("_u").set_index("_u")["_v"]


def load_history(path: Path | None = None) -> pd.DataFrame:
    """Load raw history CSV and attach industry / influence columns when absent."""
    src = path or HISTORY_CSV
    if not src.exists():
        raise SystemExit(f"缺少 {src}，请先抓取 history")
    df = pd.read_csv(src)
    meta = _read_ranked_meta()
    if meta is not None:
        df["username"] = df["username"].astype(str)
        m = meta.copy()
        m["username"] = m["username"].astype(str)
        merge_cols = [c for c in m.columns if c != "username"]
        for c in merge_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df.merge(m[["username", *merge_cols]], on="username", how="left", suffixes=("", "_rk"))
        for c in merge_cols:
            rk_col = f"{c}_rk"
            if rk_col in df.columns:
                df[c] = df[c].where(df[c].notna(), df[rk_col])
                df.drop(columns=[rk_col], inplace=True)
    else:
        vw = _views_seed_from_users()
        if not vw.empty:
            df["views_seed"] = df["username"].astype(str).str.lower().map(vw)

    if "kol_rank_in_industry" in df.columns:
        df["kol_rank_in_industry"] = pd.to_numeric(df["kol_rank_in_industry"], errors="coerce")
    if "views_seed" in df.columns:
        df["views_seed"] = pd.to_numeric(df["views_seed"], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------


def clean_tweet_text(text: Any) -> str:
    """Strip URLs / emoji noise / collapse whitespace."""
    t = str(text or "")
    t = _URL_RE.sub("", t)
    try:
        t = emoji.replace_emoji(t, replace="")
    except Exception:
        pass
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _text_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Ticker resolution
# -----------------------------------------------------------------------------


def _first_cashtag(text: str) -> str | None:
    m = re.search(r"\$([A-Za-z]{1,6})\b", text or "")
    return m.group(1).upper() if m else None


def _normalize_equity_symbol(sym: str) -> str:
    s = sym.upper().strip()
    if s in {"BTC", "ETH"}:
        return f"{s}-USD"
    return s


def _should_skip_yf_symbol(sym: str) -> bool:
    """单字母/黑名单 cashtag 不走 yfinance，改用行业 ETF。"""
    if not sym:
        return True
    base = sym.upper().replace("-USD", "").replace("-USDT", "").strip()
    if base in CASHTAG_SKIP_YF:
        return True
    if len(base) <= 2 and base.isalpha():
        return True
    return False


def _sanitize_ticker(sym: str, ind_etf: str) -> str:
    sym = str(sym or "").strip()
    if not sym or _should_skip_yf_symbol(sym):
        return ind_etf if ind_etf else ""
    return sym


def _refresh_tickers_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """按当前正文重新解析 ticker（returns-only 时修正历史脏符号如 K/POD）。"""
    out = df.copy()
    raw_col = "text_raw" if "text_raw" in out.columns else "text"
    if raw_col not in out.columns:
        return out
    inds = out["industry"] if "industry" in out.columns else pd.Series(["unclassified"] * len(out))
    tickers, srcs, inds_t = [], [], []
    for raw, ind in zip(out[raw_col].fillna("").astype(str), inds.fillna("unclassified")):
        tkr, src, ind_t = _resolve_tickers(raw, ind)
        tickers.append(tkr)
        srcs.append(src)
        inds_t.append(ind_t)
    out["ticker"] = tickers
    out["ticker_src"] = srcs
    out["industry_ticker"] = inds_t
    return out


def _resolve_tickers(text_raw: str, industry: str | float | None) -> tuple[str, str, str]:
    """
    Returns (ticker, ticker_src, industry_ticker).
    ticker_src ∈ {cashtag, industry, none}
    """
    ind = str(industry or "unclassified").strip()
    ind_etf = INDUSTRY_ETF_MAP.get(ind) or ""
    ind_etf = str(ind_etf) if ind_etf else ""

    cashtag = _first_cashtag(text_raw or "")
    info = classify_text(text_raw or "")
    tickers = list(info.get("tickers") or [])

    if cashtag:
        if _should_skip_yf_symbol(cashtag):
            if ind_etf:
                return ind_etf, "industry", ind_etf
            return "", "none", ind_etf
        sym = _normalize_equity_symbol(cashtag)
        return sym, "cashtag", ind_etf

    if tickers:
        sym0 = str(tickers[0])
        if _should_skip_yf_symbol(sym0):
            if ind_etf:
                return ind_etf, "industry", ind_etf
            return "", "none", ind_etf
        return sym0, "cashtag" if "$" in text_raw else "industry", ind_etf

    if ind_etf:
        return ind_etf, "industry", ind_etf

    return "", "none", ind_etf


# -----------------------------------------------------------------------------
# Bot / quality flags
# -----------------------------------------------------------------------------


def _detect_rt(text_raw: str) -> bool:
    t = str(text_raw or "")
    return bool(_RT_RE.match(t)) or t.lstrip().startswith("RT：")


def annotate_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate text_clean, hashing, RT flags, duplicate markers,
    frequency / engagement heuristics, and ticker columns.
    """
    out = df.copy()
    out["text_raw"] = out["text"].astype(str)
    out["text_clean"] = out["text_raw"].map(clean_tweet_text)
    out["text_hash"] = out["text_clean"].map(_text_hash)

    out["likes_num"] = pd.to_numeric(out.get("likes"), errors="coerce")
    out["is_rt"] = out["text_raw"].map(_detect_rt)

    out = out.sort_values(["username", "tweet_time"])
    dup_key = out["username"].astype(str) + "|" + out["text_clean"]
    out["is_duplicate"] = dup_key.duplicated(keep="first")

    tt = pd.to_datetime(out["tweet_time"], utc=True, errors="coerce")
    out["event_date"] = tt.dt.date

    out = out.drop(columns=["posts_that_day"], errors="ignore")
    posts = out.groupby(["username", "event_date"], observed=False).size().rename("posts_that_day")
    out = out.merge(posts.reset_index(), on=["username", "event_date"], how="left")
    out["freq_abnormal"] = out["posts_that_day"] >= _MAX_POSTS_PER_DAY_BOT

    med_views = float(pd.to_numeric(out["views_seed"], errors="coerce").median() or 1.0)
    out["likes_too_low"] = out["likes_num"].fillna(0) < _MIN_LIKES_LOW
    out["influence_low"] = pd.to_numeric(out["views_seed"], errors="coerce").fillna(0) < max(
        500.0, 0.15 * med_views
    )

    flags: list[str] = []
    for row in out.itertuples(index=False):
        r: list[str] = []
        if getattr(row, "freq_abnormal", False):
            r.append("freq")
        if getattr(row, "likes_too_low", False):
            r.append("likes_low")
        if getattr(row, "influence_low", False):
            r.append("inf_low")
        if getattr(row, "is_duplicate", False):
            r.append("dup")
        flags.append(";".join(r))
    out["bot_flags"] = flags

    bad = (
        out["freq_abnormal"].fillna(False)
        | out["is_duplicate"].fillna(False)
        | (
            out["likes_too_low"].fillna(False)
            & (out["text_clean"].str.len() < 24)
        )
    )
    out["is_bot"] = bad
    out["is_snapshot"] = False

    tickers = []
    srcs = []
    ind_syms = []
    for raw, ind in zip(out["text_raw"].tolist(), out.get("industry", pd.Series(["unclassified"] * len(out))).tolist()):
        tkr, src, ind_t = _resolve_tickers(raw, ind)
        tickers.append(tkr)
        srcs.append(src)
        ind_syms.append(ind_t)
    out["ticker"] = tickers
    out["ticker_src"] = srcs
    out["industry_ticker"] = ind_syms

    return out


def _apply_lexicon_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labs: list[str] = []
    scores: list[float] = []
    for t in out["text_clean"].fillna("").tolist():
        lab, sc = score_sentiment(t)
        labs.append(lab)
        scores.append(sc)
    out["sentiment"] = labs
    out["sentiment_score"] = scores
    return out


# -----------------------------------------------------------------------------
# Market data helpers (yfinance)
# -----------------------------------------------------------------------------


def _is_crypto(sym: str) -> bool:
    u = sym.upper()
    return "-USD" in u or "-USDT" in u or u.endswith("=X")


def _yf_sleep() -> None:
    time.sleep(_YF_SLEEP_SEC)


def _download_daily(sym: str) -> pd.DataFrame:
    import yfinance as yf

    _yf_sleep()
    end = datetime.now(timezone.utc) + pd.Timedelta(days=2)
    start = datetime.now(timezone.utc) - pd.Timedelta(days=420)
    try:
        px = yf.download(
            sym,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if px is None or px.empty:
        return pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(-1, axis=1)
    return px


def _download_5m(sym: str) -> pd.DataFrame:
    import yfinance as yf

    _yf_sleep()
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=_INTRADAY_LOOKBACK_DAYS)
    try:
        px = yf.download(
            sym,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="5m",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if px is None or px.empty:
        return pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(-1, axis=1)
    return px


def _event_calendar_day(ts: pd.Timestamp, sym: str) -> date:
    ts = ts if ts.tzinfo else ts.tz_localize("UTC")
    if _is_crypto(sym):
        return ts.date()
    return ts.tz_convert("America/New_York").date()


def _daily_slice(close: pd.Series, event_day: date) -> tuple[int | None, list[date]]:
    idx = close.index
    dates: list[date] = []
    for x in idx:
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        dates.append(ts.date())
    pos_list = [i for i, d in enumerate(dates) if d <= event_day]
    if not pos_list:
        return None, dates
    return pos_list[-1], dates


def _forward_daily_returns(close: pd.Series, pos: int) -> tuple[dict[str, float | None], list[str]]:
    out: dict[str, float | None] = {}
    pending: list[str] = []
    c0 = float(close.iloc[pos])
    horizons = {"ret_1d": 1, "ret_5d": 5, "ret_20d": 20}
    if c0 == 0:
        return {k: None for k in horizons}, ["bad_anchor"]

    for key, h in horizons.items():
        j = pos + h
        if j >= len(close):
            out[key] = None
            pending.append(f"{key}_pending")
        else:
            out[key] = float(close.iloc[j] / c0 - 1.0)
    return out, pending


def _intraday_pack(
    sym: str,
    event_ts: pd.Timestamp,
    df5: pd.DataFrame,
) -> tuple[float | None, float | None, list[str]]:
    notes: list[str] = []
    if df5.empty:
        return None, None, ["no_intraday"]

    close = df5["Close"] if "Close" in df5.columns else df5.iloc[:, -1]
    idx = pd.DatetimeIndex(pd.to_datetime(df5.index, utc=True))

    ets = event_ts if event_ts.tzinfo else event_ts.tz_localize("UTC")
    ets = ets.tz_convert("UTC")

    bar_idx = int(idx.searchsorted(ets))
    bar_idx = min(max(bar_idx - 1, 0), len(close) - 1)
    c0 = float(close.iloc[bar_idx])

    def fwd(k: int, label: str) -> float | None:
        j = bar_idx + k
        if j >= len(close):
            notes.append(f"{label}_pending")
            return None
        return float(close.iloc[j] / c0 - 1.0) if c0 else None

    r5 = fwd(1, "ret_5m")
    r1h = fwd(12, "ret_1h")
    return r5, r1h, notes


def _sym_mask(df: pd.DataFrame, col: str, sym: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip() == sym


def _needs_intraday(df: pd.DataFrame, sym: str, cutoff: pd.Timestamp) -> bool:
    m = _sym_mask(df, "ticker", sym) | _sym_mask(df, "industry_ticker", sym)
    if not m.any():
        return False
    tt = pd.to_datetime(df.loc[m, "tweet_time"], utc=True, errors="coerce")
    return bool((tt >= cutoff).any())


def _price_batch_cache(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    syms: set[str] = set()
    for c in ("ticker", "industry_ticker"):
        if c in df.columns:
            for x in df[c].dropna().tolist():
                s = str(x).strip()
                if s and not _should_skip_yf_symbol(s):
                    syms.add(s)
    daily: dict[str, pd.DataFrame] = {}
    intra: dict[str, pd.DataFrame] = {}
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=_INTRADAY_LOOKBACK_DAYS)

    import logging

    yf_log = logging.getLogger("yfinance")
    old_level = yf_log.level
    yf_log.setLevel(logging.CRITICAL)

    for sym in sorted(syms):
        daily[sym] = _download_daily(sym)
        need_intra = _needs_intraday(df, sym, cutoff)
        intra[sym] = _download_5m(sym) if need_intra else pd.DataFrame()

    yf_log.setLevel(old_level)

    return daily, intra


def aligned_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate close_at_event, ret_5m/1h/1d/5d/20d (+ industry twins) using yfinance.
    Appends human-readable align_note / ind_align_note lists.
    """
    out = _refresh_tickers_from_text(df.copy()).reset_index(drop=True)
    cols = [
        "close_at_event",
        "ret_5m",
        "ret_1h",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "align_note",
        "ind_close_at_event",
        "ind_ret_5m",
        "ind_ret_1h",
        "ind_ret_1d",
        "ind_ret_5d",
        "ind_ret_20d",
        "ind_align_note",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan if "note" not in c else ""

    daily_cache, intra_cache = _price_batch_cache(out)
    now = pd.Timestamp.now(tz="UTC")
    stale_cut = now - pd.Timedelta(days=_INTRADAY_LOOKBACK_DAYS)

    cloc = out.columns.get_loc

    for pos in range(len(out)):
        row = out.iloc[pos]
        m_notes: list[str] = []
        i_notes: list[str] = []

        ts = pd.to_datetime(row.get("tweet_time"), utc=True, errors="coerce")
        sym = str(row.get("ticker") or "").strip()
        ind_sym = str(row.get("industry_ticker") or "").strip()

        close_ev = np.nan
        r5 = r1h = np.nan
        dret: dict[str, float | None] = {"ret_1d": np.nan, "ret_5d": np.nan, "ret_20d": np.nan}

        if not sym:
            m_notes.append("no_ticker")
        else:
            dfx = daily_cache.get(sym, pd.DataFrame())
            if dfx.empty or "Close" not in dfx.columns:
                m_notes.append("no_daily_data")
            else:
                close_s = dfx["Close"].dropna()
                ev_day = _event_calendar_day(ts, sym) if pd.notna(ts) else date.today()
                anchor_pos, _ = _daily_slice(close_s, ev_day)
                if anchor_pos is None:
                    m_notes.append("no_anchor")
                else:
                    close_ev = float(close_s.iloc[anchor_pos])
                    dret, pend = _forward_daily_returns(close_s, anchor_pos)
                    m_notes.extend(pend)

            df5 = intra_cache.get(sym, pd.DataFrame())
            if pd.notna(ts):
                if ts < stale_cut:
                    m_notes.append("intraday_stale")
                    r5 = r1h = np.nan
                elif df5 is None or df5.empty:
                    m_notes.append("no_intraday_window")
                    r5 = r1h = np.nan
                else:
                    r5, r1h, irn = _intraday_pack(sym, ts, df5)
                    m_notes.extend(irn)
            else:
                r5 = r1h = np.nan

        iclose = np.nan
        ir5 = ir1h = np.nan
        idret = {"ret_1d": np.nan, "ret_5d": np.nan, "ret_20d": np.nan}

        if not ind_sym:
            i_notes.append("no_industry")
        else:
            dfi = daily_cache.get(ind_sym, pd.DataFrame())
            if dfi.empty or "Close" not in dfi.columns:
                i_notes.append("no_daily_data")
            else:
                cls_i = dfi["Close"].dropna()
                ev_day_i = _event_calendar_day(ts, ind_sym) if pd.notna(ts) else date.today()
                pos_i, _ = _daily_slice(cls_i, ev_day_i)
                if pos_i is None:
                    i_notes.append("no_anchor")
                else:
                    iclose = float(cls_i.iloc[pos_i])
                    idret, pend_i = _forward_daily_returns(cls_i, pos_i)
                    i_notes.extend(pend_i)

            df5i = intra_cache.get(ind_sym, pd.DataFrame())
            if pd.notna(ts):
                if ts < stale_cut:
                    i_notes.append("intraday_stale")
                    ir5 = ir1h = np.nan
                elif df5i is None or df5i.empty:
                    i_notes.append("no_intraday_window")
                    ir5 = ir1h = np.nan
                else:
                    ir5, ir1h, irn2 = _intraday_pack(ind_sym, ts, df5i)
                    i_notes.extend(irn2)
            else:
                ir5 = ir1h = np.nan

        out.iloc[pos, cloc("close_at_event")] = close_ev
        out.iloc[pos, cloc("ret_5m")] = r5
        out.iloc[pos, cloc("ret_1h")] = r1h
        out.iloc[pos, cloc("ret_1d")] = dret["ret_1d"]
        out.iloc[pos, cloc("ret_5d")] = dret["ret_5d"]
        out.iloc[pos, cloc("ret_20d")] = dret["ret_20d"]
        out.iloc[pos, cloc("align_note")] = ",".join(sorted(set(m_notes)))

        out.iloc[pos, cloc("ind_close_at_event")] = iclose
        out.iloc[pos, cloc("ind_ret_5m")] = ir5
        out.iloc[pos, cloc("ind_ret_1h")] = ir1h
        out.iloc[pos, cloc("ind_ret_1d")] = idret["ret_1d"]
        out.iloc[pos, cloc("ind_ret_5d")] = idret["ret_5d"]
        out.iloc[pos, cloc("ind_ret_20d")] = idret["ret_20d"]
        out.iloc[pos, cloc("ind_align_note")] = ",".join(sorted(set(i_notes)))

    return out


# -----------------------------------------------------------------------------
# Event selection & orchestration
# -----------------------------------------------------------------------------


def select_event_rows(df: pd.DataFrame, *, drop_bots: bool = True) -> pd.DataFrame:
    """Research-ready subset: non-bot rows with a resolved ticker & enough text."""
    m = df["text_clean"].fillna("").str.len() >= _MIN_TEXT_LEN_EVENT
    m &= df["ticker"].fillna("").astype(str).str.len() > 0
    if drop_bots:
        m &= ~df["is_bot"].fillna(False)
    return df.loc[m].copy()


def enrich_returns_only(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Repair timestamps then refresh market columns only (NLP columns preserved)."""
    base = df if df is not None else pd.read_csv(HISTORY_CLEAN_CSV)
    base = repair_tweet_times(base)
    base = _refresh_tickers_from_text(base)
    base = aligned_returns(base)
    return base


def enrich_returns_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Full numeric enrichment: timestamps → cleaning annotations → lexicon sentiment → returns."""
    base = repair_tweet_times(df)
    base = annotate_cleaning(base)
    base = _apply_lexicon_sentiment(base)
    base = aligned_returns(base)
    return base


def refresh_returns_only(*, write_master: bool = False) -> pd.DataFrame:
    """CLI `--returns-only`: load clean CSV, repair times, refresh yfinance fields, export."""
    if not HISTORY_CLEAN_CSV.exists():
        raise SystemExit(f"缺少 {HISTORY_CLEAN_CSV}，请先完整运行 python3 data_cleaning.py")
    df = enrich_returns_only(pd.read_csv(HISTORY_CLEAN_CSV))
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_CLEAN_CSV, index=False, encoding="utf-8-sig")
    events = select_event_rows(df, drop_bots=True)
    events.to_csv(EVENTS_CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"returns-only: {len(df)} 行 -> {HISTORY_CLEAN_CSV} | 事件 {len(events)} -> {EVENTS_CLEAN_CSV}")
    if write_master:
        from data_mining import build_master_from_clean

        build_master_from_clean()
    return df


def run_cleaning(*, skip_returns: bool = False, write_master: bool = False) -> pd.DataFrame:
    """End-to-end Step2 from raw history."""
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_history()
    df = repair_tweet_times(df)
    df = annotate_cleaning(df)
    df = _apply_lexicon_sentiment(df)
    if skip_returns:
        for c in (
            "close_at_event",
            "ret_5m",
            "ret_1h",
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "align_note",
            "ind_close_at_event",
            "ind_ret_5m",
            "ind_ret_1h",
            "ind_ret_1d",
            "ind_ret_5d",
            "ind_ret_20d",
            "ind_align_note",
        ):
            if c not in df.columns:
                df[c] = np.nan if "note" not in c else ""
    else:
        df = aligned_returns(df)

    df.to_csv(HISTORY_CLEAN_CSV, index=False, encoding="utf-8-sig")
    events = select_event_rows(df, drop_bots=True)
    events.to_csv(EVENTS_CLEAN_CSV, index=False, encoding="utf-8-sig")

    print(f"clean: {len(df)} 行 -> {HISTORY_CLEAN_CSV}")
    print(f"events: {len(events)} 行 -> {EVENTS_CLEAN_CSV}")

    if write_master:
        from data_mining import build_master_from_clean

        build_master_from_clean()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Step2: 推文清洗 + yfinance 对齐")
    p.add_argument("--returns-only", action="store_true", help="仅刷新行情收益列（保留 NLP）")
    p.add_argument("--skip-returns", action="store_true", help="跳过 yfinance（快速清洗）")
    p.add_argument("--write-master", action="store_true", help="完成后重建 kol_master.csv")
    args = p.parse_args()

    if args.returns_only:
        refresh_returns_only(write_master=args.write_master)
        return

    run_cleaning(skip_returns=args.skip_returns, write_master=args.write_master)


if __name__ == "__main__":
    main()
