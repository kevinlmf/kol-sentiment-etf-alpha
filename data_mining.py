"""
Quant Interview — KOL 数据挖掘统一入口 (users.csv)

流程:
  1) 行业分类 + 每行业 Top-K (阅读数)
  2) Twikit 抓取近 N 天推文 → history_tweets.csv
  3) 回写 users.csv / 构建 kol_master.csv / IC 回测

推荐一键:
  python3 data_mining.py --pipeline

分步:
  python3 data_mining.py --classify-only --top-k 5
  python3 data_mining.py --fetch-twikit --days 30 --resume --sleep-user 8
  python3 data_mining.py --enrich-only --days 30
  python3 data_mining.py --master
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent
USERS_CSV = ROOT / "users.csv"
IMPORT_DIR = ROOT / "data" / "import"
OUTPUTS_DIR = ROOT / "data" / "outputs"
HISTORY_CSV = IMPORT_DIR / "history_tweets.csv"
COOKIES_JSON = IMPORT_DIR / "twikit_cookies.json"
FAILED_TXT = IMPORT_DIR / "fetch_failed_users.txt"
KOL_RANKED_CSV = OUTPUTS_DIR / "kol_ranked.csv"
TOP_KOL_CSV = OUTPUTS_DIR / "top_kol_panel.csv"
MASTER_CSV = OUTPUTS_DIR / "kol_master.csv"
EVENTS_CLEAN_CSV = OUTPUTS_DIR / "tweet_events_clean.csv"
DAILY_FACTOR_CSV = OUTPUTS_DIR / "daily_factor.csv"
BACKTEST_CSV = OUTPUTS_DIR / "backtest_summary.csv"

INDUSTRY_ETF_MAP = {
    "ai_tech": "QQQ",
    "ai_tools": "QQQ",
    "semiconductor": "SOXX",
    "crypto": "BTC-USD",
    "macro_us": "SPY",
    "ev": "TSLA",
    "cloud": "MSFT",
    "consumer_tech": "QQQ",
    "general_ai": "QQQ",
    "unclassified": None,
}

FORWARD_HORIZONS = (1, 3, 5, 10, 20)

# IC 回测：情绪列 × 收益列（与 data_cleaning 对齐）
IC_SENTIMENT_COLS = ("sentiment_score_nlp", "sentiment_score")
IC_RETURN_COLS = (
    "ind_ret_5m",
    "ind_ret_1h",
    "ind_ret_1d",
    "ind_ret_5d",
    "ind_ret_20d",
    "ret_5m",
    "ret_1h",
    "ret_1d",
    "ret_5d",
    "ret_20d",
)

_POS = ("利好", "看涨", "上涨", "bull", "bullish", "beat", "strong", "超预期", "moon")
_NEG = ("利空", "看跌", "下跌", "bear", "bearish", "crash", "miss", "weak", "暴跌")
_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
USERNAME_RE = re.compile(r"x\.com/([^/?#]+)", re.IGNORECASE)

COL_NAME, COL_URL, COL_VIEWS = "昵称", "首页地址", "阅读数"
COL_TEXT, COL_TIME, COL_UPDATED = "最近一条推文", "最近推文时间", "数据更新于"
COL_INDUSTRY, COL_INDUSTRY_LABEL = "行业", "行业标签"
COL_RANK, COL_TOP5 = "行业排名", "行业Top5"

HISTORY_COLS = [
    "tweet_id",
    "username",
    "tweet_time",
    "text",
    "likes",
    "retweets",
    "replies",
    "views",
    "source",
]

INDUSTRY_MAP = {
    "semiconductor": "semiconductor",
    "半导体": "semiconductor",
    "芯片": "semiconductor",
    "nvidia": "semiconductor",
    "英伟达": "semiconductor",
    "gpu": "semiconductor",
    "ai": "ai_tech",
    "人工智能": "ai_tech",
    "大模型": "ai_tech",
    "llm": "ai_tech",
    "openai": "ai_tech",
    "gpt": "ai_tech",
    "claude": "ai_tech",
    "gemini": "ai_tech",
    "agent": "ai_tools",
    "mcp": "ai_tools",
    "cursor": "ai_tools",
    "copilot": "ai_tools",
    "crypto": "crypto",
    "比特币": "crypto",
    "以太坊": "crypto",
    "区块链": "crypto",
    "btc": "crypto",
    "eth": "crypto",
    "web3": "crypto",
    "defi": "crypto",
    "特斯拉": "ev",
    "电动车": "ev",
    "tsla": "ev",
    "美联储": "macro_us",
    "加息": "macro_us",
    "降息": "macro_us",
    "宏观": "macro_us",
    "苹果": "consumer_tech",
    "iphone": "consumer_tech",
    "aapl": "consumer_tech",
    "微软": "cloud",
    "azure": "cloud",
    "msft": "cloud",
    "谷歌": "consumer_tech",
    "google": "consumer_tech",
}

INDUSTRY_LABELS = {
    "ai_tech": "AI / 大模型",
    "ai_tools": "AI 工具 / Agent",
    "semiconductor": "半导体",
    "crypto": "加密",
    "macro_us": "美股宏观",
    "ev": "新能源 / 电动车",
    "cloud": "云计算",
    "consumer_tech": "消费科技",
    "general_ai": "泛科技",
    "unclassified": "未分类",
}

_INDUSTRY_PATTERNS = sorted(INDUSTRY_MAP.keys(), key=len, reverse=True)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def _clean_text(text: str) -> str:
    t = re.sub(r"https?://\S+", "", str(text))
    return re.sub(r"\s+", " ", t).strip()


def _cloudflare_hint() -> str:
    return (
        f"X 被 Cloudflare 拦截。请用浏览器登录 x.com，导出 cookies 到:\n"
        f"  {COOKIES_JSON}\n"
        f'  格式: {{"auth_token": "...", "ct0": "..."}}\n'
        f"  (Chrome DevTools → Application → Cookies → x.com)"
    )


def _cookies_path() -> Path:
    return Path(os.environ.get("X_COOKIES_FILE", str(COOKIES_JSON)))


def _cookies_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    keys = {k.lower() for k in data} if isinstance(data, dict) else set()
    return "auth_token" in keys and "ct0" in keys


def _patch_twikit_keybyte() -> None:
    import re as _re

    from twikit.x_client_transaction import transaction as tr

    tr.INDICES_REGEX = _re.compile(
        r"\(?(\w+)\[(\d{1,2})\]\s*,\s*16\)?",
        _re.VERBOSE | _re.MULTILINE,
    )

    async def init_with_fallback(self, session, headers):
        home = await tr.handle_x_migration(session, headers)
        self.home_page_response = self.validate_response(home)
        try:
            (
                self.DEFAULT_ROW_INDEX,
                self.DEFAULT_KEY_BYTES_INDICES,
            ) = await self.get_indices(self.home_page_response, session, headers)
            self.key = self.get_key(response=self.home_page_response)
            self.key_bytes = self.get_key_bytes(key=self.key)
            self.animation_key = self.get_animation_key(
                key_bytes=self.key_bytes, response=self.home_page_response
            )
        except Exception:
            self.key = "mentUisV_1yPzH_3IcNS_nRaF_R_b"
            self.key_bytes = self.get_key_bytes(self.key)
            self.DEFAULT_ROW_INDEX = 2
            self.DEFAULT_KEY_BYTES_INDICES = [13, 14, 7]
            self.animation_key = "0"

    tr.ClientTransaction.init = init_with_fallback


def _patch_twikit_user() -> None:
    """补齐 X API 变更后 User 对象缺失字段，避免 KeyError。"""
    from twikit.user import User

    _orig = User.__init__
    defaults = {
        "created_at": "",
        "name": "",
        "screen_name": "",
        "profile_image_url_https": "",
        "location": "",
        "description": "",
        "pinned_tweet_ids_str": [],
        "verified": False,
        "possibly_sensitive": False,
        "can_dm": False,
        "can_media_tag": False,
        "want_retweets": False,
        "default_profile": True,
        "default_profile_image": False,
        "has_custom_timelines": False,
        "followers_count": 0,
        "fast_followers_count": 0,
        "favourites_count": 0,
        "friends_count": 0,
        "listed_count": 0,
        "media_count": 0,
        "normal_followers_count": 0,
        "statuses_count": 0,
        "is_translator": False,
        "translator_type": "none",
        "withheld_in_countries": [],
    }

    def _safe_init(self, client, data):
        legacy = dict(data.get("legacy") or {})
        for k, v in defaults.items():
            legacy.setdefault(k, v if not isinstance(v, list) else list(v))
        entities = legacy.setdefault("entities", {})
        desc_ent = entities.setdefault("description", {})
        desc_ent.setdefault("urls", [])
        url_ent = entities.setdefault("url", {})
        url_ent.setdefault("urls", [])
        legacy["entities"] = entities
        data["legacy"] = legacy
        data.setdefault("is_blue_verified", False)
        data.setdefault("rest_id", legacy.get("id_str") or data.get("rest_id", ""))
        _orig(self, client, data)

    User.__init__ = _safe_init


def read_users() -> pd.DataFrame:
    df = pd.read_csv(USERS_CSV)
    df["_username"] = df[COL_URL].astype(str).str.extract(USERNAME_RE, expand=False)
    if df["_username"].isna().any():
        bad = df.loc[df["_username"].isna(), COL_URL].head(3).tolist()
        raise ValueError(f"无法解析 x.com 用户名: {bad}")
    return df


def classify_profile(nickname: str, username: str, latest: str) -> str:
    """规则分类：昵称 + 用户名 + 最近推文。"""
    blob = f"{nickname} {username} {latest}".lower()
    for kw in _INDUSTRY_PATTERNS:
        if kw in blob:
            return INDUSTRY_MAP[kw]
    return "unclassified"


def classify_text(text: str) -> dict:
    lower = (text or "").lower()
    tickers: list[str] = []
    seen: set[str] = set()
    for tag in _CASHTAG.findall(text or ""):
        sym = tag.upper()
        t = f"{sym}-USD" if sym in {"ETH", "BTC"} else sym
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    industries: list[str] = []
    for kw in _INDUSTRY_PATTERNS:
        if kw in lower:
            ind = INDUSTRY_MAP[kw]
            if ind not in industries:
                industries.append(ind)
    primary = industries[0] if industries else "unclassified"
    if not tickers and primary in INDUSTRY_ETF_MAP and INDUSTRY_ETF_MAP[primary]:
        tickers = [INDUSTRY_ETF_MAP[primary]]
    return {"tickers": tickers[:3], "primary_industry": primary}


def rank_kols_by_industry(users: pd.DataFrame, top_k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按行业分组，用附件「阅读数」作影响力，每行业取 Top-K。
    返回 (全量带排名, Top-K 面板)。
    """
    rows: list[dict] = []
    for _, row in users.iterrows():
        uname = str(row["_username"])
        ind = classify_profile(
            str(row.get(COL_NAME, "")),
            uname,
            str(row.get(COL_TEXT, "")),
        )
        rows.append(
            {
                "nickname": row.get(COL_NAME, ""),
                "username": uname,
                "profile_url": row.get(COL_URL, ""),
                "views_seed": int(
                    pd.to_numeric(row.get(COL_VIEWS, 0), errors="coerce") or 0
                ),
                "latest_tweet_snippet": str(row.get(COL_TEXT, ""))[:100],
                "industry": ind,
                "industry_label": INDUSTRY_LABELS.get(ind, ind),
            }
        )
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(["industry", "views_seed"], ascending=[True, False])
    ranked["rank_in_industry"] = ranked.groupby("industry").cumcount() + 1
    ranked["influence_score"] = ranked.groupby("industry")["views_seed"].transform(
        lambda s: round(100.0 * s / s.max(), 2) if s.max() else 0.0
    )
    ranked["is_top_k"] = ranked["rank_in_industry"] <= top_k
    top = ranked[ranked["is_top_k"]].copy()
    return ranked, top


def write_industry_to_users(ranked: pd.DataFrame) -> None:
    """把行业、排名写回 users.csv。"""
    users = read_users()
    meta = (
        ranked.set_index("username")[["industry", "industry_label", "rank_in_industry", "is_top_k"]]
        .rename(
            columns={
                "industry": COL_INDUSTRY,
                "industry_label": COL_INDUSTRY_LABEL,
                "rank_in_industry": COL_RANK,
                "is_top_k": COL_TOP5,
            }
        )
    )
    meta[COL_TOP5] = meta[COL_TOP5].map({True: "Y", False: ""})
    users = users.drop(columns=[COL_INDUSTRY, COL_INDUSTRY_LABEL, COL_RANK, COL_TOP5], errors="ignore")
    users = users.merge(
        meta.reset_index(),
        left_on="_username",
        right_on="username",
        how="left",
    ).drop(columns=["username"], errors="ignore")
    for col in (COL_INDUSTRY, COL_INDUSTRY_LABEL):
        users[col] = users[col].fillna("unclassified" if col == COL_INDUSTRY else "未分类")
    users[COL_RANK] = users[COL_RANK].fillna(0).astype(int)
    users[COL_TOP5] = users[COL_TOP5].fillna("")
    out = users.drop(columns=["_username"], errors="ignore")
    out.to_csv(USERS_CSV, index=False, encoding="utf-8-sig")


def export_rankings(ranked: pd.DataFrame, top: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(KOL_RANKED_CSV, index=False, encoding="utf-8-sig")
    top.to_csv(TOP_KOL_CSV, index=False, encoding="utf-8-sig")


def print_ranking_summary(ranked: pd.DataFrame, top: pd.DataFrame, top_k: int) -> None:
    print(f"\n=== 行业分类 (119 KOL → 每行业 Top-{top_k}) ===")
    for ind, g in ranked.groupby("industry", sort=False):
        label = INDUSTRY_LABELS.get(ind, ind)
        print(f"\n  [{label}] {len(g)} 人")
        for r in g[g["rank_in_industry"] <= top_k].itertuples():
            print(f"    #{r.rank_in_industry} @{r.username}  阅读数={r.views_seed:,}")
    print(f"\n  Top 面板: {len(top)} 人 -> {TOP_KOL_CSV}")
    print(f"  全量排名: -> {KOL_RANKED_CSV}")


def run_classify(top_k: int = 5, write_users: bool = True) -> list[str]:
    users = read_users()
    ranked, top = rank_kols_by_industry(users, top_k=top_k)
    export_rankings(ranked, top)
    if write_users:
        write_industry_to_users(ranked)
        print(
            f"  已写回 users.csv 列: {COL_INDUSTRY}, {COL_INDUSTRY_LABEL}, "
            f"{COL_RANK}, {COL_TOP5}"
        )
    print_ranking_summary(ranked, top, top_k)
    return top["username"].tolist()


def init_columns() -> None:
    df = read_users()
    changed = False
    for col in (COL_TIME, COL_UPDATED):
        if col not in df.columns:
            df[col] = ""
            changed = True
        df[col] = df[col].astype("string")
    if changed:
        out = df.drop(columns=["_username"], errors="ignore")
        out.to_csv(USERS_CSV, index=False, encoding="utf-8-sig")
        print(f"已添加列: {COL_TIME}, {COL_UPDATED} -> {USERS_CSV}")
    else:
        print("users.csv 已有时间列")


def sort_history_by_industry(history_path: Path | None = None) -> Path:
    """为 history 增加行业/排名列，并按 行业→排名→用户名→推文时间(降序) 排序。"""
    path = history_path or HISTORY_CSV
    if not path.exists():
        raise SystemExit(f"缺少 {path}")
    from data_cleaning import repair_tweet_times

    hist = repair_tweet_times(pd.read_csv(path))
    if KOL_RANKED_CSV.exists():
        meta = pd.read_csv(KOL_RANKED_CSV)
    else:
        users = read_users()
        meta, _ = rank_kols_by_industry(users, top_k=5)
    meta = meta.set_index("username")
    hist["industry"] = hist["username"].map(meta["industry"])
    hist["industry_label"] = hist["username"].map(meta["industry_label"])
    hist["kol_rank_in_industry"] = hist["username"].map(meta["rank_in_industry"])
    hist["tweet_time"] = pd.to_datetime(hist["tweet_time"], utc=True, errors="coerce")
    hist = hist.sort_values(
        ["industry", "kol_rank_in_industry", "username", "tweet_time"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    hist.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已排序 history -> {path} ({len(hist)} 行)")
    return path


def _is_rate_limit(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate limit" in s


def _is_recursion_bug(exc: BaseException) -> bool:
    return "recursion" in str(exc).lower() or isinstance(exc, RecursionError)


async def _sleep_countdown(seconds: float, label: str) -> None:
    """等待时每 10 秒打印一次，避免误以为卡死。"""
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        step = min(10.0, remaining)
        await asyncio.sleep(step)
        remaining -= step
        if remaining > 0:
            print(f"  [{label}] 等待 {remaining:.0f}s ...")


async def _call_with_retry(
    label: str,
    factory: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_wait: float = 45.0,
):
    """429 / 递归解析错误时短退避重试。"""
    last: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return await factory()
        except Exception as exc:
            last = exc
            transient = _is_rate_limit(exc) or _is_recursion_bug(exc)
            if transient and attempt == max_retries - 1:
                raise
            if not transient:
                raise
            wait = base_wait * (1.4**attempt)
            kind = "限流 429" if _is_rate_limit(exc) else "解析异常(多为限流)"
            print(
                f"  [{label}] {kind}，{wait:.0f}s 后重试 "
                f"({attempt + 1}/{max_retries})"
            )
            await _sleep_countdown(wait, label)
    if last:
        raise last
    raise RuntimeError("retry exhausted")


def _rows_from_timeline(
    page,
    uname: str,
    since_ts: pd.Timestamp,
    ingested: str,
    cap: int,
) -> tuple[list[dict], bool]:
    """从一页时间线提取推文。返回 (rows, stop_paging)。"""
    rows: list[dict] = []
    stop = False
    for tw in page:
        if len(rows) >= cap:
            return rows, stop
        text = _clean_text(
            getattr(tw, "full_text", None)
            or getattr(tw, "text", None)
            or ""
        )
        created = getattr(tw, "created_at", None) or getattr(tw, "date", None)
        if created is None:
            continue
        tt = pd.Timestamp(created)
        if tt.tzinfo is None:
            tt = tt.tz_localize("UTC")
        else:
            tt = tt.tz_convert("UTC")
        if tt < since_ts:
            return rows, True
        rows.append(
            {
                "tweet_id": str(
                    getattr(tw, "id", None) or f"tw_{uname}_{len(rows)}"
                ),
                "username": uname,
                "tweet_time": tt.isoformat(),
                "text": text,
                "likes": getattr(tw, "favorite_count", None)
                or getattr(tw, "like_count", None),
                "retweets": getattr(tw, "retweet_count", None),
                "replies": getattr(tw, "reply_count", None),
                "views": getattr(tw, "view_count", None),
                "source": "twikit",
                "ingested_at": ingested,
            }
        )
    return rows, stop


def repair_history_file() -> pd.DataFrame:
    """用 Snowflake / ingested_at 修复 history 中缺失的 tweet_time。"""
    from data_cleaning import repair_tweet_times

    if not HISTORY_CSV.exists():
        raise SystemExit(f"缺少 {HISTORY_CSV}")
    df = repair_tweet_times(pd.read_csv(HISTORY_CSV))
    df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    ok = df["tweet_time"].notna().sum()
    print(f"history 时间修复: {ok}/{len(df)} 条有 tweet_time -> {HISTORY_CSV}")
    return df


def append_history_rows(rows: list[dict], replace: bool = False) -> int:
    """每抓完一个 KOL 就落盘，避免中断丢数据。"""
    from data_cleaning import repair_tweet_times

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if new_df.empty and not replace:
        return len(pd.read_csv(HISTORY_CSV)) if HISTORY_CSV.exists() else 0
    if replace and not HISTORY_CSV.exists() and new_df.empty:
        pd.DataFrame(columns=HISTORY_COLS).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
        return 0
    if replace and new_df.empty and HISTORY_CSV.exists():
        HISTORY_CSV.unlink()
        return 0
    if replace and not HISTORY_CSV.exists():
        out = new_df
    elif HISTORY_CSV.exists() and not new_df.empty:
        old = pd.read_csv(HISTORY_CSV)
        out = pd.concat([old, new_df], ignore_index=True)
    elif not new_df.empty:
        out = new_df
    else:
        out = repair_tweet_times(pd.read_csv(HISTORY_CSV))
        out.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
        return len(out)
    out = repair_tweet_times(out)
    out = out.drop_duplicates("tweet_id", keep="last")
    out = out.sort_values(["username", "tweet_time"])
    out.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    return len(out)


def _read_failed_users() -> set[str]:
    if not FAILED_TXT.exists():
        return set()
    return {
        ln.strip().lower()
        for ln in FAILED_TXT.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }


def _write_failed_users(failed: set[str]) -> None:
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_TXT.write_text(
        "\n".join(sorted(failed)) + ("\n" if failed else ""),
        encoding="utf-8",
    )


def _users_in_history(days: int) -> set[str]:
    if not HISTORY_CSV.exists():
        return set()
    from data_cleaning import repair_tweet_times

    df = repair_tweet_times(pd.read_csv(HISTORY_CSV))
    if df.empty or "username" not in df.columns:
        return set()
    df["tweet_time"] = pd.to_datetime(df["tweet_time"], utc=True, errors="coerce")
    since = pd.Timestamp.now("UTC") - pd.Timedelta(days=days)
    recent = df[df["tweet_time"] >= since]
    return {str(u).lower() for u in recent["username"].dropna().unique()}


async def _fetch_one_user_once(
    client,
    uname: str,
    *,
    since_ts: pd.Timestamp,
    max_per_user: int,
    sleep_page: float,
    ingested: str,
    tweet_page_size: int = 20,
    max_pages: int = 1,
) -> tuple[list[dict], str | None]:
    """单次抓取。默认只抓第一页(不翻页)，翻页最易触发解析异常。"""
    rows: list[dict] = []

    async def _get_user():
        return await client.get_user_by_screen_name(uname)

    try:
        user = await _call_with_retry(uname, _get_user, max_retries=3, base_wait=45.0)
    except Exception as exc:
        if _is_rate_limit(exc):
            return rows, "rate_limit"
        if _is_recursion_bug(exc):
            return rows, "recursion_limit"
        return rows, str(exc)[:200]

    page = None
    last_exc: Exception | None = None
    for size in (tweet_page_size, min(15, tweet_page_size), 10):
        try:
            page = await _call_with_retry(
                f"{uname}/tweets",
                lambda s=size, u=user: u.get_tweets("Tweets", count=s),
                max_retries=2,
                base_wait=40.0,
            )
            break
        except Exception as exc:
            last_exc = exc
            await _sleep_countdown(30, f"{uname}/tweets")

    if page is None:
        exc = last_exc or RuntimeError("get_tweets failed")
        if _is_rate_limit(exc):
            return rows, "rate_limit"
        if _is_recursion_bug(exc):
            return rows, "recursion_limit"
        return rows, str(exc)[:200]

    try:
        pages_done = 0
        while page is not None and len(rows) < max_per_user and pages_done < max_pages:
            batch, stop = _rows_from_timeline(page, uname, since_ts, ingested, max_per_user)
            rows.extend(batch)
            if stop or len(rows) >= max_per_user:
                break
            pages_done += 1
            if pages_done >= max_pages:
                break
            try:
                await asyncio.sleep(sleep_page)
                page = await _call_with_retry(
                    f"{uname}/page",
                    lambda: page.next(),
                    max_retries=2,
                    base_wait=40.0,
                )
            except Exception as exc:
                if rows:
                    print(
                        f"  @{uname}: 翻页失败，保留已抓 {len(rows)} 条 "
                        f"({type(exc).__name__})"
                    )
                    return rows, None
                if _is_rate_limit(exc):
                    return rows, "rate_limit"
                if _is_recursion_bug(exc):
                    return rows, "recursion_limit"
                return rows, str(exc)[:200]
    except RecursionError:
        return (rows, None) if rows else (rows, "recursion_limit")
    except Exception as exc:
        if rows:
            return rows, None
        if _is_rate_limit(exc):
            return rows, "rate_limit"
        if _is_recursion_bug(exc):
            return rows, "recursion_limit"
        return rows, str(exc)[:200]

    return rows, None


async def _fetch_one_user(
    client,
    uname: str,
    *,
    since_ts: pd.Timestamp,
    max_per_user: int,
    sleep_page: float,
    ingested: str,
    max_pages: int = 1,
    user_retries: int = 2,
) -> tuple[list[dict], str | None]:
    """整用户级重试（限流/递归时短睡再试）。"""
    last_err: str | None = None
    for attempt in range(user_retries):
        rows, err = await _fetch_one_user_once(
            client,
            uname,
            since_ts=since_ts,
            max_per_user=max_per_user,
            sleep_page=sleep_page,
            ingested=ingested,
            max_pages=max_pages,
        )
        if err in (None, "rate_limit", "recursion_limit"):
            return rows, err
        last_err = err
        if attempt < user_retries - 1:
            wait = 8.0 * (attempt + 1)
            print(f"  @{uname} {err}，{wait:.0f}s 后整用户重试 ({attempt + 1}/{user_retries})")
            await asyncio.sleep(wait)
    return [], last_err


async def _open_twikit(client) -> None:
    path = _cookies_path()
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    if _cookies_ready(path):
        try:
            await client.load_cookies(str(path))
        except TypeError:
            client.load_cookies(str(path))
        print(f"已加载 cookies: {path}")
        return

    user = os.environ.get("X_USERNAME") or os.environ.get("TWITTER_USERNAME")
    email = os.environ.get("X_EMAIL") or os.environ.get("TWITTER_EMAIL")
    password = os.environ.get("X_PASSWORD") or os.environ.get("TWITTER_PASSWORD")
    if not all([user, email, password]):
        raise SystemExit(
            f"缺少 cookies 文件: {path}\n"
            + _cloudflare_hint()
        )
    print("未找到 cookies，尝试 API 登录（易被 Cloudflare 拦）...")
    print(f"  建议: 浏览器登录 x.com 后导出 cookies 到\n  {path}")
    try:
        await client.login(
            auth_info_1=user.strip(),
            auth_info_2=email.strip(),
            password=password,
            cookies_file=str(path),
        )
        print(f"已保存 cookies -> {path}")
    except Exception as exc:
        err = str(exc)
        if any(x in err for x in ("403", "Cloudflare")) or "blocked" in err.lower():
            raise SystemExit(_cloudflare_hint()) from exc
        raise


async def _fetch_async(
    usernames: list[str],
    days: int,
    max_per_user: int,
    *,
    sleep_user: float = 5.0,
    sleep_page: float = 1.5,
    resume: bool = False,
    replace: bool = False,
    stop_on_rate_limit: bool = True,
    retry_failed: bool = False,
    skip_users: list[str] | None = None,
    max_pages: int = 1,
    tweet_page_size: int = 20,
    refetch_users: bool = False,
) -> pd.DataFrame:
    _patch_twikit_keybyte()
    _patch_twikit_user()
    from twikit import Client

    client = Client("en-US")
    await _open_twikit(client)
    print(
        f"  模式: 每页 {tweet_page_size} 条, 最多翻页 {max_pages} 次 "
        f"(翻页易限流; 要更多请加 --max-pages 2)"
    )

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ts = pd.Timestamp(since)
    ingested = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    skip_set = {u.lower() for u in (skip_users or [])}
    todo = [u for u in usernames if u.lower() not in skip_set]

    done: set[str] = set()
    if resume and not replace:
        if refetch_users:
            print("  --refetch-users: 已抓取的 KOL 也会再次请求（按 tweet_id 去重合并）")
        else:
            repair_history_file()
            done = _users_in_history(days)
        skip_failed = set() if retry_failed else _read_failed_users()
        if retry_failed and skip_failed:
            print(f"  --retry-failed: 将重试上次失败 {len(skip_failed)} 人")
        todo = [u for u in todo if u.lower() not in done and u.lower() not in skip_failed]
        print(
            f"  断点续跑: history 窗口内已有 {len(done)} 人, 待抓 {len(todo)}"
            + (
                f" (跳过失败名单 {len(skip_failed)} 人, 见 {FAILED_TXT})"
                if skip_failed and not retry_failed
                else ""
            )
        )

    if replace:
        append_history_rows([], replace=True)

    all_rows: list[dict] = []
    failed: set[str] = set()
    total = len(todo)

    for i, uname in enumerate(todo, 1):
        print(f"\n  [{i}/{total}] @{uname}")
        rows, err = await _fetch_one_user(
            client,
            uname,
            since_ts=since_ts,
            max_per_user=max_per_user,
            sleep_page=sleep_page,
            ingested=ingested,
            max_pages=max_pages,
        )
        if rows:
            append_history_rows(rows, replace=False)
            n_hist = len(pd.read_csv(HISTORY_CSV)) if HISTORY_CSV.exists() else 0
            print(f"  @{uname}: {len(rows)} 条 (history 共 {n_hist} 行)")
            all_rows.extend(rows)
        if err:
            print(f"  失败: @{uname}: {err}")
            failed.add(uname.lower())
            if err == "rate_limit" and stop_on_rate_limit:
                print(
                    f"\n  触发全局限流，暂停队列。请等待 15~30 分钟后:\n"
                    f"    python3 data_mining.py --fetch-twikit --days {days} "
                    f"--resume --sleep-user {sleep_user:.0f}\n"
                    f"  或先抓别人(跳过失败名单): 同上命令不加 --retry-failed\n"
                )
                break
            if err == "recursion_limit":
                print(
                    f"  已记入 {FAILED_TXT}；--resume 默认跳过。"
                    f" 等 15 分钟后: --retry-failed --skip-user 其他人 或单独重试该账号"
                )
        elif not rows:
            print(f"  @{uname}: 0 条 (近 {days} 天无帖或账号受限)")

        print(f"  进度 {i}/{total}, 本轮新增 {len(all_rows)} 条")
        if i < total:
            await asyncio.sleep(sleep_user)

    if failed:
        _write_failed_users(failed)
        print(f"  失败/待重试 {len(failed)} 人 -> {FAILED_TXT}")
        print(
            "  继续抓其余人: python3 data_mining.py --fetch-twikit "
            "--days 30 --resume --sleep-user 10"
        )
        print(
            "  只重试失败: python3 data_mining.py --fetch-twikit "
            "--days 30 --retry-failed --sleep-user 12"
        )

    if not all_rows and not HISTORY_CSV.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    if HISTORY_CSV.exists():
        return pd.read_csv(HISTORY_CSV)
    return pd.DataFrame(all_rows)


def fetch_posts(
    usernames: list[str],
    days: int = 30,
    max_per_user: int = 120,
    **kwargs,
) -> pd.DataFrame:
    try:
        import twikit  # noqa: F401
    except ImportError:
        raise SystemExit("请先: pip install -r requirements.txt")
    return asyncio.run(_fetch_async(usernames, days, max_per_user, **kwargs))


def save_history(df: pd.DataFrame, replace: bool = False) -> Path:
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not replace and HISTORY_CSV.exists() and not df.empty:
        old = pd.read_csv(HISTORY_CSV)
        df = pd.concat([old, df], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates("tweet_id", keep="last")
        df = df.sort_values(["username", "tweet_time"])
    df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    print(f"history: {len(df)} 行 -> {HISTORY_CSV}")
    return HISTORY_CSV


def score_sentiment(text: str) -> tuple[str, float]:
    t = (text or "").lower()
    p = sum(1 for w in _POS if w in t)
    n = sum(1 for w in _NEG if w in t)
    if p > n:
        return "positive", min(1.0, 0.2 + 0.2 * (p - n))
    if n > p:
        return "negative", max(-1.0, -0.2 - 0.2 * (n - p))
    return "neutral", 0.0


def enrich_users(df: pd.DataFrame, days: int = 30) -> int:
    init_columns()
    users = read_users()
    tw = df.copy()
    tw["tweet_time"] = pd.to_datetime(tw["tweet_time"], utc=True, errors="coerce")
    tw = tw.dropna(subset=["tweet_time", "username"])
    since = pd.Timestamp.now("UTC") - pd.Timedelta(days=days)
    tw = tw[tw["tweet_time"] >= since]
    if tw.empty:
        print(f"近 {days} 天无推文，users.csv 未改")
        return 0
    latest = tw.sort_values("tweet_time").groupby("username", as_index=False).last()
    now_str = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for col in (COL_TIME, COL_UPDATED):
        if col in users.columns:
            users[col] = users[col].astype("string")
    n = 0
    for row in latest.itertuples():
        m = users["_username"] == row.username
        if not m.any():
            continue
        users.loc[m, COL_TEXT] = _clean_text(str(row.text))[:8000]
        users.loc[m, COL_TIME] = pd.Timestamp(row.tweet_time).isoformat()
        users.loc[m, COL_UPDATED] = now_str
        n += int(m.sum())
    users.drop(columns=["_username"], inplace=True, errors="ignore")
    users.to_csv(USERS_CSV, index=False, encoding="utf-8-sig")
    print(f"users.csv 已更新 {n} 人 (近 {days} 天推文 {len(tw)} 条)")
    return n


def _price_returns(ticker: str, event: pd.Timestamp) -> dict[str, float | None]:
    import yfinance as yf

    start = (event - pd.Timedelta(days=25)).strftime("%Y-%m-%d")
    end = (event + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        px = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
    except Exception:
        return {}
    if px is None or px.empty:
        return {}
    if isinstance(px.columns, pd.MultiIndex):
        if px.columns.nlevels > 1:
            px = px.droplevel(1, axis=1)
    close = px["Close"] if "Close" in px.columns else px.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return {}
    idx = close.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    event_naive = event.tz_convert(None) if event.tzinfo else event
    pos = min(max(idx.searchsorted(event_naive), 0), len(close) - 1)
    c0 = float(close.iloc[pos])
    out: dict[str, float | None] = {"close_at_event": c0}
    for h in FORWARD_HORIZONS:
        j = min(pos + h, len(close) - 1)
        out[f"ret_{h}d"] = (
            float(close.iloc[j] / c0 - 1.0) if c0 else None
        )
    return out


def build_master(top_k_only: bool = True) -> pd.DataFrame:
    """history + 行业元数据 → kol_master (帖×标的×收益)。"""
    if not HISTORY_CSV.exists():
        raise SystemExit(f"缺少 {HISTORY_CSV}，请先 --fetch-twikit")
    hist = pd.read_csv(HISTORY_CSV)
    hist["tweet_time"] = pd.to_datetime(hist["tweet_time"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["tweet_time", "username"])

    if KOL_RANKED_CSV.exists():
        meta = pd.read_csv(KOL_RANKED_CSV)
    else:
        users = read_users()
        meta, _ = rank_kols_by_industry(users, top_k=5)
    if top_k_only and "is_top_k" in meta.columns:
        meta = meta[meta["is_top_k"]]
    meta = meta.set_index("username")

    rows: list[dict] = []
    for tw in hist.itertuples():
        uname = str(tw.username)
        if uname not in meta.index:
            continue
        m = meta.loc[uname]
        info = classify_text(str(tw.text))
        tickers = info["tickers"] or (
            [INDUSTRY_ETF_MAP.get(m["industry"])]
            if INDUSTRY_ETF_MAP.get(m["industry"])
            else []
        )
        tickers = [t for t in tickers if t]
        if not tickers:
            continue
        sent, score = score_sentiment(str(tw.text))
        event = pd.Timestamp(tw.tweet_time)
        for ticker in tickers:
            rets = _price_returns(ticker, event)
            rows.append(
                {
                    "industry": m["industry"],
                    "industry_label": m["industry_label"],
                    "kol_rank_in_industry": int(m["rank_in_industry"]),
                    "influence_score": float(m.get("influence_score", 0)),
                    "username": uname,
                    "nickname": m.get("nickname", ""),
                    "tweet_id": tw.tweet_id,
                    "tweet_time_utc": pd.Timestamp(event).isoformat(),
                    "text": str(tw.text)[:500],
                    "sentiment": sent,
                    "sentiment_score": score,
                    "ticker": ticker,
                    **rets,
                }
            )

    if not rows:
        raise SystemExit("kol_master 无行：history 为空或与 Top-K 无交集")

    master = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")
    n_tw = master["tweet_id"].nunique()
    print("\n=== kol_master.csv ===")
    print(f"  行: {len(master)} | 推文: {n_tw} | KOL: {master['username'].nunique()}")
    print(f"  路径: {MASTER_CSV}")
    return master


def _pick_sentiment_col(df: pd.DataFrame) -> str:
    if "sentiment_score_nlp" in df.columns and df["sentiment_score_nlp"].std(skipna=True) > 0:
        return "sentiment_score_nlp"
    return "sentiment_score"


def _safe_ic(sub: pd.DataFrame, x_col: str, y_col: str, min_n: int = 8) -> float | None:
    if x_col not in sub.columns or y_col not in sub.columns:
        return None
    s = sub[[x_col, y_col]].dropna()
    if len(s) < min_n:
        return None
    if s[x_col].std(skipna=True) == 0 or s[y_col].std(skipna=True) == 0:
        return None
    v = s[x_col].corr(s[y_col])
    return float(v) if pd.notna(v) else None


def build_master_from_clean(path: Path | None = None) -> pd.DataFrame:
    """
    从 Step2 清洗结果构建 kol_master，并保留行业代表股收益 ind_ret_*。
    需先运行: python3 data_cleaning.py --write-master
    """
    src = path or EVENTS_CLEAN_CSV
    if not src.exists():
        raise SystemExit(f"缺少 {src}，请先运行 python3 data_cleaning.py")

    ev = pd.read_csv(src)
    ranked = pd.read_csv(KOL_RANKED_CSV) if KOL_RANKED_CSV.exists() else None
    nick_map, infl_map = {}, {}
    if ranked is not None:
        nick_map = dict(zip(ranked["username"], ranked.get("nickname", ranked["username"])))
        infl_map = dict(zip(ranked["username"], ranked.get("influence_score", 0)))

    rows = []
    for r in ev.itertuples():
        rows.append(
            {
                "industry": getattr(r, "industry", None),
                "industry_label": getattr(r, "industry_label", None),
                "kol_rank_in_industry": getattr(r, "kol_rank_in_industry", None),
                "influence_score": infl_map.get(r.username, 0),
                "username": r.username,
                "nickname": nick_map.get(r.username, r.username),
                "tweet_id": r.tweet_id,
                "tweet_time_utc": getattr(
                    r, "tweet_time", None
                ) or getattr(r, "tweet_time_utc", None),
                "text": getattr(r, "text_clean", getattr(r, "text", "")),
                "sentiment": r.sentiment,
                "sentiment_score": r.sentiment_score,
                "sentiment_nlp": getattr(r, "sentiment_nlp", None),
                "sentiment_score_nlp": getattr(r, "sentiment_score_nlp", None),
                "ticker": r.ticker,
                "ticker_src": getattr(r, "ticker_src", None),
                "industry_ticker": getattr(r, "industry_ticker", None),
                "close_at_event": getattr(r, "close_at_event", None),
                "ret_5m": getattr(r, "ret_5m", None),
                "ret_1h": getattr(r, "ret_1h", None),
                "ret_1d": getattr(r, "ret_1d", None),
                "ret_5d": getattr(r, "ret_5d", None),
                "ret_20d": getattr(r, "ret_20d", None),
                "ind_close_at_event": getattr(r, "ind_close_at_event", None),
                "ind_ret_5m": getattr(r, "ind_ret_5m", None),
                "ind_ret_1h": getattr(r, "ind_ret_1h", None),
                "ind_ret_1d": getattr(r, "ind_ret_1d", None),
                "ind_ret_5d": getattr(r, "ind_ret_5d", None),
                "ind_ret_20d": getattr(r, "ind_ret_20d", None),
            }
        )
    if not rows:
        raise SystemExit("tweet_events_clean 无行")

    master = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")
    n_tw = master["tweet_id"].nunique()
    print("\n=== kol_master.csv (from clean) ===")
    print(f"  行: {len(master)} | 推文: {n_tw} | KOL: {master['username'].nunique()}")
    print(f"  路径: {MASTER_CSV}")
    return master


def build_daily_factor(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["event_date"] = pd.to_datetime(m["tweet_time_utc"], utc=True, errors="coerce").dt.date
    m["w"] = m.groupby("username")["influence_score"].transform("max").fillna(1)
    sent_col = _pick_sentiment_col(m)
    m["weighted_sent"] = m[sent_col] * m["w"]
    agg: dict = {
        "factor_sentiment": ("weighted_sent", "sum"),
        "weight_sum": ("w", "sum"),
        "n_posts": ("tweet_id", "nunique"),
    }
    if sent_col == "sentiment_score_nlp":
        m["weighted_sent_lex"] = m["sentiment_score"] * m["w"]
        agg["factor_sentiment_lex"] = ("weighted_sent_lex", "sum")
    daily = (
        m.groupby(["event_date", "industry", "industry_label", "ticker"], as_index=False)
        .agg(**agg)
    )
    daily["iwks"] = daily["factor_sentiment"] / daily["weight_sum"].replace(0, pd.NA)
    if "factor_sentiment_lex" in daily.columns:
        daily["iwks_lex"] = daily["factor_sentiment_lex"] / daily["weight_sum"].replace(
            0, pd.NA
        )
    daily.to_csv(DAILY_FACTOR_CSV, index=False, encoding="utf-8-sig")
    print(f"  日频因子 ({sent_col}): {DAILY_FACTOR_CSV}")
    return daily


def run_backtest(master: pd.DataFrame) -> pd.DataFrame:
    sent_col = _pick_sentiment_col(master)
    rows: list[dict] = []
    for (ind, label), sub in master.groupby(["industry", "industry_label"]):
        row: dict = {
            "industry": ind,
            "industry_label": label,
            "n_events": len(sub),
            "sentiment_col": sent_col,
            "etf": INDUSTRY_ETF_MAP.get(ind),
        }
        for ret_col in IC_RETURN_COLS:
            ic = _safe_ic(sub, sent_col, ret_col)
            row[f"ic_{ret_col}"] = round(ic, 4) if ic is not None else None
        rows.append(row)
    bt = pd.DataFrame(rows)
    bt.to_csv(BACKTEST_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== backtest_summary.csv (IC, {sent_col}) ===")
    show = [
        "industry",
        "n_events",
        "ic_ind_ret_1h",
        "ic_ind_ret_1d",
        "ic_ind_ret_5d",
        "ic_ind_ret_20d",
        "ic_ret_1h",
        "ic_ret_1d",
    ]
    show = [c for c in show if c in bt.columns]
    print(bt[show].to_string(index=False))
    return bt


def _run_fetch_flow(args: argparse.Namespace) -> None:
    users = read_users()
    if args.all_kols:
        names = users["_username"].drop_duplicates().tolist()
        print(
            f"users.csv: {len(users)} KOL → 抓取全部 {len(names)} 人, 近 {args.days} 天"
        )
    else:
        names = run_classify(top_k=args.top_k, write_users=True)
        print(
            f"\n抓取目标: 各行业 Top-{args.top_k} 共 {len(names)} 人, 近 {args.days} 天"
        )
    if args.limit_users:
        names = names[: args.limit_users]
        print(f"  (试跑 limit-users={args.limit_users})")

    if args.replace_history and args.resume:
        print("警告: --replace-history 与 --resume 冲突，已忽略 replace")
        args.replace_history = False
    if args.replace_history and HISTORY_CSV.exists():
        HISTORY_CSV.unlink()
        print(f"已删除旧 history: {HISTORY_CSV}")

    init_columns()
    print(
        f"  节流: 每人间隔 {args.sleep_user}s, 翻页间隔 {args.sleep_page}s, "
        f"上限 {args.max_per_user} 条/人, max_pages={args.max_pages}"
    )
    df = fetch_posts(
        names,
        days=args.days,
        max_per_user=args.max_per_user,
        sleep_user=args.sleep_user,
        sleep_page=args.sleep_page,
        resume=args.resume,
        replace=args.replace_history,
        retry_failed=args.retry_failed,
        skip_users=args.skip_user,
        max_pages=args.max_pages,
        tweet_page_size=args.tweet_page_size,
        refetch_users=args.refetch_users,
    )
    if df.empty:
        print("未抓到推文。请检查 cookies / 网络后重试。")
        return
    if not args.resume:
        save_history(df, replace=args.replace_history)
    enrich_users(df, days=args.days)


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(
        description="KOL 数据挖掘 (users.csv → history → kol_master)"
    )
    p.add_argument(
        "--pipeline",
        action="store_true",
        help="分类 + 抓 Top-K + 回写 users + kol_master + IC",
    )
    p.add_argument("--master", action="store_true", help="构建 kol_master + IC")
    p.add_argument(
        "--from-clean",
        action="store_true",
        help="从 data/outputs/tweet_events_clean.csv 构建（Step2/3 之后）",
    )
    p.add_argument(
        "--nlp",
        action="store_true",
        help="先跑 nlp.py 再 --from-clean",
    )
    p.add_argument(
        "--fetch-twikit",
        action="store_true",
        help="抓取推文 (默认与 --pipeline 含)",
    )
    p.add_argument("--init-columns", action="store_true", help="仅给 users.csv 加时间列")
    p.add_argument(
        "--classify-only",
        action="store_true",
        help="仅行业分类+Top-K，写入 users.csv 与 data/outputs/",
    )
    p.add_argument("--top-k", type=int, default=5, help="每行业影响力 Top-K (默认 5)")
    p.add_argument("--all-kols", action="store_true", help="抓取全部 119 人 (默认只抓 Top-K)")
    p.add_argument("--enrich-only", action="store_true", help="仅从 history 回写 users.csv")
    p.add_argument("--sort-history", action="store_true", help="为 history 加行业列并排序")
    p.add_argument("--days", type=int, default=30, help="最近天数 (默认 30=一月)")
    p.add_argument("--max-per-user", type=int, default=40, help="每 KOL 最多条数 (默认 40)")
    p.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="翻页次数 (默认 1=不翻页, 最稳; 2 约 40 条)",
    )
    p.add_argument("--tweet-page-size", type=int, default=20, help="每页请求条数 (默认 20)")
    p.add_argument("--limit-users", type=int, default=None, help="试跑前 N 人")
    p.add_argument(
        "--replace-history",
        action="store_true",
        help="清空后重抓 (勿与 --resume 同用)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="跳过 history 已有数据 + 默认跳过上次失败用户",
    )
    p.add_argument(
        "--refetch-users",
        action="store_true",
        help="与 --resume 同用：不跳过已抓取 KOL，加深翻页并 merge 去重",
    )
    p.add_argument(
        "--repair-history",
        action="store_true",
        help="仅用 Snowflake 修复 history_tweets.csv 的 tweet_time",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="--resume 时仍重试 fetch_failed_users.txt 中的账号",
    )
    p.add_argument(
        "--skip-user",
        action="append",
        default=[],
        metavar="USER",
        help="跳过指定 username，可多次指定",
    )
    p.add_argument(
        "--sleep-user",
        type=float,
        default=5.0,
        help="每个 KOL 之间暂停秒数 (防 429, 默认 5)",
    )
    p.add_argument(
        "--sleep-page",
        type=float,
        default=1.5,
        help="翻页间隔秒数 (默认 1.5)",
    )
    args = p.parse_args()

    if args.init_columns:
        init_columns()
        return

    if args.repair_history:
        repair_history_file()
        return

    if args.enrich_only:
        if not HISTORY_CSV.exists():
            raise SystemExit(f"缺少 {HISTORY_CSV}，请先抓取")
        hist = pd.read_csv(HISTORY_CSV)
        enrich_users(hist, days=args.days)
        return

    if args.sort_history:
        sort_history_by_industry()
        return

    if args.classify_only:
        run_classify(top_k=args.top_k)
        return

    if args.pipeline:
        args.fetch_twikit = True
        args.master = True
        # 全量 KOL：python3 data_mining.py --pipeline --all-kols

    if args.fetch_twikit:
        print(f"附件 users.csv: {len(read_users())} KOL")
        _run_fetch_flow(args)
        if not args.master:
            print("\n完成。可加 --master 构建 kol_master。")
            return

    if args.master:
        if args.nlp:
            from nlp import run_nlp

            run_nlp(use_transformers=False)
        if args.from_clean:
            master = build_master_from_clean()
        else:
            master = build_master(top_k_only=not args.all_kols)
        build_daily_factor(master)
        run_backtest(master)
        return

    p.print_help()


if __name__ == "__main__":
    main()
