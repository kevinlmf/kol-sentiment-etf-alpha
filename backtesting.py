#!/usr/bin/env python3
"""
策略回测 — 日频多空 + lag 信号 + 净值/夏普/回撤 + 样本内外

用法:
  python3 backtesting.py --all --signal sig_contrarian --mode lag1
  python3 backtesting.py --all --per-industry --mode lag1
  python3 backtesting.py --compare-signals
  python3 backtesting.py --compare-benchmarks --all --mode lag1
  python3 backtesting.py --backtest-multi-horizon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_mining import INDUSTRY_ETF_MAP, OUTPUTS_DIR
from signal_testing import (
    MULTI_HORIZON_MATRIX_CSV,
    _safe_ic,
    aggregate_daily_signals,
    enrich_daily_signals,
    load_events,
    BEST_PER_INDUSTRY_CSV,
    SIGNAL_DEFS,
    apply_costs,
    backtest_by_industry,
    build_combined_portfolio,
    build_daily_panel,
    build_daily_panel_per_industry,
    build_ml_daily_panel,
    build_signal_variants,
    equity_curve,
    finalize_panel,
    performance_stats,
    performance_stats_combined,
    time_split,
)

EQUITY_CSV = OUTPUTS_DIR / "strategy_equity_curve.csv"
TRADES_CSV = OUTPUTS_DIR / "strategy_daily_trades.csv"
REPORT_CSV = OUTPUTS_DIR / "strategy_backtest_report.csv"
REPORT_JSON = OUTPUTS_DIR / "strategy_backtest_report.json"

TRADES_ALL_CSV = OUTPUTS_DIR / "strategy_daily_trades_all.csv"
EQUITY_COMBINED_CSV = OUTPUTS_DIR / "strategy_equity_combined.csv"
REPORT_ALL_CSV = OUTPUTS_DIR / "strategy_backtest_report_all.csv"
REPORT_BY_INDUSTRY_CSV = OUTPUTS_DIR / "strategy_backtest_by_industry.csv"

TRADING_DAYS_PER_YEAR = 252

BENCHMARK_CSV = OUTPUTS_DIR / "strategy_benchmark_comparison.csv"
MULTI_HORIZON_BACKTEST_CSV = OUTPUTS_DIR / "signal_matrix_backtest.csv"


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


def run_strategy(
    *,
    signal_col: str = "sig_contrarian",
    ret_col: str = "ind_ret_1d",
    industry: str | None = "ai_tech",
    max_rank: int | None = None,
    cost_bps: float = 5.0,
    train_ratio: float = 0.7,
    mode: str = "lag1",
    save_outputs: bool = True,
    universe_all: bool = False,
    per_industry: bool = False,
) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_events()
    tweets = build_signal_variants(raw)
    tweets = filter_tweets(tweets, industry=industry, max_rank=max_rank)

    if per_industry:
        if not BEST_PER_INDUSTRY_CSV.exists():
            raise SystemExit(
                f"缺少 {BEST_PER_INDUSTRY_CSV}，请先运行:\n"
                "  python3 signal_testing.py --scan --lag 1 --ret-col ind_ret_1d"
            )
        mapping = pd.read_csv(BEST_PER_INDUSTRY_CSV)
        if "lag_days" in mapping.columns:
            mapping = mapping[mapping["lag_days"] == (1 if mode == "lag1" else 0)]
        panel = build_daily_panel_per_industry(tweets, mapping, ret_col=ret_col)
        if panel.empty:
            raise SystemExit("分行业信号面板为空，请检查 mapping 与数据行业是否一致")
    elif signal_col in ("sig_ml_up", "ml_up"):
        panel = build_ml_daily_panel(
            tweets,
            ret_col=ret_col,
            train_ratio=train_ratio,
        )
    else:
        panel = build_daily_panel(tweets, signal_col, ret_col=ret_col)

    if mode == "lag1":
        pnl_col, pos_col = "pnl_lag1", "position_lag1"
    else:
        pnl_col, pos_col = "pnl_contemp", "position_contemp"

    panel = finalize_panel(panel, pnl_col, pos_col, cost_bps, mode, signal_col)
    combined = build_combined_portfolio(panel)

    train, test = time_split(panel, train_ratio)
    for part in (train, test):
        part.attrs = dict(panel.attrs)
    stats_full = performance_stats(panel, pnl_col, pos_col, split="full", net_pnl=True)
    stats_train = performance_stats(train, pnl_col, pos_col, split="train", net_pnl=True)
    stats_test = performance_stats(test, pnl_col, pos_col, split="test", net_pnl=True)

    comb_train, comb_test = time_split(combined, train_ratio)
    comb_stats_full = performance_stats_combined(combined, split="combined_full")
    comb_stats_test = performance_stats_combined(comb_test, split="combined_test")

    for st in (stats_full, stats_train, stats_test, comb_stats_full, comb_stats_test):
        st["cost_bps"] = cost_bps
        st["signal"] = signal_col
        st["industry"] = industry if industry not in (None, "all", "*") else "all"
        st["max_rank"] = max_rank
        st["ret_col"] = ret_col
        st["mode"] = mode

    by_ind = backtest_by_industry(panel, pnl_col, pos_col)

    out_cols = [
        "event_date",
        "industry",
        "industry_label",
        "etf",
        "n_posts",
        "n_kol",
        "signal_raw",
        "signal_lag1",
        "fwd_ret",
        "position_lag1",
        "position_contemp",
        "pnl_lag1",
        "pnl_contemp",
        "pnl_gross",
        "pnl_net",
        "equity_industry",
        "strategy_mode",
        "signal_name",
    ]
    if save_outputs:
        panel[out_cols].to_csv(
            TRADES_ALL_CSV if universe_all else TRADES_CSV,
            index=False,
            encoding="utf-8-sig",
        )
        if not universe_all:
            panel[out_cols].to_csv(EQUITY_CSV, index=False, encoding="utf-8-sig")
            panel[out_cols].to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
        combined.to_csv(EQUITY_COMBINED_CSV, index=False, encoding="utf-8-sig")
        report = pd.DataFrame([stats_full, stats_train, stats_test, comb_stats_full, comb_stats_test])
        report.to_csv(
            REPORT_ALL_CSV if universe_all else REPORT_CSV,
            index=False,
            encoding="utf-8-sig",
        )
        by_ind.to_csv(REPORT_BY_INDUSTRY_CSV, index=False, encoding="utf-8-sig")
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": stats_full,
                    "combined": comb_stats_full,
                    "splits": [stats_full, stats_train, stats_test],
                    "by_industry": by_ind.to_dict(orient="records"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    else:
        report = pd.DataFrame([stats_full, stats_train, stats_test])

    return {
        "panel": panel,
        "combined": combined,
        "by_industry": by_ind,
        "report": report,
        "stats_full": stats_full,
        "stats_test": stats_test,
        "comb_stats_full": comb_stats_full,
        "comb_stats_test": comb_stats_test,
    }


def _backtest_from_panel(
    panel: pd.DataFrame,
    *,
    mode: str,
    cost_bps: float,
    signal_col: str,
    train_ratio: float = 0.7,
) -> dict:
    pnl_col = "pnl_lag1" if mode == "lag1" else "pnl_contemp"
    pos_col = "position_lag1" if mode == "lag1" else "position_contemp"
    panel = finalize_panel(panel, pnl_col, pos_col, cost_bps, mode, signal_col)
    combined = build_combined_portfolio(panel)
    train, test = time_split(panel, train_ratio)
    for part in (train, test):
        part.attrs = dict(panel.attrs)
    stats_full = performance_stats(panel, pnl_col, pos_col, split="full", net_pnl=True)
    stats_test = performance_stats(test, pnl_col, pos_col, split="test", net_pnl=True)
    comb_stats_full = performance_stats_combined(combined, split="combined_full")
    _, comb_test = time_split(combined, train_ratio)
    comb_stats_test = performance_stats_combined(comb_test, split="combined_test")
    return {
        "panel": panel,
        "combined": combined,
        "stats_full": stats_full,
        "stats_test": stats_test,
        "comb_stats_full": comb_stats_full,
        "comb_stats_test": comb_stats_test,
    }


def build_synthetic_panel(
    tweets: pd.DataFrame,
    ret_col: str,
    *,
    kind: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """合成信号面板：buy_hold / flat / random。"""
    post_cols = [c for c in SIGNAL_DEFS if c in tweets.columns]
    daily = enrich_daily_signals(aggregate_daily_signals(tweets, post_cols))
    if ret_col not in daily.columns:
        raise ValueError(f"缺少 {ret_col}")
    daily = daily.sort_values(["industry", "event_date"]).reset_index(drop=True)
    daily["event_date"] = pd.to_datetime(daily["event_date"])
    daily["etf"] = daily["industry"].map(INDUSTRY_ETF_MAP)
    if kind == "buy_hold_long":
        daily["signal_raw"] = 1.0
    elif kind == "always_flat":
        daily["signal_raw"] = 0.0
    elif kind == "random_sign":
        rng = np.random.default_rng(random_seed)
        daily["signal_raw"] = rng.choice([-1.0, 1.0], size=len(daily))
    else:
        raise ValueError(f"未知合成信号: {kind}")
    daily["signal_lag1"] = daily.groupby("industry")["signal_raw"].shift(1)
    daily["fwd_ret"] = daily[ret_col]
    daily["position_lag1"] = np.sign(daily["signal_lag1"]).replace(0, np.nan).fillna(0)
    daily["position_contemp"] = np.sign(daily["signal_raw"]).replace(0, np.nan).fillna(0)
    daily["pnl_lag1"] = daily["position_lag1"] * daily["fwd_ret"]
    daily["pnl_contemp"] = daily["position_contemp"] * daily["fwd_ret"]
    return daily


def _row_from_backtest(
    name: str,
    res: dict,
    *,
    meta: dict,
) -> dict:
    sf = res["stats_full"]
    st = res["stats_test"]
    cf = res["comb_stats_full"]
    ct = res["comb_stats_test"]
    return {
        "strategy": name,
        **meta,
        "leg_days": sf.get("n_days"),
        "leg_total_return": sf.get("total_return"),
        "leg_sharpe": sf.get("sharpe"),
        "leg_max_drawdown": sf.get("max_drawdown"),
        "leg_ic": sf.get("ic_signal_fwd_ret"),
        "comb_days": cf.get("n_days"),
        "comb_total_return": cf.get("total_return"),
        "comb_sharpe": cf.get("sharpe"),
        "comb_max_drawdown": cf.get("max_drawdown"),
        "test_leg_return": st.get("total_return"),
        "test_comb_return": ct.get("total_return"),
        "test_comb_sharpe": ct.get("sharpe"),
    }


def compare_benchmarks(
    *,
    ret_col: str = "ind_ret_1d",
    cost_bps: float = 5.0,
    mode: str = "lag1",
    train_ratio: float = 0.7,
    industry: str | None = None,
) -> pd.DataFrame:
    """统一口径对比：基准 / 主信号 / ML / 分行业最优。"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_events()
    tweets = build_signal_variants(raw)
    tweets = filter_tweets(tweets, industry=industry, max_rank=None)
    universe = industry if industry not in (None, "all", "*") else "all"

    rows: list[dict] = []
    base_meta = {"ret_col": ret_col, "mode": mode, "cost_bps": cost_bps, "universe": universe}

    for kind, label in (
        ("buy_hold_long", "买入持有（有事件日做多）"),
        ("always_flat", "始终空仓"),
        ("random_sign", "随机多空（seed=42）"),
    ):
        try:
            panel = build_synthetic_panel(tweets, ret_col, kind=kind)
            res = _backtest_from_panel(
                panel, mode=mode, cost_bps=cost_bps, signal_col=kind, train_ratio=train_ratio
            )
            rows.append(_row_from_backtest(label, res, meta={**base_meta, "signal": kind}))
        except Exception as exc:
            rows.append({"strategy": label, "signal": kind, "error": str(exc), **base_meta})

    signal_specs = [
        ("sig_baseline", "跟随情绪 baseline"),
        ("sig_contrarian", "反转 contrarian"),
        ("sig_kol_breadth_contrarian", "主信号 breadth×反转"),
        ("sig_ml_up", "ML 二分类 sig_ml_up"),
    ]
    for sig, label in signal_specs:
        try:
            if sig == "sig_ml_up":
                panel = build_ml_daily_panel(tweets, ret_col=ret_col, train_ratio=train_ratio)
            else:
                panel = build_daily_panel(tweets, sig, ret_col=ret_col)
            res = _backtest_from_panel(
                panel, mode=mode, cost_bps=cost_bps, signal_col=sig, train_ratio=train_ratio
            )
            rows.append(_row_from_backtest(label, res, meta={**base_meta, "signal": sig}))
        except Exception as exc:
            rows.append({"strategy": label, "signal": sig, "error": str(exc), **base_meta})

    if BEST_PER_INDUSTRY_CSV.exists() and universe == "all":
        try:
            mapping = pd.read_csv(BEST_PER_INDUSTRY_CSV)
            if "ret_col" in mapping.columns:
                mapping = mapping[mapping["ret_col"] == ret_col]
            if "lag_days" in mapping.columns:
                mapping = mapping[mapping["lag_days"] == (1 if mode == "lag1" else 0)]
            panel = build_daily_panel_per_industry(tweets, mapping, ret_col=ret_col)
            res = _backtest_from_panel(
                panel,
                mode=mode,
                cost_bps=cost_bps,
                signal_col="per_industry_best",
                train_ratio=train_ratio,
            )
            rows.append(
                _row_from_backtest(
                    "分行业 IC 最优（1d 路由）",
                    res,
                    meta={**base_meta, "signal": "per_industry_best"},
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "strategy": "分行业 IC 最优（1d 路由）",
                    "signal": "per_industry_best",
                    "error": str(exc),
                    **base_meta,
                }
            )

    board = pd.DataFrame(rows)
    board.to_csv(BENCHMARK_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== 策略 Benchmark 对比 ({ret_col} | {mode} | {universe}) ===")
    print(f"  -> {BENCHMARK_CSV}\n")
    show_cols = [
        "strategy",
        "comb_total_return",
        "comb_sharpe",
        "comb_max_drawdown",
        "leg_ic",
        "comb_days",
    ]
    if "error" not in board.columns or board["error"].isna().all():
        print(board[show_cols].to_string(index=False))
    else:
        ok = board[board.get("error", pd.Series(dtype=object)).isna()]
        if not ok.empty:
            print(ok[show_cols].to_string(index=False))
    return board


def backtest_multi_horizon_matrix(
    *,
    cost_bps: float = 5.0,
    train_ratio: float = 0.7,
    min_ic_days: int = 5,
) -> pd.DataFrame:
    """按 signal_matrix_frequency_industry.csv 逐格回测（分行业 × 频率最优 signal）。"""
    if not MULTI_HORIZON_MATRIX_CSV.exists():
        raise SystemExit(
            f"缺少 {MULTI_HORIZON_MATRIX_CSV}，请先运行:\n"
            "  python3 signal_testing.py --scan-multi-horizon"
        )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(MULTI_HORIZON_MATRIX_CSV)
    matrix = matrix[matrix["scope"] == "per_industry"].copy()
    matrix = matrix[matrix["n_days"].fillna(0) >= min_ic_days]
    matrix = matrix[~matrix["industry"].isin(("unclassified", "consumer_tech", "semiconductor"))]

    raw = load_events()
    tweets = build_signal_variants(raw)
    rows: list[dict] = []

    print("\n=== 多频率 × 分行业 回测（矩阵逐格）===")
    for _, spec in matrix.iterrows():
        ind = spec["industry"]
        sig = spec["best_signal"]
        ret_col = spec["ret_col"]
        value_col = spec.get("signal_col")
        lag = int(spec.get("lag_days", 1))
        mode = "lag1" if lag > 0 else "contemp"
        label = spec.get("industry_label", ind)
        horizon = spec.get("horizon_label", spec.get("horizon"))
        ic_scan = spec.get("ic")

        sub = filter_tweets(tweets, industry=str(ind), max_rank=None)
        name = f"{label}|{horizon}|{sig}"
        try:
            panel = build_daily_panel(
                sub,
                str(sig),
                ret_col=str(ret_col),
                value_col=str(value_col) if pd.notna(value_col) else None,
            )
            res = _backtest_from_panel(
                panel,
                mode=mode,
                cost_bps=cost_bps,
                signal_col=str(sig),
                train_ratio=train_ratio,
            )
            sf = res["stats_full"]
            cf = res["comb_stats_full"]
            row = {
                "industry": ind,
                "industry_label": label,
                "horizon": spec.get("horizon"),
                "horizon_label": horizon,
                "ret_col": ret_col,
                "lag_days": lag,
                "mode": mode,
                "signal": sig,
                "signal_col": value_col,
                "ic_scan": ic_scan,
                "n_days": sf.get("n_days"),
                "total_return": sf.get("total_return"),
                "sharpe": sf.get("sharpe"),
                "max_drawdown": sf.get("max_drawdown"),
                "ic_backtest": sf.get("ic_signal_fwd_ret"),
                "comb_days": cf.get("n_days"),
                "comb_total_return": cf.get("total_return"),
                "comb_sharpe": cf.get("sharpe"),
            }
            rows.append(row)
            tr = sf.get("total_return")
            tr_s = f"{tr:+.2%}" if tr is not None else "n/a"
            print(
                f"  {label:12} {horizon:4} {sig:28} "
                f"ret={tr_s} sharpe={sf.get('sharpe')} ic={sf.get('ic_signal_fwd_ret')}"
            )
        except Exception as exc:
            rows.append(
                {
                    "industry": ind,
                    "industry_label": label,
                    "horizon": spec.get("horizon"),
                    "horizon_label": horizon,
                    "signal": sig,
                    "error": str(exc),
                }
            )
            print(f"  {label:12} {horizon:4} {sig:28} ERROR: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(MULTI_HORIZON_BACKTEST_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  -> {MULTI_HORIZON_BACKTEST_CSV}")
    return out


def compare_signals(
    *,
    ret_col: str,
    industry: str | None,
    max_rank: int | None,
    cost_bps: float,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sig in SIGNAL_DEFS:
        try:
            res = run_strategy(
                signal_col=sig,
                ret_col=ret_col,
                industry=industry,
                max_rank=max_rank,
                cost_bps=cost_bps,
                mode=mode,
                save_outputs=False,
            )
            for split in ("full", "test"):
                st = res["stats_test"] if split == "test" else res["stats_full"]
                rows.append(
                    {
                        "signal": sig,
                        "description": SIGNAL_DEFS[sig],
                        "split": split,
                        **{k: st.get(k) for k in (
                            "n_days", "total_return", "sharpe", "max_drawdown",
                            "win_rate", "ic_signal_fwd_ret",
                        )},
                    }
                )
        except Exception as exc:
            rows.append({"signal": sig, "split": "error", "error": str(exc)})
    board = pd.DataFrame(rows)
    path = OUTPUTS_DIR / "strategy_signal_comparison.csv"
    board.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n信号策略对比 -> {path}")
    if not board.empty:
        show = board[board["split"] == "test"].sort_values(
            "sharpe", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce")
        )
        print(show.head(12).to_string(index=False))
    return board


def main() -> None:
    p = argparse.ArgumentParser(description="完整策略回测（净值/夏普/回撤）")
    p.add_argument("--signal", default="sig_contrarian", help="帖级信号列名")
    p.add_argument("--ret-col", default="ind_ret_1d")
    p.add_argument(
        "--industry",
        default="ai_tech",
        help="行业代码；填 all 表示全行业+全KOL",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="全行业全KOL完整交易（= --industry all，输出 *_all.csv）",
    )
    p.add_argument("--max-rank", type=int, default=None)
    p.add_argument("--cost-bps", type=float, default=5.0, help="单边换手成本 bp")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--mode", choices=("lag1", "contemp"), default="lag1")
    p.add_argument("--compare-signals", action="store_true")
    p.add_argument(
        "--compare-benchmarks",
        action="store_true",
        help="对比基准/主信号/ML/分行业最优（统一 ret-col + mode）",
    )
    p.add_argument(
        "--backtest-multi-horizon",
        action="store_true",
        help="按 signal_matrix_frequency_industry.csv 逐格回测",
    )
    p.add_argument(
        "--per-industry",
        action="store_true",
        help="各行业使用 signal_best_per_industry.csv 中的最优信号",
    )
    args = p.parse_args()

    universe_all = args.all or args.industry in ("all", "*")
    ind_arg = None if universe_all else (args.industry or None)

    if args.compare_benchmarks:
        compare_benchmarks(
            ret_col=args.ret_col,
            cost_bps=args.cost_bps,
            mode=args.mode,
            train_ratio=args.train_ratio,
            industry=ind_arg,
        )
        return

    if args.backtest_multi_horizon:
        backtest_multi_horizon_matrix(cost_bps=args.cost_bps, train_ratio=args.train_ratio)
        return

    if args.compare_signals:
        compare_signals(
            ret_col=args.ret_col,
            industry=ind_arg,
            max_rank=args.max_rank,
            cost_bps=args.cost_bps,
            mode=args.mode,
        )
        return

    if args.per_industry and not BEST_PER_INDUSTRY_CSV.exists():
        print("未找到分行业最优信号表，先运行 signal_testing --scan …")
        from signal_testing import run_by_industry

        run_by_industry(
            ret_cols=[args.ret_col],
            max_rank=args.max_rank,
            lag=1 if args.mode == "lag1" else 0,
        )

    res = run_strategy(
        signal_col=args.signal,
        ret_col=args.ret_col,
        industry=ind_arg,
        max_rank=args.max_rank,
        cost_bps=args.cost_bps,
        train_ratio=args.train_ratio,
        mode=args.mode,
        universe_all=universe_all,
        per_industry=args.per_industry,
    )
    sf, st = res["stats_full"], res["stats_test"]
    cf, ct = res["comb_stats_full"], res["comb_stats_test"]

    print("\n=== 完整策略回测 ===")
    sig_note = (
        "分行业最优（见 signal_best_per_industry.csv）"
        if args.per_industry
        else f"{args.signal} ({SIGNAL_DEFS.get(args.signal, '')})"
    )
    print(f"  信号: {sig_note}")
    print(f"  范围: {'全行业 × 全KOL' if universe_all else args.industry} | 模式: {args.mode} | 成本: {args.cost_bps}bp")
    print(f"  收益列: {args.ret_col}")
    print(f"  交易腿数(日×行业): {len(res['panel'])} | 组合交易日: {len(res['combined'])}")

    print(f"\n  【分行业腿】全样本 leg-days={sf['n_days']}:")
    print(f"    总收益(累加腿): {sf['total_return']:.2%}" if sf.get("total_return") is not None else "")
    print(f"    夏普: {sf.get('sharpe')} | 回撤: {sf.get('max_drawdown')} | IC: {sf.get('ic_signal_fwd_ret')}")

    print(f"\n  【等权组合】每日各行业平均 pnl → 一条净值:")
    print(f"    全样本 {cf.get('n_days')} 日 | 总收益: {cf.get('total_return')}")
    print(f"    夏普: {cf.get('sharpe')} | 回撤: {cf.get('max_drawdown')}")
    print(f"    测试集 {ct.get('n_days')} 日 | 总收益: {ct.get('total_return')} | 夏普: {ct.get('sharpe')}")

    if not res["by_industry"].empty:
        print("\n  【分行业摘要】")
        print(
            res["by_industry"][
                ["industry_label", "n_days", "total_return", "sharpe", "ic_signal_fwd_ret"]
            ].to_string(index=False)
        )

    if universe_all:
        print(f"\n  全部交易明细 -> {TRADES_ALL_CSV}")
        print(f"  组合净值     -> {EQUITY_COMBINED_CSV}")
        print(f"  总报告       -> {REPORT_ALL_CSV}")
        print(f"  分行业报告   -> {REPORT_BY_INDUSTRY_CSV}")
    else:
        print(f"\n  净值曲线 -> {EQUITY_CSV}")
        print(f"  日报表   -> {TRADES_CSV}")
        print(f"  报告     -> {REPORT_CSV}")


if __name__ == "__main__":
    main()
