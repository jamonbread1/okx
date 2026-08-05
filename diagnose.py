#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OKX Quant 回测诊断器。

用途
----
读取 `v2.run_backtest` 生成的 `MULTI_*` 结果文件，做两类检查：

1. 策略表现诊断:收益、胜率、交易数、策略归因、集中度、IS/OOS 落差。
2. 回测真实性检测：逐笔核对 open/close、拒单、未平仓、策略归因丢失、价格异常、
   订单簿/合成深度使用、K 线 OHLC 范围外成交等常见“假成交”风险。

示例
----
python diagnose.py
python diagnose.py --result-dir data/backtest_results_v2 --bar 1D
python diagnose.py --phase oos --max-rows 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


OK = "✅"
WARN = "⚠️"
BAD = "❌"
INFO = "ℹ️"
MAG = "🔎"
CHART = "📊"
LEDGER = "🧾"
SHIELD = "🛡️"
MONEY = "💰"


@dataclass
class Issue:
    level: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Lot:
    symbol: str
    direction: str
    strategy: str
    ts: str
    px: float
    sz: float
    pnl: float
    note: str = ""


@dataclass
class RoundTrip:
    symbol: str
    direction: str
    strategy: str
    entry_ts: str
    exit_ts: str
    entry_px: float
    exit_px: float
    size: float
    open_fee_pnl: float
    close_pnl: float
    funding_pnl: float
    net_pnl: float
    close_reason: str


def _fmt_num(x: Any, digits: int = 4) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_metrics(result_dir: str) -> dict:
    path = os.path.join(result_dir, "MULTI_metrics.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _phase_files(result_dir: str, phase: str) -> list[tuple[str, str]]:
    phase = phase.lower()
    files = []
    if phase in ("all", "is", "in", "in_sample"):
        files.append(("IS", os.path.join(result_dir, "MULTI_IS_trades.csv")))
    if phase in ("all", "oos", "out", "out_of_sample"):
        files.append(("OOS", os.path.join(result_dir, "MULTI_OOS_trades.csv")))
    return files


def _as_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        v = row.get(col, default)
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _as_str(row: pd.Series, col: str, default: str = "") -> str:
    v = row.get(col, default)
    if pd.isna(v):
        return default
    return str(v)


def _is_open(side: str) -> bool:
    return side.startswith("open")


def _is_close(side: str) -> bool:
    return side.startswith("close") or side.startswith("partial")


def _is_reject(side: str) -> bool:
    return side.startswith("reject")


def _direction_from_side(side: str, row_dir: str) -> str:
    if row_dir in ("long", "short", "funding"):
        return row_dir
    if "long" in side:
        return "long"
    if "short" in side:
        return "short"
    return row_dir or "flat"


def audit_ledger(df: pd.DataFrame, phase: str) -> tuple[list[RoundTrip], list[Issue], dict[str, Any]]:
    issues: list[Issue] = []
    round_trips: list[RoundTrip] = []
    open_lots: dict[tuple[str, str], deque[Lot]] = defaultdict(deque)
    funding_by_key: dict[tuple[str, str], float] = defaultdict(float)
    counters = Counter()

    if df.empty:
        return [], [Issue("warn", f"{phase}: trades 文件为空")], {"rows": 0}

    required = {"symbol", "side", "sz", "px", "pnl"}
    missing = required - set(df.columns)
    if missing:
        issues.append(Issue("bad", f"{phase}: trades 缺少必要列 {sorted(missing)}"))

    if "ts" in df.columns:
        df = df.copy()
        # 保留 CSV 原始成交顺序。不要按 side 排序，否则同一时间戳的
        # open/close 会被字母序改写，制造“孤儿平仓”假阳性。
        df["_orig_order"] = range(len(df))
        df["_ts_sort"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values(["_ts_sort", "_orig_order"], kind="stable")

    for idx, row in df.iterrows():
        side = _as_str(row, "side")
        symbol = _as_str(row, "symbol", "UNKNOWN")
        direction = _direction_from_side(side, _as_str(row, "dir"))
        strategy = _as_str(row, "strategy", "default") or "default"
        ts = _as_str(row, "ts")
        sz = abs(_as_float(row, "sz"))
        px = _as_float(row, "px")
        pnl = _as_float(row, "pnl")
        note = _as_str(row, "note")

        counters[side] += 1
        counters[f"strategy:{strategy}"] += 1

        if _is_reject(side):
            issues.append(Issue("warn", f"{phase}: 拒单 {side}", {"row": int(idx), "symbol": symbol, "reason": _as_str(row, "fill_reason")}))
            continue

        if side == "funding":
            # 资金费挂到同方向的当前持仓；如果方向未知则按 symbol 级别暂存。
            for key in list(open_lots.keys()):
                if key[0] == symbol and open_lots[key]:
                    funding_by_key[key] += pnl
            continue

        if (_is_open(side) or _is_close(side)) and (sz <= 0 or px <= 0):
            issues.append(Issue("bad", f"{phase}: 非法成交数量/价格", {"row": int(idx), "side": side, "sz": sz, "px": px}))
            continue

        key = (symbol, direction)
        if _is_open(side):
            open_lots[key].append(Lot(symbol, direction, strategy, ts, px, sz, pnl, note))
            continue

        if _is_close(side):
            remaining = sz
            if not open_lots[key]:
                issues.append(Issue("bad", f"{phase}: 平仓没有对应开仓", {"row": int(idx), "symbol": symbol, "side": side, "sz": sz, "strategy": strategy}))
                continue

            while remaining > 1e-12 and open_lots[key]:
                lot = open_lots[key][0]
                take = min(remaining, lot.sz)
                ratio = take / max(lot.sz, 1e-12)
                open_fee_part = lot.pnl * ratio
                close_pnl_part = pnl * (take / max(sz, 1e-12))
                funding_part = funding_by_key.get(key, 0.0) * ratio
                round_trips.append(RoundTrip(
                    symbol=symbol,
                    direction=direction,
                    strategy=lot.strategy,
                    entry_ts=lot.ts,
                    exit_ts=ts,
                    entry_px=lot.px,
                    exit_px=px,
                    size=take,
                    open_fee_pnl=open_fee_part,
                    close_pnl=close_pnl_part,
                    funding_pnl=funding_part,
                    net_pnl=open_fee_part + close_pnl_part + funding_part,
                    close_reason=_as_str(row, "close_reason", note),
                ))
                lot.sz -= take
                remaining -= take
                if lot.sz <= 1e-12:
                    open_lots[key].popleft()
                    funding_by_key[key] = 0.0

            if remaining > 1e-10:
                issues.append(Issue("bad", f"{phase}: 平仓数量超过已开仓", {"row": int(idx), "symbol": symbol, "side": side, "over_sz": remaining}))

            if strategy == "default" and "eod_force_close" in note:
                issues.append(Issue("warn", f"{phase}: 期末强平使用 default 策略归因", {"row": int(idx), "symbol": symbol, "side": side}))

    for (symbol, direction), lots in open_lots.items():
        for lot in lots:
            if lot.sz > 1e-12:
                issues.append(Issue("bad", f"{phase}: 期末仍有未平仓 lot", {"symbol": symbol, "direction": direction, "strategy": lot.strategy, "sz": lot.sz, "entry_ts": lot.ts}))

    stats = {
        "rows": int(len(df)),
        "round_trips": int(len(round_trips)),
        "rejects": int(sum(v for k, v in counters.items() if str(k).startswith("reject"))),
        "opens": int(sum(v for k, v in counters.items() if str(k).startswith("open"))),
        "closes": int(sum(v for k, v in counters.items() if str(k).startswith("close") or str(k).startswith("partial"))),
    }
    return round_trips, issues, stats


def strategy_diagnostics(metrics: dict, phase_tables: dict[str, pd.DataFrame]) -> list[Issue]:
    issues: list[Issue] = []
    for phase_name, key in (("IS", "in_sample"), ("OOS", "out_of_sample")):
        m = metrics.get(key) or metrics.get("in_sample_70pct" if phase_name == "IS" else "out_of_sample_30pct") or {}
        attr = (m.get("attribution") or {}).get("per_strategy") or {}
        trade_count = int(m.get("trade_count") or m.get("close_leg_count") or 0)
        if trade_count < 15:
            issues.append(Issue("warn", f"{phase_name}: 交易数过低，统计不稳定", {"trade_count": trade_count}))
        if attr and trade_count > 0:
            top_name, top_val = max(attr.items(), key=lambda kv: int((kv[1] or {}).get("n") or 0))
            top_n = int((top_val or {}).get("n") or 0)
            if top_n / max(trade_count, 1) > 0.75:
                issues.append(Issue("warn", f"{phase_name}: 策略交易过度集中", {"strategy": top_name, "share": round(top_n / trade_count, 3)}))
            worst_name, worst_val = min(attr.items(), key=lambda kv: float((kv[1] or {}).get("pnl") or 0.0))
            worst_pnl = float((worst_val or {}).get("pnl") or 0.0)
            if worst_pnl < 0:
                issues.append(Issue("info", f"{phase_name}: 最大负贡献策略", {"strategy": worst_name, "pnl": round(worst_pnl, 4)}))

    is_m = metrics.get("in_sample") or {}
    oos_m = metrics.get("out_of_sample") or {}
    if is_m and oos_m:
        is_wr = is_m.get("win_rate")
        oos_wr = oos_m.get("win_rate")
        if is_wr is not None and oos_wr is not None and float(is_wr) - float(oos_wr) > 0.10:
            issues.append(Issue("warn", "IS/OOS 胜率落差明显", {"is_win_rate": is_wr, "oos_win_rate": oos_wr}))
    return issues


def load_bars_for_check(symbols: list[str], bar: str, start: Any, end: Any) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    try:
        from backtest.trade_pipeline import load_bars
    except Exception:
        return out
    for symbol in symbols:
        try:
            df = load_bars(symbol, bar=bar, start=pd.Timestamp(start) if start else None, end=pd.Timestamp(end) if end else None)
            if not df.empty:
                df = df.copy()
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                out[symbol] = df.sort_values("ts").reset_index(drop=True)
        except Exception:
            continue
    return out


def check_prices_against_bars(df: pd.DataFrame, bar_dfs: dict[str, pd.DataFrame], phase: str, tolerance_bps: float = 20.0) -> list[Issue]:
    issues: list[Issue] = []
    if df.empty or not bar_dfs or "ts" not in df.columns:
        return issues
    tol = tolerance_bps / 10000.0
    for idx, row in df.iterrows():
        side = _as_str(row, "side")
        if not (_is_open(side) or _is_close(side)) or _is_reject(side):
            continue
        symbol = _as_str(row, "symbol")
        bars = bar_dfs.get(symbol)
        if bars is None or bars.empty:
            continue
        ts = pd.Timestamp(_as_str(row, "ts")) if _as_str(row, "ts") else pd.NaT
        if pd.isna(ts):
            continue
        px = _as_float(row, "px")
        # 找到 ts 所在或之前最近一根 bar。回测 next-open 与当前 cursor 可能同 ts，故用最近值。
        loc = bars["ts"].searchsorted(ts, side="right") - 1
        if loc < 0 or loc >= len(bars):
            continue
        b = bars.iloc[int(loc)]
        low = float(b.get("low", float("nan")))
        high = float(b.get("high", float("nan")))
        if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(px)):
            continue
        slip_bps = abs(_as_float(row, "slip_bps", 0.0))
        dynamic_tol = max(tol, (slip_bps + 5.0) / 10000.0)
        lower = low * (1 - dynamic_tol)
        upper = high * (1 + dynamic_tol)
        if px < lower or px > upper:
            issues.append(Issue("bad", f"{phase}: 成交价明显超出对应 K 线 OHLC", {"row": int(idx), "symbol": symbol, "ts": str(ts), "px": px, "low": low, "high": high, "side": side}))
    return issues


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * max(20, len(title)))


def print_issues(issues: list[Issue], max_rows: int) -> None:
    if not issues:
        print(f"{OK} 未发现问题")
        return
    icon = {"bad": BAD, "warn": WARN, "info": INFO}
    for i, issue in enumerate(issues[:max_rows], 1):
        ctx = ""
        if issue.context:
            ctx = " | " + ", ".join(f"{k}={v}" for k, v in issue.context.items())
        print(f"{i:03d}. {icon.get(issue.level, INFO)} {issue.message}{ctx}")
    if len(issues) > max_rows:
        print(f"... 还有 {len(issues) - max_rows} 条，使用 --max-rows 调大显示数量")


def round_trips_to_frame(rts: list[RoundTrip]) -> pd.DataFrame:
    return pd.DataFrame([rt.__dict__ for rt in rts])


def main() -> None:
    ap = argparse.ArgumentParser(description="OKX Quant 回测诊断器")
    ap.add_argument("--result-dir", default="data/backtest_results_v2")
    ap.add_argument("--phase", default="all", choices=["all", "is", "oos", "in", "out", "in_sample", "out_of_sample"])
    ap.add_argument("--bar", default="", help="用于 OHLC 价格核对；默认从 metrics.json 的 main_bar 读取")
    ap.add_argument("--max-rows", type=int, default=60)
    ap.add_argument("--skip-bar-check", action="store_true", help="跳过成交价 vs K 线 OHLC 检查")
    ap.add_argument("--write-roundtrips", action="store_true", help="写出 diagnose_round_trips.csv")
    args = ap.parse_args()

    result_dir = args.result_dir
    metrics = _load_metrics(result_dir)
    if not os.path.isdir(result_dir):
        print(f"{BAD} 结果目录不存在: {result_dir}")
        sys.exit(1)

    print(f"{MAG} OKX Quant Diagnose")
    print(f"result_dir={result_dir}")

    print_section(f"{CHART} 绩效摘要")
    for label, key in (("IS", "in_sample"), ("OOS", "out_of_sample")):
        m = metrics.get(key) or {}
        if not m:
            print(f"{WARN} {label}: metrics 缺失")
            continue
        print(
            f"{label}: return={_fmt_num(m.get('total_return'))} "
            f"pnl={_fmt_num(m.get('total_pnl'))} "
            f"trades={m.get('trade_count', m.get('close_leg_count', '-'))} "
            f"win={_fmt_num(m.get('win_rate'))} "
            f"sharpe={_fmt_num(m.get('sharpe'))} "
            f"dd={_fmt_num(m.get('max_drawdown'))}"
        )
        attr = (m.get("attribution") or {}).get("per_strategy") or {}
        if attr:
            print("  strategy:", ", ".join(f"{k}: pnl={v.get('pnl')} n={v.get('n')}" for k, v in sorted(attr.items())))

    phase_tables: dict[str, pd.DataFrame] = {}
    all_round_trips: list[RoundTrip] = []
    all_issues: list[Issue] = []

    print_section(f"{LEDGER} 逐笔流水核对")
    for label, path in _phase_files(result_dir, args.phase):
        df = _read_csv(path)
        phase_tables[label] = df
        rts, issues, stats = audit_ledger(df, label)
        all_round_trips.extend(rts)
        all_issues.extend(issues)
        print(f"{label}: rows={stats.get('rows', 0)} opens={stats.get('opens', 0)} closes={stats.get('closes', 0)} rejects={stats.get('rejects', 0)} round_trips={stats.get('round_trips', 0)}")

    print_section(f"{SHIELD} 回测真实性检测")
    diag_issues = strategy_diagnostics(metrics, phase_tables)
    all_issues.extend(diag_issues)

    if not args.skip_bar_check:
        bar = args.bar or (metrics.get("in_sample") or metrics.get("out_of_sample") or {}).get("main_bar") or ""
        symbols_text = (metrics.get("in_sample") or metrics.get("out_of_sample") or {}).get("symbols", "")
        symbols = [s.strip() for s in str(symbols_text).split(",") if s.strip()]
        range_meta = metrics.get("range") or {}
        if bar and symbols:
            bars = load_bars_for_check(symbols, str(bar), range_meta.get("start"), range_meta.get("end"))
            if bars:
                for label, df in phase_tables.items():
                    all_issues.extend(check_prices_against_bars(df, bars, label))
                print(f"{OK} 已执行成交价 vs {bar} K线 OHLC 检查，symbols={len(bars)}/{len(symbols)}")
            else:
                print(f"{WARN} 未加载到 K 线，跳过 OHLC 成交价核对")
        else:
            print(f"{WARN} metrics 中缺少 bar/symbols，跳过 OHLC 成交价核对")

    # 成交模式统计
    print_section(f"{MONEY} 成交/深度统计")
    for label, df in phase_tables.items():
        if df.empty:
            continue
        for col in ("ord_type", "fill_mode", "book_mode", "fill_reason"):
            if col in df.columns:
                vc = df[col].fillna("NA").astype(str).value_counts().head(8)
                print(f"{label} {col}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
        if "slip_bps" in df.columns:
            s = pd.to_numeric(df["slip_bps"], errors="coerce").dropna().abs()
            if not s.empty:
                print(f"{label} slip_bps abs: mean={s.mean():.2f} p95={s.quantile(.95):.2f} max={s.max():.2f}")
                if s.quantile(.95) > 50:
                    all_issues.append(Issue("warn", f"{label}: 95% 滑点超过 50bps", {"p95_bps": round(float(s.quantile(.95)), 2)}))

    print_section(f"{MAG} 问题清单")
    severity_rank = {"bad": 0, "warn": 1, "info": 2}
    all_issues = sorted(all_issues, key=lambda x: severity_rank.get(x.level, 9))
    print_issues(all_issues, args.max_rows)

    bad_n = sum(1 for x in all_issues if x.level == "bad")
    warn_n = sum(1 for x in all_issues if x.level == "warn")
    info_n = sum(1 for x in all_issues if x.level == "info")
    print_section("结论")
    if bad_n:
        print(f"{BAD} FAIL: bad={bad_n}, warn={warn_n}, info={info_n}")
    elif warn_n:
        print(f"{WARN} PASS_WITH_WARNINGS: warn={warn_n}, info={info_n}")
    else:
        print(f"{OK} PASS: 未发现严重回测真实性问题")

    if args.write_roundtrips and all_round_trips:
        out = os.path.join(result_dir, "diagnose_round_trips.csv")
        round_trips_to_frame(all_round_trips).to_csv(out, index=False)
        print(f"{OK} round trips 已写出: {out}")


if __name__ == "__main__":
    main()
