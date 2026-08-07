# -*- coding: utf-8 -*-
"""tools.split_oos — 把 IS / OOS 回测按时间窗口拆段分析。

读 `data/backtest_results/MULTI_IS_equity.csv` 和 `MULTI_OOS_equity.csv`，
按预设的窗口边界输出每段：
  - 段内 equity delta (= end - start，单位 USD)
  - 段内 return (delta / start)
  - 段内 max drawdown (peak - trough / peak, 负数)
  - 段内 trade 数（按 close_long/close_short 计数）
  - 段内 funding 行数

默认窗口（BTC 历史大事记）：
  2019-12 ~ 2020-12 : 1D pre-bull 起步
  2021-01 ~ 2021-11 : 大牛顶部
  2021-12 ~ 2022-12 : 熊市
  2023-01 ~ 2023-12 : 复苏
  2024-01 ~ 2024-12 : ETF + 第二轮大牛
  2025-01 ~ 2026-08 : 顶部震荡 / 慢牛

用法
----
  # 默认窗口拆段
  python tools/split_oos.py --result-dir data/backtest_results

  # 自定义窗口（年-月-日 ~ 年-月-日）
  python tools/split_oos.py \\
    --result-dir data/backtest_results \\
    --windows 2020-01-01:2020-12-31,2021-01-01:2021-12-31,2022-01-01:2022-12-31

  # 输出 markdown 报告
  python tools/split_oos.py --result-dir data/backtest_results --format md

  # 同时叠加分析 trades（按 close ts 划段）
  python tools/split_oos.py --result-dir data/backtest_results --trades
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd


# ---- 默认窗口（覆盖 2019-12 ~ 2026-08 全期）----
DEFAULT_WINDOWS: List[Tuple[str, str]] = [
    ("2019-12-01", "2020-12-31"),  # 起步 + 312 暴跌
    ("2021-01-01", "2021-11-10"),  # 大牛顶部（11/10 是 ATH 附近）
    ("2021-11-11", "2022-12-31"),  # 熊市（FTX 暴雷等）
    ("2023-01-01", "2023-12-31"),  # 复苏
    ("2024-01-01", "2024-12-31"),  # ETF + 第二轮大牛
    ("2025-01-01", "2026-08-31"),  # 顶部震荡 / 慢牛
]


@dataclass
class SegmentMetrics:
    start: str
    end: str
    n_bars: int = 0
    equity_start: float = 0.0
    equity_end: float = 0.0
    equity_delta: float = 0.0
    segment_return: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    funding_pnl: float = 0.0
    funding_count: int = 0


@dataclass
class Report:
    windows: List[SegmentMetrics] = field(default_factory=list)
    overall: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---- helpers ----
def _parse_windows(raw: str) -> List[Tuple[str, str]]:
    """解析 --windows 2020-01-01:2020-12-31,2021-01-01:2021-12-31"""
    out: List[Tuple[str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":")
        out.append((a.strip(), b.strip()))
    return out


def _load_equity(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 容错：列名可能是 ts/equity 或其他
    if "ts" not in df.columns or "equity" not in df.columns:
        # 尝试 index 0/1
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "ts", df.columns[1]: "equity"})
        else:
            return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["ts", "equity"]).sort_values("ts").reset_index(drop=True)
    return df


def _load_trades(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["pnl"] = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    return df


def _segment_equity(eq: pd.DataFrame, start: str, end: str) -> SegmentMetrics:
    m = SegmentMetrics(start=start, end=end)
    s = pd.Timestamp(start, tz="UTC") if "T" in start or "Z" in start else pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") if "T" in end or "Z" in end else pd.Timestamp(end, tz="UTC")
    seg = eq[(eq["ts"] >= s) & (eq["ts"] <= e)].reset_index(drop=True)
    if seg.empty:
        m.warnings = ["no equity in segment"]  # type: ignore[attr-defined]
        return m
    m.n_bars = len(seg)
    m.equity_start = float(seg["equity"].iloc[0])
    m.equity_end = float(seg["equity"].iloc[-1])
    m.equity_delta = m.equity_end - m.equity_start
    m.segment_return = m.equity_delta / m.equity_start if m.equity_start > 0 else 0.0
    # max drawdown: peak -> trough within segment
    running_max = seg["equity"].cummax()
    dd = (seg["equity"] - running_max) / running_max
    m.max_drawdown = float(dd.min()) if not dd.empty else 0.0
    idx_min = int(dd.idxmin()) if not dd.empty else 0
    m.peak_equity = float(running_max.iloc[idx_min]) if idx_min < len(running_max) else m.equity_start
    m.trough_equity = float(seg["equity"].iloc[idx_min]) if idx_min < len(seg) else m.equity_end
    return m


def _segment_trades(
    trades: pd.DataFrame, start: str, end: str, base: SegmentMetrics,
) -> SegmentMetrics:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    seg = trades[(trades["ts"] >= s) & (trades["ts"] <= e)].reset_index(drop=True)
    closes = seg[seg["side"].astype(str).str.startswith("close")] if "side" in seg.columns else seg.iloc[:0]
    funding = seg[seg["side"].astype(str) == "funding"] if "side" in seg.columns else seg.iloc[:0]
    base.trade_count = int(len(closes))
    base.win_count = int((closes["pnl"] > 0).sum()) if not closes.empty else 0
    base.loss_count = int((closes["pnl"] < 0).sum()) if not closes.empty else 0
    base.funding_count = int(len(funding))
    base.funding_pnl = float(funding["pnl"].sum()) if not funding.empty else 0.0
    return base


def _phase_label(start: str, end: str) -> str:
    return f"{start[:10]} ~ {end[:10]}"


def _format_row(seg: SegmentMetrics, fmt: str) -> str:
    if fmt == "md":
        cells = [
            f"`{_phase_label(seg.start, seg.end)}`",
            f"{seg.n_bars}",
            f"{seg.equity_start:.2f}",
            f"{seg.equity_end:.2f}",
            f"{seg.equity_delta:+.2f}",
            f"{seg.segment_return*100:+.2f}%",
            f"{seg.max_drawdown*100:.2f}%",
            f"{seg.trade_count}",
            f"{seg.win_count}",
            f"{seg.loss_count}",
            f"{seg.funding_pnl:+.2f}",
        ]
        return "| " + " | ".join(cells) + " |"
    return (
        f"{_phase_label(seg.start, seg.end):<28s}  "
        f"bars={seg.n_bars:>5}  eq=[{seg.equity_start:9.2f} -> {seg.equity_end:9.2f}]  "
        f"delta={seg.equity_delta:+8.2f}  ret={seg.segment_return*100:+6.2f}%  "
        f"mdd={seg.max_drawdown*100:6.2f}%  trades={seg.trade_count:>3} "
        f"(W{seg.win_count}/L{seg.loss_count})  funding_pnl={seg.funding_pnl:+.2f}"
    )


def _format_header(fmt: str) -> str:
    if fmt == "md":
        return (
            "| 区间 | bars | equity_start | equity_end | delta | return | max_dd | "
            "trades | wins | losses | funding_pnl |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
    return (
        f"{'区间':<28s}  {'bars':>5}  {'equity':<25s}  {'delta':>9s}  "
        f"{'return':>8s}  {'max_dd':>8s}  {'trades':>8s}  {'funding':>9s}"
    )


def build_report(
    result_dir: str,
    windows: List[Tuple[str, str]],
    include_trades: bool = True,
) -> Report:
    rep = Report()
    is_eq = _load_equity(os.path.join(result_dir, "MULTI_IS_equity.csv"))
    oos_eq = _load_equity(os.path.join(result_dir, "MULTI_OOS_equity.csv"))
    if is_eq is None or oos_eq is None:
        rep.warnings.append(
            f"missing equity csv in {result_dir} (need MULTI_IS_equity.csv + MULTI_OOS_equity.csv)"
        )
        return rep
    full_eq = pd.concat([is_eq, oos_eq], ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    is_trades = _load_trades(os.path.join(result_dir, "MULTI_IS_trades.csv"))
    oos_trades = _load_trades(os.path.join(result_dir, "MULTI_OOS_trades.csv"))
    full_trades = (
        pd.concat([is_trades, oos_trades], ignore_index=True)
        if is_trades is not None and oos_trades is not None
        else None
    )

    for s, e in windows:
        m = _segment_equity(full_eq, s, e)
        if include_trades and full_trades is not None:
            m = _segment_trades(full_trades, s, e, m)
        rep.windows.append(m)

    if not full_eq.empty:
        rep.overall = {
            "ts_start": str(full_eq["ts"].iloc[0]),
            "ts_end": str(full_eq["ts"].iloc[-1]),
            "n_bars": int(len(full_eq)),
            "equity_start": float(full_eq["equity"].iloc[0]),
            "equity_end": float(full_eq["equity"].iloc[-1]),
            "total_return": float(
                (full_eq["equity"].iloc[-1] - full_eq["equity"].iloc[0]) / max(full_eq["equity"].iloc[0], 1e-9)
            ),
        }
    return rep


def format_report(rep: Report, fmt: str = "text") -> str:
    if fmt == "json":
        out = {
            "overall": rep.overall,
            "warnings": rep.warnings,
            "windows": [
                {k: getattr(m, k) for k in m.__dataclass_fields__}
                for m in rep.windows
            ],
        }
        return json.dumps(out, indent=2, ensure_ascii=False, default=str)

    lines: List[str] = []
    if rep.warnings:
        lines.append("⚠️  WARNINGS:")
        for w in rep.warnings:
            lines.append(f"  - {w}")
        lines.append("")
    if rep.overall:
        o = rep.overall
        lines.append(
            f"📊 Overall: {o.get('ts_start', '?')[:10]} ~ {o.get('ts_end', '?')[:10]}  "
            f"bars={o.get('n_bars', 0)}  equity=[{o.get('equity_start', 0):.2f} -> {o.get('equity_end', 0):.2f}]  "
            f"return={o.get('total_return', 0)*100:+.2f}%"
        )
        lines.append("")
    lines.append("📅 Per-Window:")
    lines.append(_format_header(fmt))
    for seg in rep.windows:
        lines.append(_format_row(seg, fmt))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="按时间窗口拆解 IS/OOS 回测结果")
    ap.add_argument("--result-dir", default="data/backtest_results",
                    help="MULTI_* csv 所在目录")
    ap.add_argument("--windows", default="",
                    help="自定义窗口，逗号分隔，格式 start:end (YYYY-MM-DD:YYYY-MM-DD)；"
                         "留空使用默认 BTC 大事记 6 段")
    ap.add_argument("--format", choices=["text", "md", "json"], default="text")
    ap.add_argument("--no-trades", dest="trades", action="store_false",
                    help="不读 trades.csv（只算 equity 段）")
    ap.add_argument("--out", default="", help="输出到文件（默认 stdout）")
    args = ap.parse_args()

    windows = _parse_windows(args.windows) if args.windows else DEFAULT_WINDOWS
    rep = build_report(args.result_dir, windows, include_trades=args.trades)
    text = format_report(rep, fmt=args.format)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"✅ 写入 {args.out} ({len(rep.windows)} 段)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
