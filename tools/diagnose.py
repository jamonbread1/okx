# -*- coding: utf-8 -*-
"""tools.diagnose — 策略诊断报告。

读 ``data/backtest_results/MULTI_*_trades.csv`` 和 ``*_equity.csv``，
输出关键诊断指标：

1. 绩效摘要（IS / OOS）
2. 逐笔流水核对（opens / closes / rejects / round_trips）
3. 回测真实性检测（已执行成交价 vs OHLC）
4. 成交/深度统计（fill_mode / slip_bps）
5. **退出原因分布**（initial_stop / r1.5_partial / chandelier / max_bars / mfe_decay / breakout_fail / panic / hard_tp）
6. **MFE / MAE 分布**（按 R 桶统计：0~0.5R / 0.5~1R / 1~1.5R / 1.5~2R / 2~3R / 3R+）
7. **右尾截断检测**（MFE >= 2R 但 realized_r < 1R 的比例 → +2R 后策略是否让收益跑出去）
8. **多空比**（长 / 短 / funding）
9. **资金暴露率**（time_in_market = 有仓时间 / 总时间）

用法:
  python tools/diagnose.py --result-dir data/backtest_results
  python tools/diagnose.py --result-dir data/backtest_results --out data/diagnose_report.md

输出：控制台 + 可选 md 报告
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---- 退出原因归一化 ----
# 引擎 close reason 可能含 : SL long / chandelier / r1.5 partial / mfe_decay / max_bars / breakout_fail / panic / timeout
EXIT_PATTERNS = [
    ("chandelier", r"chandelier"),
    ("r1.5_partial", r"r1\.5 partial"),
    ("hard_tp", r"^TP\d+\s*清"),
    ("partial_tp", r"^TP\d+\s*@"),
    ("mfe_decay", r"mfe_decay"),
    ("breakout_fail", r"breakout_fail"),
    ("max_bars", r"max_bars"),
    ("panic_gap", r"panic_gap"),
    ("timeout", r"^timeout"),
    ("sl_hit", r"^SL\s"),
]


def _classify_exit(reason: str) -> str:
    s = str(reason or "")
    for name, pat in EXIT_PATTERNS:
        if re.search(pat, s):
            return name
    if "SL" in s or "sl" in s:
        return "sl_hit"
    if "close" in s.lower() and "no signal" not in s.lower():
        return "other"
    return "unknown"


def _mfe_bucket(r: float) -> str:
    if r < 0:
        return "<0"
    if r < 0.5:
        return "0~0.5"
    if r < 1.0:
        return "0.5~1.0"
    if r < 1.5:
        return "1.0~1.5"
    if r < 2.0:
        return "1.5~2.0"
    if r < 3.0:
        return "2.0~3.0"
    if r < 5.0:
        return "3.0~5.0"
    return "5.0+"


def _list_phases(result_dir: str) -> List[str]:
    """列出 result_dir 下的所有 phase（in_sample / out_of_sample 等）。"""
    phases = set()
    for fp in glob.glob(os.path.join(result_dir, "MULTI_*_equity.csv")):
        fn = os.path.basename(fp)
        m = re.match(r"MULTI_(.+?)_equity\.csv$", fn)
        if m:
            phases.add(m.group(1))
    if not phases:
        # 兜底：用 summary JSON 找 phase 字段
        for fp in glob.glob(os.path.join(result_dir, "MULTI_*_summary.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                p = d.get("phase")
                if p:
                    phases.add(p)
            except Exception:
                pass
    return sorted(phases)


def _phase_paths(phase: str, result_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """返回 (trades, equity, summary, fill_stats) 路径（可能 None）。"""
    p = {
        "trades": os.path.join(result_dir, f"MULTI_{phase}_trades.csv"),
        "equity": os.path.join(result_dir, f"MULTI_{phase}_equity.csv"),
        "summary": os.path.join(result_dir, f"MULTI_{phase}_summary.json"),
        "fills": os.path.join(result_dir, f"MULTI_{phase}_fills.csv"),
    }
    return p["trades"], p["equity"], p["summary"], p["fills"]


def _load_summary(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_trades(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_equity(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_fills(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _exit_distribution(trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """按 reason 计算退出分布。"""
    if "reason" not in trades.columns or "side" not in trades.columns:
        return {}
    closes = trades[trades["side"].astype(str).str.startswith("close")].copy()
    if closes.empty:
        return {}
    closes["exit_class"] = closes["reason"].fillna("").astype(str).map(_classify_exit)
    by_class = closes.groupby("exit_class")
    out: Dict[str, Dict[str, float]] = {}
    for klass, grp in by_class:
        pnl = pd.to_numeric(grp.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        n = int(len(grp))
        win = int((pnl > 0).sum())
        out[str(klass)] = {
            "n": n,
            "win_rate": (win / n) if n else 0.0,
            "total_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()) if n else 0.0,
            "median_pnl": float(pnl.median()) if n else 0.0,
        }
    return out


def _mfe_mae_distribution(trades: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """按 MFE/MAE 桶统计关闭笔数。"""
    out = {"mfe_buckets": {}, "mae_buckets": {}}
    if "reason" not in trades.columns or "side" not in trades.columns:
        return out
    closes = trades[trades["side"].astype(str).str.startswith("close")].copy()
    if closes.empty:
        return out
    # mfe_r / mae_r 可能在 close 行的 reason 字段里以 " mfe=1.23R mae=0.45R" 形式
    # 也可能在专用列 mfe_r / mae_r 里
    if "mfe_r" in closes.columns:
        mfe = pd.to_numeric(closes["mfe_r"], errors="coerce")
    else:
        mfe = pd.Series([float("nan")] * len(closes))
    if "mae_r" in closes.columns:
        mae = pd.to_numeric(closes["mae_r"], errors="coerce")
    else:
        mae = pd.Series([float("nan")] * len(closes))
    mfe_buckets = mfe.map(_mfe_bucket).value_counts().to_dict() if mfe.notna().any() else {}
    mae_buckets = mae.map(_mfe_bucket).value_counts().to_dict() if mae.notna().any() else {}
    out["mfe_buckets"] = {str(k): int(v) for k, v in mfe_buckets.items()}
    out["mae_buckets"] = {str(k): int(v) for k, v in mae_buckets.items()}
    return out


def _right_tail_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    """右尾截断检测：MFE >= 2R 但 realized_r < 1R 的比例。

    如果 realized_pnl_r 在 mfe>=2r 的交易里 80%+ 都是 <1R → 说明 +2R 后立刻被打回去，
    策略没让右尾收益跑出去。
    """
    out: Dict[str, Any] = {}
    if "side" not in trades.columns:
        return out
    closes = trades[trades["side"].astype(str).str.startswith("close")].copy()
    if closes.empty:
        return out
    if "mfe_r" not in closes.columns or "pnl_r" not in closes.columns:
        # 退化：用 mfe / pnl 字段估
        mfe_col = "mfe_r" if "mfe_r" in closes.columns else None
        pnl_col = "pnl_r" if "pnl_r" in closes.columns else ("pnl" if "pnl" in closes.columns else None)
        if not mfe_col or not pnl_col:
            return out
        mfe = pd.to_numeric(closes[mfe_col], errors="coerce")
        pnl_r = pd.to_numeric(closes[pnl_col], errors="coerce")
    else:
        mfe = pd.to_numeric(closes["mfe_r"], errors="coerce")
        pnl_r = pd.to_numeric(closes["pnl_r"], errors="coerce")
    if not mfe.notna().any() or not pnl_r.notna().any():
        return out
    valid = mfe.notna() & pnl_r.notna()
    if not valid.any():
        return out
    mfe_v = mfe[valid]
    pnl_v = pnl_r[valid]
    n_total = int(len(mfe_v))
    n_mfe_ge_2 = int((mfe_v >= 2.0).sum())
    n_mfe_ge_2_pnl_lt_1 = int(((mfe_v >= 2.0) & (pnl_v < 1.0)).sum())
    n_mfe_ge_2_pnl_ge_2 = int(((mfe_v >= 2.0) & (pnl_v >= 2.0)).sum())
    n_mfe_ge_3 = int((mfe_v >= 3.0).sum())
    n_mfe_ge_3_pnl_ge_3 = int(((mfe_v >= 3.0) & (pnl_v >= 3.0)).sum())
    out = {
        "n_closed": n_total,
        "n_mfe_ge_2": n_mfe_ge_2,
        "pct_mfe_ge_2": (n_mfe_ge_2 / n_total) if n_total else 0.0,
        "truncation_pct": (n_mfe_ge_2_pnl_lt_1 / n_mfe_ge_2) if n_mfe_ge_2 else 0.0,
        "n_mfe_ge_2_realized_ge_2": n_mfe_ge_2_pnl_ge_2,
        "right_tail_conversion": (n_mfe_ge_2_pnl_ge_2 / n_mfe_ge_2) if n_mfe_ge_2 else 0.0,
        "n_mfe_ge_3": n_mfe_ge_3,
        "n_mfe_ge_3_realized_ge_3": n_mfe_ge_3_pnl_ge_3,
        "right_tail_3R_conversion": (n_mfe_ge_3_pnl_ge_3 / n_mfe_ge_3) if n_mfe_ge_3 else 0.0,
    }
    return out


def _direction_breakdown(summary: Dict[str, Any]) -> Dict[str, Any]:
    """多空比 + funding。"""
    attr = summary.get("attribution") or {}
    per_dir = attr.get("per_direction") or {}
    out = {}
    for k, v in per_dir.items():
        out[str(k)] = float(v) if isinstance(v, (int, float)) else 0.0
    return out


def _time_in_market(equity: pd.DataFrame, trades: pd.DataFrame) -> Optional[float]:
    """资金暴露率 = 仓位非零的 K 线数 / 总 K 线数。"""
    if equity is None or equity.empty or "ts" not in equity.columns:
        return None
    eq = equity.copy()
    eq["ts"] = pd.to_datetime(eq["ts"], errors="coerce", utc=True)
    eq = eq.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if eq.empty:
        return None
    # 多空仓位列在 multi_engine 输出里通常叫 pos_long_sz / pos_short_sz
    pos_cols = [c for c in eq.columns if c.startswith("pos_") and c.endswith("_sz")]
    if not pos_cols:
        return None
    has_pos = (eq[pos_cols].fillna(0.0).sum(axis=1) > 0).astype(int)
    return float(has_pos.mean())


def _fill_stats(fills: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if fills is None or fills.empty:
        return {}
    out: Dict[str, Any] = {"n": int(len(fills))}
    if "ord_type" in fills.columns:
        out["ord_type"] = fills["ord_type"].fillna("NA").value_counts().to_dict()
    if "fill_mode" in fills.columns:
        out["fill_mode"] = fills["fill_mode"].fillna("NA").value_counts().to_dict()
    if "book_mode" in fills.columns:
        out["book_mode"] = fills["book_mode"].fillna("NA").value_counts().to_dict()
    if "fill_reason" in fills.columns:
        out["fill_reason"] = fills["fill_reason"].fillna("NA").value_counts().to_dict()
    if "slip_bps" in fills.columns:
        sb = pd.to_numeric(fills["slip_bps"], errors="coerce").dropna().abs()
        if not sb.empty:
            out["slip_bps_abs"] = {
                "mean": float(sb.mean()),
                "p50": float(sb.median()),
                "p95": float(sb.quantile(0.95)),
                "max": float(sb.max()),
            }
    return out


def _round_trip_count(trades: pd.DataFrame) -> int:
    if "side" not in trades.columns or "ts" not in trades.columns:
        return 0
    opens = trades["side"].astype(str).str.startswith("open").sum()
    closes = trades["side"].astype(str).str.startswith("close").sum()
    return int(min(opens, closes))


def diagnose_phase(phase: str, result_dir: str) -> Dict[str, Any]:
    t_path, e_path, s_path, f_path = _phase_paths(phase, result_dir)
    summary = _load_summary(s_path)
    trades = _load_trades(t_path)
    equity = _load_equity(e_path)
    fills = _load_fills(f_path)

    # 预先算 bar_clamp 占比 (给 build_warnings 用)
    n_clamped, n_total_fill, clamp_ratio = _fill_clamp_ratio(fills)

    out: Dict[str, Any] = {
        "phase": phase,
        "_result_dir": result_dir,  # 给 build_warnings 用, 读 fills.csv
        "_n_clamped": n_clamped,
        "_n_total_fill": n_total_fill,
        "_clamp_ratio": clamp_ratio,
        "summary": {
            "total_return": summary.get("total_return", 0.0),
            "sharpe": summary.get("sharpe", 0.0),
            "max_drawdown": summary.get("max_drawdown", 0.0),
            "trade_count": summary.get("trade_count", 0),
            "win_rate": summary.get("win_rate", 0.0),
            "total_pnl": summary.get("total_pnl", 0.0),
            "calmar": summary.get("calmar", 0.0),
            "volatility_ann": summary.get("volatility_ann", 0.0),
            "bars": summary.get("bars", 0),
        },
        "attribution": summary.get("attribution", {}),
        "round_trip_count": _round_trip_count(trades) if trades is not None else 0,
        "opens": int((trades["side"].astype(str).str.startswith("open")).sum()) if trades is not None and "side" in trades.columns else 0,
        "closes": int((trades["side"].astype(str).str.startswith("close")).sum()) if trades is not None and "side" in trades.columns else 0,
        "rejects": int((trades["side"].astype(str).str.contains("reject")).sum()) if trades is not None and "side" in trades.columns else 0,
    }

    if trades is not None and not trades.empty:
        out["exit_distribution"] = _exit_distribution(trades)
        out["mfe_mae_distribution"] = _mfe_mae_distribution(trades)
        out["right_tail"] = _right_tail_stats(trades)
    if equity is not None and not equity.empty:
        tim = _time_in_market(equity, trades if trades is not None else pd.DataFrame())
        if tim is not None:
            out["time_in_market"] = tim
    if fills is not None and not fills.empty:
        out["fill_stats"] = _fill_stats(fills)

    out["direction_breakdown"] = _direction_breakdown(summary)
    return out


def format_console(report: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("🔎 OKX Quant Diagnose")
    lines.append(f"result_dir={report.get('result_dir', '?')}")
    lines.append("")

    for phase in report.get("phases", []):
        d = report["diagnostics"].get(phase, {})
        s = d.get("summary", {})
        lines.append(f"📊 绩效摘要（{phase}）")
        lines.append("-" * 60)
        lines.append(
            f"  return={s.get('total_return', 0):.4f} pnl={s.get('total_pnl', 0):.2f} "
            f"trades={s.get('trade_count', 0)} win={s.get('win_rate', 0):.4f} "
            f"sharpe={s.get('sharpe', 0):.4f} dd={s.get('max_drawdown', 0):.4f} "
            f"calmar={s.get('calmar', 0):.2f}"
        )
        # attribution
        attr = d.get("attribution", {})
        per_s = (attr.get("per_strategy") or {})
        if per_s:
            parts = [f"{k}: pnl={v.get('pnl', 0):.2f} n={v.get('n', 0)}" if isinstance(v, dict) else f"{k}={v}" for k, v in per_s.items()]
            lines.append(f"  per_strategy: " + ", ".join(parts))
        per_d = (attr.get("per_direction") or {})
        if per_d:
            parts = [f"{k}={v:.2f}" for k, v in per_d.items() if isinstance(v, (int, float))]
            lines.append(f"  per_direction: " + ", ".join(parts))
        lines.append("")

        # round_trip
        lines.append("🧾 逐笔流水核对")
        lines.append("-" * 60)
        lines.append(
            f"  opens={d.get('opens', 0)} closes={d.get('closes', 0)} "
            f"rejects={d.get('rejects', 0)} round_trips={d.get('round_trip_count', 0)}"
        )
        lines.append("")

        # right tail
        rt = d.get("right_tail", {})
        if rt:
            lines.append("🎯 右尾截断检测（MFE vs Realized R）")
            lines.append("-" * 60)
            lines.append(
                f"  n_closed={rt.get('n_closed', 0)} | MFE≥2R 笔数={rt.get('n_mfe_ge_2', 0)} "
                f"({rt.get('pct_mfe_ge_2', 0):.1%} of total)"
            )
            lines.append(
                f"  MFE≥2R 但 realized<1R 比例 = {rt.get('truncation_pct', 0):.1%}  "
                f"(<30% 算健康)"
            )
            lines.append(
                f"  MFE≥2R → realized≥2R 转化率 = {rt.get('right_tail_conversion', 0):.1%}  "
                f"(<20% 说明 +2R 启 Chandelier 后被打回)"
            )
            if rt.get("n_mfe_ge_3", 0) > 0:
                lines.append(
                    f"  MFE≥3R 笔数 = {rt.get('n_mfe_ge_3', 0)}, "
                    f"其中 realized≥3R = {rt.get('n_mfe_ge_3_realized_ge_3', 0)} "
                    f"({rt.get('right_tail_3R_conversion', 0):.1%})"
                )
            lines.append("")

        # exit distribution
        ex = d.get("exit_distribution", {})
        if ex:
            lines.append("🚪 退出原因分布")
            lines.append("-" * 60)
            sorted_ex = sorted(ex.items(), key=lambda kv: -kv[1]["n"])
            for klass, info in sorted_ex:
                lines.append(
                    f"  {klass:20s} n={info['n']:4d}  win={info['win_rate']:.1%}  "
                    f"total_pnl={info['total_pnl']:8.2f}  avg={info['avg_pnl']:7.2f}"
                )
            lines.append("")

        # MFE / MAE
        mfe_mae = d.get("mfe_mae_distribution", {})
        if mfe_mae.get("mfe_buckets") or mfe_mae.get("mae_buckets"):
            lines.append("📈 MFE / MAE 分布（按 R 桶）")
            lines.append("-" * 60)
            mfe_b = mfe_mae.get("mfe_buckets", {})
            mae_b = mfe_mae.get("mae_buckets", {})
            buckets = ["<0", "0~0.5", "0.5~1.0", "1.0~1.5", "1.5~2.0", "2.0~3.0", "3.0~5.0", "5.0+"]
            for b in buckets:
                mn = mfe_b.get(b, 0)
                an = mae_b.get(b, 0)
                lines.append(f"  {b:>10s}  MFE={mn:4d}  MAE={an:4d}")
            lines.append("")

        # time in market
        if "time_in_market" in d:
            lines.append("⏱  资金暴露率（time_in_market）")
            lines.append("-" * 60)
            lines.append(f"  {d['time_in_market']:.1%} of K-lines have non-zero position")
            lines.append("")

        # fill stats
        fs = d.get("fill_stats", {})
        if fs:
            lines.append("💰 成交 / 深度统计")
            lines.append("-" * 60)
            for k, v in fs.items():
                if isinstance(v, dict):
                    parts = [f"{kk}={vv}" for kk, vv in v.items()]
                    lines.append(f"  {k}: " + ", ".join(parts))
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

    # 全局警告
    if report.get("warnings"):
        lines.append("⚠️  警告 / 建议")
        lines.append("-" * 60)
        for w in report["warnings"]:
            lines.append(f"  {w}")
        lines.append("")
    return "\n".join(lines)


def format_markdown(report: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# OKX Quant 诊断报告")
    lines.append(f"\nresult_dir: `{report.get('result_dir', '?')}`\n")
    for phase in report.get("phases", []):
        d = report["diagnostics"].get(phase, {})
        s = d.get("summary", {})
        lines.append(f"## {phase}")
        lines.append("")
        lines.append(f"- 收益: **{s.get('total_return', 0):.2%}** (pnl={s.get('total_pnl', 0):.2f})")
        lines.append(f"- Sharpe: **{s.get('sharpe', 0):.3f}** | Calmar: {s.get('calmar', 0):.2f}")
        lines.append(f"- 交易: {s.get('trade_count', 0)} | 胜率: {s.get('win_rate', 0):.1%}")
        lines.append(f"- max_dd: {s.get('max_drawdown', 0):.2%}")
        lines.append("")
        # exit table
        ex = d.get("exit_distribution", {})
        if ex:
            lines.append("### 退出原因分布")
            lines.append("")
            lines.append("| 原因 | n | win_rate | total_pnl | avg_pnl |")
            lines.append("|---|---:|---:|---:|---:|")
            for klass, info in sorted(ex.items(), key=lambda kv: -kv[1]["n"]):
                lines.append(
                    f"| {klass} | {info['n']} | {info['win_rate']:.1%} | "
                    f"{info['total_pnl']:.2f} | {info['avg_pnl']:.2f} |"
                )
            lines.append("")
        # right tail
        rt = d.get("right_tail", {})
        if rt:
            lines.append("### 右尾截断检测")
            lines.append("")
            lines.append(f"- n_closed: {rt.get('n_closed', 0)}")
            lines.append(f"- MFE≥2R 笔数: {rt.get('n_mfe_ge_2', 0)} ({rt.get('pct_mfe_ge_2', 0):.1%})")
            lines.append(f"- **MFE≥2R 但 realized<1R 比例: {rt.get('truncation_pct', 0):.1%}** (理想 <30%)")
            lines.append(f"- MFE≥2R → realized≥2R 转化率: {rt.get('right_tail_conversion', 0):.1%} (理想 ≥20%)")
            lines.append("")
        # time in market
        if "time_in_market" in d:
            lines.append(f"### 资金暴露率: {d['time_in_market']:.1%}")
            lines.append("")
    if report.get("warnings"):
        lines.append("## 警告 / 建议")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def _fill_clamp_ratio(fills: Optional[pd.DataFrame]) -> Tuple[int, int, float]:
    """统计 bar_high_clamp / bar_low_clamp 占比, 用于修正 003 误报.

    Returns: (n_clamped, n_total, ratio)
    """
    if fills is None or fills.empty or "fill_reason" not in fills.columns:
        return 0, 0, 0.0
    fr = fills["fill_reason"].fillna("").astype(str)
    n_total = int(len(fr))
    if n_total == 0:
        return 0, 0, 0.0
    n_clamped = int(fr.str.contains("bar_high_clamp|bar_low_clamp").sum())
    return n_clamped, n_total, (n_clamped / n_total) if n_total else 0.0


def _strategy_concentration(summary: Dict[str, Any]) -> Dict[str, float]:
    """返回每个策略在总 PnL 中的占比 (用于检测 001/002 策略过度集中)."""
    attr = summary.get("attribution") or {}
    per_s = attr.get("per_strategy") or {}
    out: Dict[str, float] = {}
    for k, v in per_s.items():
        if isinstance(v, dict):
            out[str(k)] = float(v.get("pnl", 0) or 0)
    return out


def build_warnings(diagnostics: Dict[str, Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    for phase, d in diagnostics.items():
        s = d.get("summary", {})
        # 1. OOS 收益为正但接近 0（资金没在工作）
        if "out_of_sample" in phase and s.get("total_return", 0) > 0 and s.get("total_return", 0) < 0.005:
            warnings.append(
                f"[{phase}] 收益极低 ({s.get('total_return', 0):.2%}) — "
                f"考虑调高 risk_pct 或放开入场门控（min_entry_conf）"
            )
        # 2. 资金暴露率 < 10% → 仓位过小或入场太严
        tim = d.get("time_in_market")
        if tim is not None and tim < 0.10:
            warnings.append(
                f"[{phase}] 资金暴露率仅 {tim:.1%} — "
                f"长期空仓；考虑放宽 trend_min_adx / 加大 hold_bars / 放开 don_min_break_hold_bars"
            )
        # 3. 右尾截断严重
        rt = d.get("right_tail", {})
        trunc = rt.get("truncation_pct", 0)
        if trunc > 0.5:
            warnings.append(
                f"[{phase}] 右尾截断严重: MFE≥2R 但 realized<1R 比例 = {trunc:.1%} — "
                f"+2R 后被打回；考虑放宽 r2_chandelier_k 或缩短 time-stop"
            )
        # 4. 几乎全是 SL 退出
        ex = d.get("exit_distribution", {})
        sl_n = ex.get("sl_hit", {}).get("n", 0)
        total_close = sum(v.get("n", 0) for v in ex.values())
        if total_close > 0 and (sl_n / total_close) > 0.6:
            warnings.append(
                f"[{phase}] SL 退出占比 {sl_n/total_close:.0%} — "
                f"可能 SL 过近（sl_k_vol 偏小）或入场太早；考虑 min_entry_conf 上调"
            )
        # 5. max_bars 退出占比过高 → max_bars 太短
        mb = ex.get("max_bars", {}).get("n", 0)
        if total_close > 0 and (mb / total_close) > 0.3:
            warnings.append(
                f"[{phase}] max_bars 退出占比 {mb/total_close:.0%} — "
                f"max_bars 偏短；考虑延长 timeout_max_bars"
            )
        # 6. 多空严重失衡
        per_d = d.get("direction_breakdown", {})
        long_pnl = per_d.get("long", 0.0)
        short_pnl = per_d.get("short", 0.0)
        if abs(long_pnl - short_pnl) > 50 and (long_pnl + short_pnl) > 0:
            ratio = abs(long_pnl - short_pnl) / max(abs(long_pnl + short_pnl), 1e-9)
            if ratio > 0.7:
                dominant = "long" if long_pnl > short_pnl else "short"
                warnings.append(
                    f"[{phase}] 多空严重失衡: {dominant} 占 {ratio:.0%} — "
                    f"考虑调 htf_weekly_min_gap / 加多空非对称过滤"
                )
        # 7. 001/002 策略过度集中 (用户截图里那条)
        # 注意: --only 单策略跑也会触发, 标注为 INFO
        strat_pnl = _strategy_concentration(s)
        if strat_pnl:
            total_abs = sum(abs(v) for v in strat_pnl.values())
            if total_abs > 0:
                top = max(strat_pnl.items(), key=lambda kv: abs(kv[1]))
                share = abs(top[1]) / total_abs
                if share > 0.95:
                    warnings.append(
                        f"[{phase}] [INFO] 策略交易过度集中 | strategy={top[0]}, share={share:.2f} "
                        f"—— 如果是 --only 单跑属预期; 否则考虑组合启用其他策略"
                    )
        # 8. 003 大滑点警告 + bar_clamp 修正 (用户截图里那条)
        # slip_bps 高本身是 1D 周期 + delay_drift_frac=0.05 + 闪崩段的合成盘口固有特性
        # 真正要看的指标: bar_clamp 占比 (> 10% 说明 VWAP 模型有 bug)
        fs = d.get("fill_stats", {})
        sb = fs.get("slip_bps_abs", {})
        clamp_ratio = d.get("_clamp_ratio", 0.0)
        n_clamped = d.get("_n_clamped", 0)
        n_total_fill = d.get("_n_total_fill", 0)
        if sb and sb.get("p95", 0) > 50:
            if clamp_ratio < 0.05:
                warnings.append(
                    f"[{phase}] [INFO] 大滑点 p95={sb.get('p95', 0):.1f}bps, "
                    f"但 bar_clamp 占比仅 {clamp_ratio:.1%} "
                    f"({n_clamped}/{n_total_fill}) — 属于 1D 周期 + 合成盘口延迟漂移的固有特性, "
                    f"不是真执行成本. 真正成本看 fill_reason 分布和 total_pnl vs gross."
                )
            elif clamp_ratio > 0.10:
                warnings.append(
                    f"[{phase}] [WARN] 大滑点 p95={sb.get('p95', 0):.1f}bps + bar_clamp 占比 {clamp_ratio:.1%} — "
                    f"VWAP 合成盘口模型可能有问题; 建议缩小 depth_participation 或检查数据"
                )
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="v3 策略诊断（退出原因/MFE/MAE/资金暴露率/右尾截断）")
    ap.add_argument("--result-dir", default="data/backtest_results",
                    help="回测结果目录（含 MULTI_*_trades.csv / _equity.csv / _summary.json）")
    ap.add_argument("--out", default="", help="可选：输出 markdown 报告到文件")
    ap.add_argument("--phases", default="", help="逗号分隔的 phase 列表（默认自动检测）")
    args = ap.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"❌ result_dir 不存在: {args.result_dir}")
        return

    if args.phases:
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    else:
        phases = _list_phases(args.result_dir)
    if not phases:
        print(f"❌ 在 {args.result_dir} 下没找到任何 phase（需要 MULTI_<phase>_equity.csv 或 _summary.json）")
        return

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for phase in phases:
        diagnostics[phase] = diagnose_phase(phase, args.result_dir)

    report = {
        "result_dir": args.result_dir,
        "phases": phases,
        "diagnostics": diagnostics,
        "warnings": build_warnings(diagnostics),
    }
    text = format_console(report)
    print(text)
    if args.out:
        md = format_markdown(report)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n📝 Markdown 报告已写入: {args.out}")


if __name__ == "__main__":
    main()
