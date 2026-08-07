# -*- coding: utf-8 -*-
"""回测绩效指标"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _round_trip_pnls(trades: List[Dict]) -> List[float]:
    """
    按品种重建完整一轮开平仓的累计 PnL（含开仓费、分批止盈、最终平仓、期间资金费）。

    规则：
      - open_* 开始/加仓：累计 pnl（通常为 -fee）
      - close_* 减仓：累计 pnl；当推断仓位归零时结算为一笔 round-trip
      - funding：计入当前未平仓品种
    """
    pos_sz: Dict[str, float] = {}
    acc: Dict[str, float] = {}
    trips: List[float] = []

    for t in trades:
        side = str(t.get("side", "") or "")
        sym = str(t.get("symbol", "") or "_")
        pnl = float(t.get("pnl", 0) or 0)
        sz = abs(float(t.get("sz", 0) or t.get("size", 0) or 0))

        if side.startswith("reject"):
            continue

        if side.startswith("open"):
            pos_sz[sym] = pos_sz.get(sym, 0.0) + sz
            acc[sym] = acc.get(sym, 0.0) + pnl
            continue

        if side == "funding":
            if pos_sz.get(sym, 0.0) > 1e-12:
                acc[sym] = acc.get(sym, 0.0) + pnl
            continue

        if side.startswith("close") or side.startswith("partial"):
            acc[sym] = acc.get(sym, 0.0) + pnl
            if sz > 0:
                pos_sz[sym] = max(0.0, pos_sz.get(sym, 0.0) - sz)
            if pos_sz.get(sym, 0.0) <= 1e-12 and sym in acc:
                trips.append(acc.pop(sym))
                pos_sz[sym] = 0.0
            continue

        # 其它带 pnl 的流水（兼容旧格式）
        if abs(pnl) > 0 and not side.startswith("open"):
            acc[sym] = acc.get(sym, 0.0) + pnl

    # 未平仓腿不计入胜率（只统计已完成 round-trip）
    return trips


def compute_metrics(
    equity: pd.Series,
    trades: Optional[List[Dict]] = None,
    bars_per_year: float = 365 * 24 * 4,  # 15m
) -> Dict:
    eq = equity.astype(float).dropna()
    if len(eq) < 2:
        return {"error": "equity too short"}
    ret = eq.pct_change().fillna(0.0)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    vol = float(ret.std() * np.sqrt(bars_per_year)) if ret.std() > 0 else 0.0
    mean = float(ret.mean() * bars_per_year)
    sharpe = float(mean / vol) if vol > 1e-12 else 0.0
    calmar = float(total_ret / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    out = {
        "start_equity": round(float(eq.iloc[0]), 2),
        "end_equity": round(float(eq.iloc[-1]), 2),
        "total_return": round(total_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "bars": int(len(eq)),
        "volatility_ann": round(vol, 4),
    }
    if trades:
        trips = _round_trip_pnls(trades)
        n = len(trips)
        wins = sum(1 for x in trips if x > 0)
        out["trade_count"] = n
        out["win_rate"] = round(wins / n, 4) if n else 0.0
        out["avg_pnl"] = round(float(np.mean(trips)), 4) if trips else 0.0
        # 分批笔数（仅 close/partial 流水，供诊断）
        legs = [
            t for t in trades
            if str(t.get("side", "")).startswith("close")
            or str(t.get("side", "")).startswith("partial")
        ]
        out["close_leg_count"] = len(legs)
        all_pnls = [float(t.get("pnl", 0) or 0) for t in trades]
        out["total_pnl"] = round(float(np.sum(all_pnls)), 2)
    return out


def metrics_to_frame(is_m: Dict, oos_m: Dict) -> pd.DataFrame:
    keys = sorted(set(is_m) | set(oos_m))
    rows = []
    for k in keys:
        rows.append({"metric": k, "in_sample_70pct": is_m.get(k), "out_of_sample_30pct": oos_m.get(k)})
    return pd.DataFrame(rows)


def _cagr(total_return: float, days: float) -> float:
    """从 total_return 和天数算年化收益。

    raw total_return 在 IS/OOS 时长不等时会被"短时段 OOS 收益看起来更小"误导；
    CAGR 把收益折算到年化，跨时长可比。
    """
    if total_return <= -1.0:
        return -1.0
    end = 1.0 + total_return
    if days <= 0:
        return 0.0
    return end ** (365.0 / days) - 1.0


def check_overfitting(
    is_metrics: Dict,
    oos_metrics: Dict,
    max_sharpe_decay: float = 0.45,
    max_cagr_decay: float = 0.55,
    min_oos_trades: int = 15,
    is_days: float = 0.0,
    oos_days: float = 0.0,
) -> Dict:
    """过拟合检查（用 CAGR 衰减，不是 raw total_return 衰减）。

    raw total_return 在 IS/OOS 时长不等时会被"短时段 OOS 收益看起来更小"误导；
    CAGR 把收益折算到年化，跨时长可比。同时仍保留 sharpe 衰减（sharpe 本身已年化）。
    """
    warnings = []
    is_sh = float(is_metrics.get("sharpe") or 0)
    oos_sh = float(oos_metrics.get("sharpe") or 0)
    is_wr = float(is_metrics.get("win_rate") or 0)
    oos_wr = float(oos_metrics.get("win_rate") or 0)
    is_dd = abs(float(is_metrics.get("max_drawdown") or 0))
    oos_dd = abs(float(oos_metrics.get("max_drawdown") or 0))
    oos_n = int(oos_metrics.get("trade_count") or 0)

    if oos_n < min_oos_trades:
        warnings.append(f"OOS trade count low ({oos_n} < {min_oos_trades}) — estimate unstable")

    if is_sh > 0.3:
        decay = 1.0 - (oos_sh / max(is_sh, 1e-9))
        if decay > max_sharpe_decay:
            warnings.append(f"Sharpe decay {decay:.1%} > {max_sharpe_decay:.0%} (IS={is_sh:.2f} OOS={oos_sh:.2f})")

    # CAGR 衰减
    is_cagr = _cagr(float(is_metrics.get("total_return") or 0), is_days)
    oos_cagr = _cagr(float(oos_metrics.get("total_return") or 0), oos_days)
    if is_cagr > 0.02:
        decay_c = 1.0 - (oos_cagr / max(is_cagr, 1e-9))
        if decay_c > max_cagr_decay:
            warnings.append(
                f"CAGR decay {decay_c:.1%} > {max_cagr_decay:.0%} "
                f"(IS CAGR={is_cagr*100:.1f}%/y, OOS CAGR={oos_cagr*100:.1f}%/y)"
            )
        if oos_cagr < 0:
            warnings.append(f"OOS CAGR={oos_cagr*100:.2f}%/y is negative (strategy loses money OOS)")

    if is_wr > 0.45 and oos_wr < is_wr - 0.12:
        warnings.append(f"Win-rate drop IS={is_wr:.1%} → OOS={oos_wr:.1%}")

    if oos_dd > is_dd * 1.6 + 0.02:
        warnings.append(f"OOS drawdown worse: IS={is_dd:.1%} OOS={oos_dd:.1%}")

    score = 0.0
    if is_sh > 0.2:
        score += min(1.0, max(0, 1.0 - oos_sh / max(is_sh, 1e-9))) * 0.4
    if is_cagr > 0:
        score += min(1.0, max(0, 1.0 - oos_cagr / max(is_cagr, 1e-9))) * 0.3
    score += min(0.2, max(0, (is_wr - oos_wr))) * 0.5
    if oos_n < min_oos_trades:
        score += 0.15

    ok = len(warnings) == 0 and score < 0.45
    return {
        "ok": ok,
        "overfit_score": round(score, 3),
        "warnings": warnings,
        "advice": (
            "Looks reasonably stable across IS/OOS."
            if ok
            else "Possible overfit or regime shift — simplify params, lengthen data, or reduce look-ahead features."
        ),
    }
