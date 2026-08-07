# -*- coding: utf-8 -*-
"""回测用资金分配（已剥离实盘 Universe 筛选 / OKX 客户端依赖）。"""
from __future__ import annotations

from typing import Dict, List, Optional


def allocate_capital(
    total: float,
    symbols: List[str],
    weights: Optional[Dict[str, float]] = None,
    meta: Optional[Dict[str, Dict]] = None,
    mode: str = "equal",
) -> Dict[str, float]:
    """
    资金分配：
    - equal: 等权
    - liq: 按 24h 成交额加权
    - liq_invvol: 流动性加权，并对高波动标的降权（若 meta 无 atr 则退化为 liq）
    单品种上限 35%，下限 8%（品种数>=3 时），避免过度集中
    """
    if not symbols:
        return {}
    n = len(symbols)
    if weights:
        raw = {s: max(1e-9, float(weights.get(s, 0.0))) for s in symbols}
    elif mode in ("liq", "liq_invvol") and meta:
        raw = {}
        for s in symbols:
            m = meta.get(s) or {}
            vol = max(1.0, float(m.get("vol_usdt_24h") or m.get("liq_score") or 1.0))
            w = vol ** 0.5
            atr_pct = float(m.get("atr_pct") or 0)
            if mode == "liq_invvol" and atr_pct > 0:
                w = w / max(0.5, min(2.0, atr_pct / 0.015))
            raw[s] = w
    else:
        raw = {s: 1.0 for s in symbols}

    ssum = sum(raw.values()) or 1.0
    alloc = {s: total * raw[s] / ssum for s in symbols}

    if n >= 3:
        cap = total * 0.35
        floor = total * 0.08
        for _ in range(5):
            overflow = 0.0
            active = []
            for s in symbols:
                if alloc[s] > cap:
                    overflow += alloc[s] - cap
                    alloc[s] = cap
                else:
                    active.append(s)
            if overflow <= 1e-6 or not active:
                break
            add = overflow / len(active)
            for s in active:
                alloc[s] += add
        need = [s for s in symbols if alloc[s] < floor]
        rich = [s for s in symbols if alloc[s] > floor * 1.2]
        for s in need:
            deficit = floor - alloc[s]
            if not rich or deficit <= 0:
                continue
            take = deficit / len(rich)
            for r in rich:
                give = min(take, alloc[r] - floor)
                alloc[r] -= give
                alloc[s] += give
    return alloc
