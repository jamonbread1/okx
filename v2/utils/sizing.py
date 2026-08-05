# -*- coding: utf-8 -*-
"""v2 仓位计算 — Kelly 或 Volatility Targeting（P2-1）。

- kelly（默认）：Kelly 分数 × 硬上限。缺点是需要足够历史，1H 信号下单币种
  常凑不够样本而退化为恒 1.0。
- voltarget：size = (target_daily_vol * equity) / (atr_pct * price * ctVal)。
  不依赖历史样本，立即生效，是趋势跟踪反复验证过的做法。
"""
from __future__ import annotations


def calc_size(
    capital: float,
    leverage: float,
    pct: float,
    last: float,
    specs: dict,
    kelly_factor: float = 1.0,
    max_loss_pct: float = 0.01,
    sl_distance_pct: float = 0.01,
    sizing_mode: str = "kelly",
    target_daily_vol: float = 0.015,
    atr_pct: float = 0.0,
) -> float:
    """计算建仓张数。"""
    # 兼容 camelCase（OKX/multi_engine）与 snake_case
    ct_val = float(specs.get("ct_val") or specs.get("ctVal") or 0.01)
    lot_sz = float(specs.get("lot_sz") or specs.get("lotSz") or 0.01)
    min_sz = float(specs.get("min_sz") or specs.get("minSz") or 0.01)
    cash_reserve = 0.18

    if ct_val <= 0 or last <= 0 or capital <= 0:
        return 0.0

    usable = max(0.0, capital * (1 - cash_reserve))

    if sizing_mode == "voltarget":
        # 组合目标波动 / 单标的 ATR 波动 → 目标名义（用到的有效权益为 capital）
        if atr_pct and atr_pct > 1e-12:
            target_notional = max(0.0, (target_daily_vol * capital) / atr_pct)
        else:
            target_notional = usable * leverage * pct * min(1.0, kelly_factor)
    else:
        # Kelly 模式
        if kelly_factor <= 0:
            return 0.0
        target_notional = usable * leverage * pct * min(1.0, kelly_factor)

    # 硬上限: 单笔最大亏损不超过 max_loss_pct
    hard_cap = capital * max_loss_pct / max(sl_distance_pct, 1e-8)
    target_notional = min(target_notional, hard_cap)

    # 最大仓位
    max_pos = capital * 0.22 * leverage
    target_notional = min(target_notional, max_pos)

    raw = target_notional / (ct_val * last)
    size = int(raw / lot_sz) * lot_sz
    return size if size >= min_sz else 0.0
