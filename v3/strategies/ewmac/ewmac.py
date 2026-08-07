# -*- coding: utf-8 -*-
# v3.strategies.ewmac.ewmac
"""EWMAC 策略 — 指数加权移动平均交叉趋势跟踪。

参数（裸 key，无前缀）全部在同目录的 ``params.yaml`` 里维护。

参考来源:
  - Rob Carver / pysystemtrade: ewmac.py
  - 核心思想: 快慢 EMA 差值作为趋势信号，归一化后跨品种可比

核心逻辑:
  1. EWMAC(fast, slow) 信号 > threshold → 做多
  2. EWMAC(fast, slow) 信号 < -threshold → 做空
  3. ATR 归一化 → 信号跨品种可比（ewmac() 内已按价格归一化）
  4. ADX 过滤 → 只在趋势市开仓
  5. 双 EWMAC (8/32 + 16/64) 共振确认

适用 regime: trend (趋势跟踪)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v3.indicators import ewmac, adx, atr
from v3.strategies.base import Signal, StrategyBase


class EwmacTrend(StrategyBase):
    name = "ewmac"
    required_regime = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # ---- 关键参数：默认值与 params.yaml 保持一致 ----
        self.fast_1 = int(cfg.get("fast_1", 8))
        self.slow_1 = int(cfg.get("slow_1", 32))
        self.fast_2 = int(cfg.get("fast_2", 16))
        self.slow_2 = int(cfg.get("slow_2", 64))
        self.signal_threshold = float(cfg.get("threshold", 0.005))
        self.min_adx = float(cfg.get("min_adx", 20.0))
        self.position_pct = float(cfg.get("position_pct", 0.05))
        self.sl_atr_mult = float(cfg.get("sl_atr_mult", 2.0))
        self.tp_atr_mult = float(cfg.get("tp_atr_mult", 3.0))
        self.slippage_pct = float(cfg.get("slippage_pct", 0.0003))

    def generate(
        self, df: pd.DataFrame, regime: str, last: float,
        capital: float, leverage: float, specs: dict,
        kelly_factor: float = 1.0, funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        # 暖机：EWMA span=64 需要约 2 倍窗口才相对稳定
        if len(df) < self.slow_2 * 2:
            return None

        close = df["close"]
        adx_val = adx(df, 14)
        atr_val = atr(df, 14)

        if len(adx_val) < 1 or len(atr_val) < 1:
            return None

        cur_adx = float(adx_val.iloc[-1])
        if not np.isfinite(cur_adx) or cur_adx < self.min_adx:
            return None

        # 双 EWMAC 信号
        ewmac_1 = ewmac(close, self.fast_1, self.slow_1)
        ewmac_2 = ewmac(close, self.fast_2, self.slow_2)

        if len(ewmac_1) < 1 or len(ewmac_2) < 1:
            return None

        sig_1 = float(ewmac_1.iloc[-1])
        sig_2 = float(ewmac_2.iloc[-1])

        cur_price = float(close.iloc[-1])
        cur_atr = float(atr_val.iloc[-1])

        # NaN 保护
        if not all(np.isfinite(x) for x in (sig_1, sig_2, cur_price, cur_atr)):
            return None

        # 方向判断：双信号共振，且慢信号也需达到阈值的一半（避免任何微小正值放行）
        reson_thresh = self.signal_threshold * 0.5
        direction = None
        reason = ""
        if sig_1 > self.signal_threshold and sig_2 > reson_thresh:
            direction = "long"
            reason = f"[EWMAC]L fast={sig_1:.4f} slow={sig_2:.4f} adx={cur_adx:.1f}"
        elif sig_1 < -self.signal_threshold and sig_2 < -reson_thresh:
            direction = "short"
            reason = f"[EWMAC]S fast={sig_1:.4f} slow={sig_2:.4f} adx={cur_adx:.1f}"

        if direction is None:
            return None

        # 资金费率过滤
        if self._funding_blocked(direction, funding_rate):
            return None

        # SL/TP
        sl_dist = max(cur_atr * self.sl_atr_mult, cur_price * 0.005)
        slip = cur_price * self.slippage_pct

        if direction == "long":
            stop = cur_price - sl_dist - slip
        else:
            stop = cur_price + sl_dist + slip

        # 仓位
        sl_pct = sl_dist / max(cur_price, 1e-12)
        size = self._calc_size(
            capital, leverage, self.position_pct, last, specs, kelly_factor,
            sl_distance_pct=sl_pct,
            atr_pct=cur_atr / max(cur_price, 1e-12),
        )
        if size <= 0:
            return None

        # 置信度
        conf = 0.30
        sig_strength = abs(sig_1) / max(self.signal_threshold, 1e-8)
        conf += 0.30 * float(np.clip(sig_strength / 3.0, 0, 1))
        conf += 0.20 * float(np.clip((cur_adx - self.min_adx) / 20.0, 0, 1))
        # 共振强度：慢信号越强加分越多（原恒加 0.20 的死代码）
        conf += 0.20 * float(np.clip(abs(sig_2) / max(self.signal_threshold, 1e-8) / 2.0, 0, 1))
        conf = min(1.0, conf)

        # 分批止盈（统一 RR 制：take_profit = 最后一批）
        rr_list = [1.0, 2.0, 3.0]
        batch_ratios = [0.30, 0.30, 0.40]
        tps = [
            (cur_price + sl_dist * rr) if direction == "long" else (cur_price - sl_dist * rr)
            for rr in rr_list
        ]

        return Signal(
            action="open_long" if direction == "long" else "open_short",
            direction=direction,
            confidence=conf,
            strategy=self.name,
            regime=regime,
            stop_loss=stop,
            take_profit=tps[-1],
            tp_batches=tps,
            batch_ratios=batch_ratios,
            size=size,
            atr=cur_atr,
            reason=reason,
            rr_list=rr_list,
        )
