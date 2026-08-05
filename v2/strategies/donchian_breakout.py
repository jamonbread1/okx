# -*- coding: utf-8 -*-
# v2.strategies.donchian_breakout
"""DON 策略 — 唐奇安通道突破（保留 v1 核心逻辑，简化参数）。

参考来源:
  - v1 MomentumAlgo (唐奇安突破)
  - Dual Thrust (je-suis-tm/quant-trading): 开盘区间突破

核心逻辑:
  1. 价格突破前 N 根 K 线的最高价 → 做多
  2. 价格跌破前 N 根 K 线的最低价 → 做空
  3. 放量确认 + ADX 过滤
  4. 影线过滤 (避免上影线假突破)

适用 regime: trend (趋势跟踪)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v2.indicators import adx, atr
from v2.strategies.base import Signal, StrategyBase


class DonchianBreakout(StrategyBase):
    name = "don"
    required_regime = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.period = int(cfg.get("don_period", 28))
        self.min_adx = float(cfg.get("don_min_adx", 25.0))
        self.vol_ratio = float(cfg.get("don_vol_ratio", 1.3))
        self.max_wick_ratio = float(cfg.get("don_max_wick_ratio", 2.0))
        self.position_pct = float(cfg.get("don_position_pct", 0.04))
        self.sl_atr_mult = float(cfg.get("don_sl_atr_mult", 1.8))
        self.tp_atr_mult = float(cfg.get("don_tp_atr_mult", 3.5))
        self.slippage_pct = float(cfg.get("don_slippage_pct", 0.0004))
        self.fee_rt = float(cfg.get("don_fee_rt", 0.0007))
        self.short_penalty = float(cfg.get("don_short_penalty", 0.7))

        # EWM 指标计算窗口（Wilder alpha=1/14 时 250 根后权重 < 1e-7，结果与全量一致）
        self.calc_window = int(cfg.get("don_calc_window", 250))

    def generate(
        self, df: pd.DataFrame, regime: str, last: float,
        capital: float, leverage: float, specs: dict,
        kelly_factor: float = 1.0, funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        if len(df) < self.period + 10:
            return None

        # --- 1) 突破检测（最廉价过滤器，先做；绝大多数 bar 在此提前退出）---
        closes = df["close"].to_numpy()
        cur_price = float(closes[-1])
        prev_close = float(closes[-2])

        # 等价于 donchian(df, period) 的最后一个值：前 period 根（不含当前）的极值
        hi = float(df["high"].to_numpy()[-self.period - 1:-1].max())
        lo = float(df["low"].to_numpy()[-self.period - 1:-1].min())

        broke_up = prev_close <= hi and cur_price > hi
        broke_down = prev_close >= lo and cur_price < lo

        if not (broke_up or broke_down):
            return None

        # --- 2) 放量确认 ---
        vols = df["vol"].to_numpy()
        vol_ma = float(vols[-20:].mean())
        vol_ratio = float(vols[-1]) / max(vol_ma, 1e-8)

        if vol_ratio < self.vol_ratio:
            return None

        direction = None
        if broke_up:
            direction = "long"
        elif broke_down and vol_ratio >= self.vol_ratio + 0.3:  # short 需更高量
            direction = "short"

        if direction is None:
            return None

        # --- 3) 影线过滤 ---
        row = df.iloc[-1]
        body = abs(float(row["close"]) - float(row["open"])) + 1e-12
        if direction == "long":
            upper_wick = float(row["high"]) - max(float(row["close"]), float(row["open"]))
            if upper_wick / body > self.max_wick_ratio:
                return None
        else:
            lower_wick = min(float(row["close"]), float(row["open"])) - float(row["low"])
            if lower_wick / body > self.max_wick_ratio:
                return None

        # --- 4) 资金费率过滤 ---
        if self._funding_blocked(direction, funding_rate):
            return None

        # --- 5) ADX / ATR（最昂贵，放最后；只对尾部窗口计算）---
        win = min(len(df), self.period + self.calc_window)
        tail = df.iloc[-win:]

        cur_adx = float(adx(tail, 14).iloc[-1])
        if not np.isfinite(cur_adx) or cur_adx < self.min_adx:
            return None

        cur_atr = float(atr(tail, 14).iloc[-1])

        if direction == "long":
            reason = f"[DON]L hi={hi:.4f} adx={cur_adx:.1f} volR={vol_ratio:.2f}"
        else:
            reason = f"[DON]S lo={lo:.4f} adx={cur_adx:.1f} volR={vol_ratio:.2f}"

        # --- SL/TP ---
        sl_dist = max(cur_atr * self.sl_atr_mult, cur_price * 0.005)
        slip = cur_price * self.slippage_pct

        if direction == "long":
            stop = cur_price - sl_dist - slip
        else:
            stop = cur_price + sl_dist + slip

        # --- 仓位 ---
        pos_pct = self.position_pct * (self.short_penalty if direction == "short" else 1.0)
        sl_pct = sl_dist / max(cur_price, 1e-12)
        size = self._calc_size(
            capital, leverage, pos_pct, last, specs, kelly_factor,
            sl_distance_pct=sl_pct,
            atr_pct=cur_atr / max(cur_price, 1e-12),
        )
        if size <= 0:
            return None

        # --- 置信度 ---
        conf = 0.30
        conf += 0.30 * float(np.clip((cur_adx - self.min_adx) / 20.0, 0, 1))
        conf += 0.25 * float(np.clip((vol_ratio - 1.0) / 1.0, 0, 1))
        conf += 0.15 * float(np.clip((cur_atr / cur_price) / 0.005, 0, 1))
        conf = min(1.0, conf)

        # --- 分批止盈 ---
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
