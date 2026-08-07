# -*- coding: utf-8 -*-
# v3.strategies.vol.vol
"""VOL 策略 — Bollinger Bands Squeeze Breakout。

参数（裸 key，无前缀）全部在同目录的 ``params.yaml`` 里维护。
``generate()`` 只读 ``self.xxx``，不再调用 ``cfg.get``。

参考来源:
  - freqtrade: BB squeeze + breakout + RSI confirmation
  - je-suis-tm/quant-trading: BB pattern recognition (bottom W / top M)
  - v1 VolatilityBreakoutAlgo

核心逻辑:
  1. BBW squeeze 检测 (BBW < threshold，取前 N 根，当前根开始扩张)
  2. 价格突破上/下轨 + 放量确认
  3. RSI 中性区过滤 (避免极端 RSI 下的假突破)
  4. HTF 趋势方向过滤（由引擎层 HtfFilter 统一处理，策略内不重复实现）

适用 regime: any（squeeze 本身即盘整末端的时序信号，突破扩张标志趋势启动）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v3.indicators import bollinger_bands, bbw, rsi, atr
from v3.strategies.base import Signal, StrategyBase


class VolBreakout(StrategyBase):
    name = "vol"
    required_regime = "any"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # ---- 关键参数：默认值与 params.yaml 保持一致 ----
        self.bb_period = int(cfg.get("bb_period", 20))
        self.bbw_squeeze = float(cfg.get("bbw_squeeze", 0.035))
        self.vol_mult = float(cfg.get("volume_mult", 1.3))
        self.rsi_low = float(cfg.get("rsi_low", 35.0))
        self.rsi_high = float(cfg.get("rsi_high", 65.0))
        self.breakout_buffer = float(cfg.get("breakout_buffer", 0.0003))
        self.position_pct = float(cfg.get("position_pct", 0.05))
        self.sl_atr_mult = float(cfg.get("sl_atr_mult", 1.8))
        self.tp_atr_mult = float(cfg.get("tp_atr_mult", 3.5))
        # 净期望 R 数阈值（见 StrategyBase._expected_r）
        self.min_expectancy = float(cfg.get("min_expectancy", 0.1))
        self.slippage_pct = float(cfg.get("slippage_pct", 0.00035))
        self.fee_rt = float(cfg.get("fee_rt", 0.0007))
        self.short_penalty = float(cfg.get("short_penalty", 0.7))
        # squeeze 需持续根数（当前根之前）
        self.squeeze_hold = int(cfg.get("squeeze_hold", 3))

    def generate(
        self, df: pd.DataFrame, regime: str, last: float,
        capital: float, leverage: float, specs: dict,
        kelly_factor: float = 1.0, funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        if len(df) < self.bb_period + 5:
            return None

        close = df["close"]
        low, mid, upper = bollinger_bands(close, self.bb_period, 2.0)
        bbw_val = bbw(close, self.bb_period, 2.0)
        rsi_val = rsi(close, 14)
        atr_val = atr(df, 14)

        if len(bbw_val) < self.squeeze_hold + 2 or len(rsi_val) < 1:
            return None

        cur_bbw = float(bbw_val.iloc[-1])
        prev_bbw = float(bbw_val.iloc[-2])
        cur_rsi = float(rsi_val.iloc[-1])
        cur_price = float(close.iloc[-1])
        cur_atr = float(atr_val.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(low.iloc[-1])

        # NaN 保护：数值非法直接退出
        if not all(np.isfinite(x) for x in (cur_bbw, cur_rsi, cur_price, cur_atr, u, l)):
            return None

        # Squeeze 时序：前 N 根处于 squeeze，当前根开始扩张
        #   （避免「突破那根已经把 BBW 撑开」导致 squeeze 与突破自相矛盾）
        squeezed = (bbw_val.iloc[-self.squeeze_hold - 1:-1] < self.bbw_squeeze).all()
        expanding = cur_bbw > prev_bbw
        if not (squeezed and expanding):
            return None

        # 放量确认
        vol_ma = float(df["vol"].iloc[-20:].mean())
        cur_vol = float(df["vol"].iloc[-1])
        vol_ratio = cur_vol / max(vol_ma, 1e-8)
        if vol_ratio < self.vol_mult:
            return None

        # RSI 中性区
        if not (self.rsi_low <= cur_rsi <= self.rsi_high):
            return None

        # 方向判断
        direction = None
        reason = ""
        if cur_price > u * (1 + self.breakout_buffer):
            direction = "long"
            reason = f"[VOL]L BB squeeze breakout bbw={cur_bbw:.4f} vol={vol_ratio:.1f}x rsi={cur_rsi:.1f}"
        elif cur_price < l * (1 - self.breakout_buffer):
            if vol_ratio >= self.vol_mult + 0.3:  # short 需更高量
                direction = "short"
                reason = f"[VOL]S BB squeeze breakdown bbw={cur_bbw:.4f} vol={vol_ratio:.1f}x rsi={cur_rsi:.1f}"

        if direction is None:
            return None

        # 资金费率过滤
        if self._funding_blocked(direction, funding_rate):
            return None

        # SL/TP
        sl_dist = max(cur_atr * self.sl_atr_mult, cur_price * 0.005)
        tp_dist = cur_atr * self.tp_atr_mult
        slip = cur_price * self.slippage_pct

        if direction == "long":
            stop = cur_price - sl_dist - slip
        else:
            stop = cur_price + sl_dist + slip

        # 净期望 R 过滤（R 倍数制，阈值与参数组合无关）
        sl_pct = sl_dist / max(cur_price, 1e-12)
        tp_pct = tp_dist / max(cur_price, 1e-12)
        exp_r = self._expected_r(sl_pct, tp_pct, 0.58, self.fee_rt, self.slippage_pct)
        if exp_r < self.min_expectancy:
            return None

        # 仓位
        pos_pct = self.position_pct * (self.short_penalty if direction == "short" else 1.0)
        size = self._calc_size(
            capital, leverage, pos_pct, last, specs, kelly_factor,
            sl_distance_pct=sl_pct,
            atr_pct=cur_atr / max(cur_price, 1e-12),
        )
        if size <= 0:
            return None

        # 置信度
        conf = 0.30
        conf += 0.30 * float(np.clip((self.bbw_squeeze - cur_bbw) / self.bbw_squeeze, 0, 1))
        conf += 0.25 * float(np.clip((vol_ratio - 1.0) / 1.5, 0, 1))
        # RSI 距离中性区边缘的余量（修正原常量项）
        conf += 0.15 * float(np.clip(1.0 - abs(cur_rsi - 50.0) / 15.0, 0, 1))
        conf = min(1.0, conf)

        # 分批止盈（统一 RR 制：take_profit = 最后一批）
        rr_list = [1.0, 1.5, 2.5]
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
