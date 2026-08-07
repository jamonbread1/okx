# -*- coding: utf-8 -*-
# v3.strategies.mr.mr
"""MR 策略 — RSI + Bollinger Bands 均值回归。

参数（裸 key，无前缀）全部在同目录的 ``params.yaml`` 里维护。

参考来源:
  - freqtrade: RSIMeanReversionStrategy (BB + RSI + EMA 趋势过滤)
  - pyquantlab: Enhanced BB Mean Reversion with ADX and RSI
  - v1 MeanReversionAlgo

核心逻辑:
  1. RSI 进入超卖/超买区 (RSI<rsi_oversold / RSI>rsi_overbought)
  2. 价格触碰 BB 下/上轨
  3. ADX < max_adx (确认震荡/弱趋势，非趋势)
  4. EMA 趋势过滤 (以 ATR 为单位的偏离距离，避免逆强趋势)
  5. 出场: 由引擎层 PositionManager 处理（见 v3/engine.py）

适用 regime: chop / mixed (震荡区)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v3.indicators import bollinger_bands, rsi, atr, adx, ema
from v3.strategies.base import Signal, StrategyBase


class MeanReversion(StrategyBase):
    name = "mr"
    required_regime = "chop"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # ---- 关键参数：默认值与 params.yaml 保持一致 ----
        self.bb_period = int(cfg.get("bb_period", 20))
        self.rsi_period = int(cfg.get("rsi_period", 14))
        self.rsi_oversold = float(cfg.get("rsi_oversold", 30.0))
        self.rsi_overbought = float(cfg.get("rsi_overbought", 70.0))
        self.max_adx = float(cfg.get("max_adx", 25.0))
        self.ema_filter = int(cfg.get("ema_filter", 50))
        # EMA 偏离阈值（ATR 单位），替代原 ±5% 的过宽容忍带
        self.ema_atr_mult = float(cfg.get("ema_atr_mult", 2.5))
        self.position_pct = float(cfg.get("position_pct", 0.04))
        self.sl_atr_mult = float(cfg.get("sl_atr_mult", 2.0))
        self.tp_atr_mult = float(cfg.get("tp_atr_mult", 1.5))
        # 净期望 R 数阈值（见 StrategyBase._expected_r）
        self.min_expectancy = float(cfg.get("min_expectancy", 0.05))
        self.slippage_pct = float(cfg.get("slippage_pct", 0.0003))
        self.fee_rt = float(cfg.get("fee_rt", 0.0007))

    def generate(
        self, df: pd.DataFrame, regime: str, last: float,
        capital: float, leverage: float, specs: dict,
        kelly_factor: float = 1.0, funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        if len(df) < max(self.bb_period, self.ema_filter) + 5:
            return None

        close = df["close"]
        low, mid, upper = bollinger_bands(close, self.bb_period, 2.0)
        rsi_val = rsi(close, self.rsi_period)
        atr_val = atr(df, 14)
        adx_val = adx(df, 14)
        ema_val = ema(close, self.ema_filter)

        if len(rsi_val) < 1 or len(adx_val) < 1:
            return None

        cur_rsi = float(rsi_val.iloc[-1])
        cur_adx = float(adx_val.iloc[-1])
        cur_price = float(close.iloc[-1])
        cur_atr = float(atr_val.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(low.iloc[-1])
        m = float(mid.iloc[-1])
        cur_ema = float(ema_val.iloc[-1])

        # NaN 保护
        if not all(np.isfinite(x) for x in (cur_rsi, cur_adx, cur_price, cur_atr, u, l, m, cur_ema)):
            return None

        # ADX 区间过滤
        if cur_adx >= self.max_adx:
            return None

        # 方向判断
        direction = None
        reason = ""
        if cur_price <= l and cur_rsi < self.rsi_oversold:
            # EMA 趋势过滤：价格偏离 EMA 距离（ATR 单位）不能过大，避免强趋势下接飞刀
            if abs(cur_price - cur_ema) < self.ema_atr_mult * cur_atr:
                direction = "long"
                reason = f"[MR]L BB下轨+RSI超卖 rsi={cur_rsi:.1f} adx={cur_adx:.1f}"
        elif cur_price >= u and cur_rsi > self.rsi_overbought:
            if abs(cur_price - cur_ema) < self.ema_atr_mult * cur_atr:
                direction = "short"
                reason = f"[MR]S BB上轨+RSI超买 rsi={cur_rsi:.1f} adx={cur_adx:.1f}"

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
        exp_r = self._expected_r(sl_pct, tp_pct, 0.62, self.fee_rt, self.slippage_pct)
        if exp_r < self.min_expectancy:
            return None

        # 仓位
        size = self._calc_size(
            capital, leverage, self.position_pct, last, specs, kelly_factor,
            sl_distance_pct=sl_pct,
            atr_pct=cur_atr / max(cur_price, 1e-12),
        )
        if size <= 0:
            return None

        # 置信度
        bb_width = max(u - l, 1e-8)
        conf = 0.35
        if direction == "long":
            conf += 0.30 * float(np.clip((self.rsi_oversold - cur_rsi) / self.rsi_oversold, 0, 1))
        else:
            conf += 0.30 * float(np.clip((cur_rsi - self.rsi_overbought) / (100 - self.rsi_overbought), 0, 1))
        conf += 0.20 * float(np.clip((self.max_adx - cur_adx) / max(self.max_adx, 1), 0, 1))
        conf += 0.15 * float(np.clip(abs(cur_price - m) / bb_width, 0, 1))
        conf = min(1.0, conf)

        # 分批止盈（统一 RR 制：take_profit = 最后一批）
        rr_list = [0.75, 1.5]
        batch_ratios = [0.50, 0.50]
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
