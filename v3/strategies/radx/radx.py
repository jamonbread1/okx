# -*- coding: utf-8 -*-
# v3.strategies.radx.radx
"""RADX — Daily RSI(2) vs ADX(2) trend filter strategy.

参数（裸 key，无前缀）全部在同目录的 ``params.yaml`` 里维护。
``min_confidence`` 默认值回退到 ``config.yaml → strategy:`` 段的
``min_open_confidence``（跨策略共享字段）。

来源与定位
----------
公开社区中有人在 2025 年分享了一个 BTC/ETH 日线长线策略思路：

    Entry: close > SMA(50), close > EMA(7), RSI(2) > ADX(2)
    Exit : RSI(2) < ADX(2)

该思路声称在 BTCUSD/BTCEUR/ETHUSD 的 2012-2025 日线回测中优于买入持有、
回撤更低，但原帖也明确未计入手续费和滑点。因此本实现作为"研究策略"加入，
默认不放入全策略自动规划；建议先用 --only radx 单独验证。

适用：1D BTC/ETH 趋势环境；long-only。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v3.indicators import adx, atr, ema, rsi
from v3.strategies.base import Signal, StrategyBase


class RsiAdxTrend(StrategyBase):
    name = "radx"
    required_regime = "any"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # ---- 关键参数：默认值与 params.yaml 保持一致 ----
        self.sma_period = int(cfg.get("sma_period", 50))
        self.ema_period = int(cfg.get("ema_period", 7))
        self.rsi_period = int(cfg.get("rsi_period", 2))
        self.adx_period = int(cfg.get("adx_period", 2))
        self.position_pct = float(cfg.get("position_pct", 0.03))
        self.sl_atr_mult = float(cfg.get("sl_atr_mult", 3.0))
        # 用极远 TP，实际退出主要由 v3.engine 中的 RSI<ADX 条件控制。
        self.tp_r = float(cfg.get("tp_r", 999.0))
        # min_confidence 优先从 params 取，缺省时回退到 strategy.min_open_confidence
        self.min_conf = float(
            cfg.get("min_confidence", cfg.get("min_open_confidence", 0.55))
        )
        self.fee_rt = float(cfg.get("fee_rt", 0.0010))
        self.slippage_pct = float(cfg.get("slippage_pct", 0.0003))
        self.min_atr_pct = float(cfg.get("min_atr_pct", 0.002))
        self.max_atr_pct = float(cfg.get("max_atr_pct", 0.12))
        self.long_only = bool(cfg.get("long_only", True))

    def generate(
        self,
        df: pd.DataFrame,
        regime: str,
        last: float,
        capital: float,
        leverage: float,
        specs: dict,
        kelly_factor: float = 1.0,
        funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        need = max(self.sma_period, self.ema_period, self.rsi_period, self.adx_period, 14) + 5
        if df is None or len(df) < need:
            return None
        if self.long_only is False:
            # 公开规则是 long-only；当前实现先不扩展 short，避免改变策略含义。
            return None

        close = df["close"].astype(float)
        cur = float(close.iloc[-1])
        if not np.isfinite(cur) or cur <= 0:
            return None

        sma50 = close.rolling(self.sma_period).mean()
        ema7 = ema(close, self.ema_period)
        rsi2 = rsi(close, self.rsi_period)
        adx2 = adx(df, self.adx_period)
        atr14 = atr(df, 14)

        vals = [sma50.iloc[-1], ema7.iloc[-1], rsi2.iloc[-1], adx2.iloc[-1], atr14.iloc[-1]]
        if not all(np.isfinite(float(x)) for x in vals):
            return None

        cur_sma = float(sma50.iloc[-1])
        cur_ema = float(ema7.iloc[-1])
        cur_rsi = float(rsi2.iloc[-1])
        cur_adx = float(adx2.iloc[-1])
        cur_atr = float(atr14.iloc[-1])
        atr_pct = cur_atr / max(cur, 1e-12)
        if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
            return None

        if not (cur > cur_sma and cur > cur_ema and cur_rsi > cur_adx):
            return None
        if self._funding_blocked("long", funding_rate):
            return None

        sl_dist = max(cur_atr * self.sl_atr_mult, cur * 0.01)
        sl_pct = sl_dist / max(cur, 1e-12)
        size = self._calc_size(
            capital, leverage, self.position_pct, last, specs, kelly_factor,
            sl_distance_pct=sl_pct,
            atr_pct=atr_pct,
        )
        if size <= 0:
            return None

        # confidence 是规则评分，不是概率。
        trend_gap = max((cur - max(cur_sma, cur_ema)) / cur, 0.0)
        rsi_adx_gap = max((cur_rsi - cur_adx) / 100.0, 0.0)
        conf = self.min_conf
        conf += 0.15 * float(np.clip(trend_gap / 0.05, 0.0, 1.0))
        conf += 0.20 * float(np.clip(rsi_adx_gap / 0.25, 0.0, 1.0))
        conf = float(np.clip(conf, 0.0, 0.90))

        return Signal(
            action="open_long",
            direction="long",
            confidence=conf,
            strategy=self.name,
            regime=regime,
            stop_loss=cur - sl_dist,
            take_profit=cur + sl_dist * self.tp_r,
            rr_list=[self.tp_r],
            batch_ratios=[1.0],
            size=float(size),
            atr=cur_atr,
            reason=(
                f"[RADX] close>SMA{self.sma_period}/EMA{self.ema_period} "
                f"RSI{self.rsi_period}={cur_rsi:.1f}>ADX{self.adx_period}={cur_adx:.1f}"
            ),
        )
