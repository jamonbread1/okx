# -*- coding: utf-8 -*-
# v3.strategies.macd.macd
"""MACD 策略 — MACD 交叉 + RSI 确认 + ADX 趋势过滤。

参数（裸 key，无前缀）全部在同目录的 ``params.yaml`` 里维护。

参考来源:
  - je-suis-tm/quant-trading: MACD Oscillator backtest
  - freqtrade: MACD + RSI + ADX 组合
  - molbal/jita3: MACD + RSI momentum scalping

核心逻辑:
  1. MACD line 上穿 signal line → 做多信号
  2. MACD line 下穿 signal line → 做空信号
  3. RSI 确认: 做多时 RSI < rsi_low(65), 做空时 RSI > rsi_high(35)
     （注意 rsi_low 是做多的上限、rsi_high 是做空的下限，命名来自旧版）
  4. ADX 过滤: ADX > min_adx 确认趋势存在
  5. Histogram 动量确认: histogram 在扩张（cur_hist > prev_hist），而非仅同号

适用 regime: trend (趋势跟踪)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v3.indicators import macd, rsi, adx, atr
from v3.strategies.base import Signal, StrategyBase


class MacdDivergence(StrategyBase):
    name = "macd"
    required_regime = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # ---- 关键参数：默认值与 params.yaml 保持一致 ----
        self.fast = int(cfg.get("fast", 12))
        self.slow = int(cfg.get("slow", 26))
        self.signal = int(cfg.get("signal", 9))
        # 注意：rsi_low 是做多的上限、rsi_high 是做空的下限，命名来自旧版
        self.rsi_low = float(cfg.get("rsi_low", 65.0))
        self.rsi_high = float(cfg.get("rsi_high", 35.0))
        self.min_adx = float(cfg.get("min_adx", 20.0))
        self.position_pct = float(cfg.get("position_pct", 0.04))
        self.sl_atr_mult = float(cfg.get("sl_atr_mult", 1.8))
        self.tp_atr_mult = float(cfg.get("tp_atr_mult", 3.0))
        self.slippage_pct = float(cfg.get("slippage_pct", 0.0003))

    def generate(
        self, df: pd.DataFrame, regime: str, last: float,
        capital: float, leverage: float, specs: dict,
        kelly_factor: float = 1.0, funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        if len(df) < self.slow + self.signal + 10:
            return None

        close = df["close"]
        macd_line, signal_line, hist = macd(close, self.fast, self.slow, self.signal)
        rsi_val = rsi(close, 14)
        adx_val = adx(df, 14)
        atr_val = atr(df, 14)

        if len(macd_line) < 3 or len(rsi_val) < 1:
            return None

        cur_adx = float(adx_val.iloc[-1])
        if not np.isfinite(cur_adx) or cur_adx < self.min_adx:
            return None

        cur_rsi = float(rsi_val.iloc[-1])
        cur_macd = float(macd_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        cur_signal = float(signal_line.iloc[-1])
        prev_signal = float(signal_line.iloc[-2])
        cur_hist = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2])
        cur_price = float(close.iloc[-1])
        cur_atr = float(atr_val.iloc[-1])

        # NaN 保护
        if not all(np.isfinite(x) for x in (cur_rsi, cur_macd, cur_signal,
                                            cur_hist, prev_hist, cur_price, cur_atr)):
            return None

        # MACD 交叉
        direction = None
        reason = ""
        if prev_macd <= prev_signal and cur_macd > cur_signal:
            # RSI 确认 + histogram 扩张（动量加速），而非仅同号
            if cur_rsi < self.rsi_low and cur_hist > prev_hist:
                direction = "long"
                reason = f"[MACD]L MACD上穿 rsi={cur_rsi:.1f} adx={cur_adx:.1f} hist={cur_hist:.4f}"
        elif prev_macd >= prev_signal and cur_macd < cur_signal:
            if cur_rsi > self.rsi_high and cur_hist < prev_hist:
                direction = "short"
                reason = f"[MACD]S MACD下穿 rsi={cur_rsi:.1f} adx={cur_adx:.1f} hist={cur_hist:.4f}"

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
        # histogram 动量以 ATR 归一化（原 abs(hist)/abs(macd) 在 macd≈0 时被噪声打满）
        conf += 0.25 * float(np.clip(abs(cur_hist) / max(cur_atr, 1e-8) / 2.0, 0, 1))
        conf += 0.25 * float(np.clip((cur_adx - self.min_adx) / 20.0, 0, 1))
        if direction == "long":
            conf += 0.20 * float(np.clip((self.rsi_low - cur_rsi) / self.rsi_low, 0, 1))
        else:
            conf += 0.20 * float(np.clip((cur_rsi - self.rsi_high) / (100 - self.rsi_high), 0, 1))
        conf = min(1.0, conf)

        # 分批止盈（统一 RR 制：take_profit = 最后一批）
        rr_list = [1.0, 2.0]
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
