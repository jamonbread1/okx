# -*- coding: utf-8 -*-
"""v2 技术指标库 — 纯 pandas/numpy 实现，无 talib 依赖。

参考来源：
  - je-suis-tm/quant-trading: BB pattern recognition, MACD, RSI, Dual Thrust
  - freqtrade/qtpylib: BB, volume, crossover 工具
  - pysystemtrade/robcarver17: EWMAC (指数加权移动平均交叉)
  - backtrader: ADX, ATR 标准实现
"""
from v2.indicators.core import (
    sma, ema, rsi, macd, atr, adx, bollinger_bands, bbw,
    donchian, stochastic, williams_r, obv, vwap,
    ewmac, crossover, crossunder, heikin_ashi,
    typical_price, median_price,
)

__all__ = [
    "sma", "ema", "rsi", "macd", "atr", "adx", "bollinger_bands", "bbw",
    "donchian", "stochastic", "williams_r", "obv", "vwap",
    "ewmac", "crossover", "crossunder", "heikin_ashi",
    "typical_price", "median_price",
]
