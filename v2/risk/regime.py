# -*- coding: utf-8 -*-
"""v2 Regime 检测 — ADX + BBW 双维度。

参考 pysystemtrade: regime 通过 ADX 判断趋势/震荡。
v2 改进: 增加 BBW 作为辅助维度 (squeeze → 适合 VOL; expansion → 适合 trend)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2.indicators import adx, bbw, atr


@dataclass
class RegimeResult:
    regime: str        # trend / chop / mixed
    adx: float
    atr_pct: float
    bbw: float
    is_squeeze: bool

    @property
    def allow_trend(self) -> bool:
        return self.regime == "trend"

    @property
    def allow_mr(self) -> bool:
        return self.regime in ("chop", "mixed")

    @property
    def allow_vol(self) -> bool:
        return self.is_squeeze


class RegimeDetector:
    """Regime 检测（ADX + BBW），带滞后带（P2-4）。

    双阈值：ADX >= adx_trend 才进入 trend；已处于 trend 时，ADX 回落到
    adx_exit 以下才退出，避免 ADX 在阈值附近抖动导致 trend/chop 频繁切换。
    """

    def __init__(self, cfg: dict):
        self.adx_trend = float(cfg.get("regime_adx_trend", 25.0))   # 进 trend
        self.adx_exit = float(cfg.get("regime_adx_exit", 20.0))     # 出 trend（< 进）
        self.adx_chop = float(cfg.get("regime_adx_chop", 20.0))
        self.squeeze_threshold = float(cfg.get("regime_squeeze_threshold", 0.035))
        self._prev: str = "chop"

    def _classify(self, adx_val: float) -> str:
        # 滞后带：一旦进入 trend，需 ADX 跌破 adx_exit 才离开
        if self._prev == "trend":
            if adx_val >= self.adx_exit:
                return "trend"
        else:
            if adx_val >= self.adx_trend:
                return "trend"
        if adx_val <= self.adx_chop:
            return "chop"
        return "mixed"

    def detect(self, df: pd.DataFrame, last: float) -> RegimeResult:
        adx_val = float(adx(df, 14).iloc[-1]) if len(df) >= 60 else 20.0
        if not np.isfinite(adx_val):
            adx_val = 20.0

        atr_val = float(atr(df, 14).iloc[-1]) if len(df) >= 30 else 0.0
        atr_pct = atr_val / max(last, 1e-12) if last > 0 else 0.0

        bbw_val = 1.0
        if len(df) >= 40:
            bbw_s = bbw(df["close"], 20, 2.0)
            if len(bbw_s) >= 1:
                bbw_val = float(bbw_s.iloc[-1])

        regime = self._classify(adx_val)
        self._prev = regime

        is_squeeze = bbw_val < self.squeeze_threshold

        return RegimeResult(
            regime=regime,
            adx=round(adx_val, 2),
            atr_pct=round(atr_pct, 6),
            bbw=round(bbw_val, 5),
            is_squeeze=is_squeeze,
        )
