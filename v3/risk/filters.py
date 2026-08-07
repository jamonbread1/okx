# -*- coding: utf-8 -*-
"""过滤器 — HTF 趋势过滤 + 冲突过滤。"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from v3.indicators import ema


class HtfFilter:
    """高周期趋势过滤。

    参考 v1 HTF trend filter，但简化为只返回方向。
    """
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("htf_enabled", True))
        self.bar = str(cfg.get("htf_bar", "4H"))
        self.fast = int(cfg.get("htf_ema_fast", 20))
        self.slow = int(cfg.get("htf_ema_slow", 50))
        self.min_gap = float(cfg.get("htf_min_gap", 0.001))

    def bias(self, df_htf: Optional[pd.DataFrame]) -> str:
        """返回 'long' / 'short' / 'neutral'。"""
        if not self.enabled or df_htf is None or len(df_htf) < self.slow:
            return "neutral"
        close = df_htf["close"].astype(float)
        e_fast = float(ema(close, self.fast).iloc[-1])
        e_slow = float(ema(close, self.slow).iloc[-1])
        if e_slow <= 0:
            return "neutral"
        gap = (e_fast - e_slow) / e_slow
        if gap > self.min_gap:
            return "long"
        if gap < -self.min_gap:
            return "short"
        return "neutral"


class ConflictFilter:
    """跨策略冲突过滤 — Constitution §3。

    参考 v1 _conflict_check。
    """
    def __init__(self, cfg: dict):
        self.window_min = float(cfg.get("conflict_window_min", 60))

    def check(
        self,
        new_direction: str,
        new_strategy: str,
        new_confidence: float,
        recent_signals: list,
        now_ts: float,
    ) -> Optional[str]:
        """检查新信号是否与近期信号冲突。返回 None 或冲突原因。"""
        cutoff = now_ts - self.window_min * 60
        for prev in recent_signals:
            if prev["ts"] < cutoff:
                continue
            if prev["direction"] == new_direction:
                continue
            if prev["strategy"] == new_strategy:
                continue
            if new_confidence <= prev["confidence"]:
                return (
                    f"Conflict: {new_strategy} {new_direction} "
                    f"(conf={new_confidence:.2f}) vs {prev['strategy']} {prev['direction']}"
                )
        return None
