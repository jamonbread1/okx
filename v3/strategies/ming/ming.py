# -*- coding: utf-8 -*-
"""MING — Donchian + MACD 1D 双触发共振策略。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v3.indicators import adx, atr, macd
from v3.strategies.base import Signal, StrategyBase


class MingBreakout(StrategyBase):
    """Donchian + MACD 双触发共振。"""

    name = "ming"
    required_regime = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # don 部分
        self.don_period = int(cfg.get("don_period", 20))
        self.don_min_adx = float(cfg.get("don_min_adx", 20.0))
        self.don_vol_ratio = float(cfg.get("don_vol_ratio", 1.0))
        self.don_max_wick_ratio = float(cfg.get("don_max_wick_ratio", 2.0))
        self.don_min_break_hold_bars = int(cfg.get("don_min_break_hold_bars", 1))
        # macd 部分
        self.macd_fast = int(cfg.get("macd_fast", 12))
        self.macd_slow = int(cfg.get("macd_slow", 26))
        self.macd_signal = int(cfg.get("macd_signal", 9))
        self.macd_min_adx = float(cfg.get("macd_min_adx", 20.0))
        # 加权评分
        self.w_don = float(cfg.get("w_don", 0.60))
        self.w_macd = float(cfg.get("w_macd", 0.40))
        if abs(self.w_don + self.w_macd - 1.0) > 1e-6:
            s = self.w_don + self.w_macd
            self.w_don = self.w_don / s
            self.w_macd = self.w_macd / s
        # 入场门控
        self.require_dual_trigger = bool(cfg.get("require_dual_trigger", True))
        self.min_entry_conf = float(cfg.get("min_entry_conf", 0.55))
        # 风险预算（引擎用）
        self.risk_pct = float(cfg.get("risk_pct", 0.005))
        # 初始止损参数
        self.sl_k_vol_low = float(cfg.get("sl_k_vol_low", 1.4))
        self.sl_k_vol_mid = float(cfg.get("sl_k_vol_mid", 1.8))
        self.sl_k_vol_high = float(cfg.get("sl_k_vol_high", 2.3))
        self.sl_struct_lookback = int(cfg.get("sl_struct_lookback", 8))
        self.sl_struct_buffer_atr = float(cfg.get("sl_struct_buffer_atr", 0.2))
        self.sl_floor_pct = float(cfg.get("sl_floor_pct", 0.005))
        # 波动率自适应
        self.natr_lookback = int(cfg.get("natr_lookback", 120))
        self.natr_pause_mult = float(cfg.get("natr_pause_mult", 1.5))
        self.natr_shrink_pct = float(cfg.get("natr_shrink_pct", 0.75))
        self.natr_block_pct = float(cfg.get("natr_block_pct", 0.95))
        # 趋势过滤
        self.trend_ma_period = int(cfg.get("trend_ma_period", 50))
        self.trend_min_adx = float(cfg.get("trend_min_adx", 18.0))
        # 1W 高周期母趋势过滤
        self.htf_weekly_ma_period = int(cfg.get("htf_weekly_ma_period", 20))
        self.htf_weekly_min_gap = float(cfg.get("htf_weekly_min_gap", 0.0))
        # 退出参数（引擎读）
        self.panic_gap_atr = float(cfg.get("panic_gap_atr", 1.0))
        self.timeout_mfe_bars = int(cfg.get("timeout_mfe_bars", 3))
        self.timeout_mfe_r = float(cfg.get("timeout_mfe_r", 0.5))
        self.timeout_breakout_bars = int(cfg.get("timeout_breakout_bars", 5))
        self.timeout_max_bars = int(cfg.get("timeout_max_bars", 10))
        # 动态 max_bars：高 NATR 段减半
        self.max_bars_vol_halve_natr = float(cfg.get("max_bars_vol_halve_natr", 0.05))
        self.max_bars_vol_halve_factor = float(cfg.get("max_bars_vol_halve_factor", 0.5))
        # +1R / +1.5R / +2R 管理
        self.r1_be_buffer_atr = float(cfg.get("r1_be_buffer_atr", 0.2))
        self.r15_partial_pct = float(cfg.get("r15_partial_pct", 0.30))
        self.r2_chandelier_N = int(cfg.get("r2_chandelier_N", 10))
        self.r2_chandelier_k = float(cfg.get("r2_chandelier_k", 3.0))
        # 成本
        self.slippage_pct = float(cfg.get("slippage_pct", 0.0004))
        self.fee_rt = float(cfg.get("fee_rt", 0.0007))
        # ATR 周期
        self.atr_period = int(cfg.get("atr_period", 14))
        self.natr_ema_period = int(cfg.get("natr_ema_period", 20))

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
        if df is None or len(df) < max(self.don_period + 20, self.macd_slow + self.macd_signal + 10):
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        vol = df["vol"].astype(float)

        cur_price = float(close.iloc[-1])
        if not np.isfinite(cur_price) or cur_price <= 0:
            return None

        atr_val = float(atr(df, self.atr_period).iloc[-1])
        adx_val = float(adx(df, self.atr_period).iloc[-1])
        if not (np.isfinite(atr_val) and np.isfinite(adx_val)) or atr_val <= 0:
            return None
        natr = atr_val / max(cur_price, 1e-12)

        natr_series = (atr(df, self.atr_period) / close.replace(0.0, np.nan)).dropna()
        natr_pct = 0.5
        if len(natr_series) >= max(self.natr_lookback, 30):
            history = natr_series.iloc[-self.natr_lookback:]
            natr_pct = float((history <= natr).mean())

        # NATR 急性扩张 → 暂停
        if len(natr_series) >= self.natr_ema_period:
            ema_natr = float(natr_series.ewm(span=self.natr_ema_period, adjust=False).mean().iloc[-1])
            if ema_natr > 0 and natr / ema_natr > self.natr_pause_mult:
                return None

        # NATR 百分位熔断
        if natr_pct >= self.natr_block_pct:
            return None

        # 趋势过滤
        if len(close) < self.trend_ma_period:
            return None
        ma50 = float(close.rolling(self.trend_ma_period).mean().iloc[-1])
        if not np.isfinite(ma50) or ma50 <= 0:
            return None
        if adx_val < self.trend_min_adx:
            return None

        # 1W 母趋势过滤
        htf_long_ok = True
        htf_short_ok = True
        if self.htf_weekly_ma_period > 0 and "ts" in df.columns and len(df) >= self.htf_weekly_ma_period * 5:
            try:
                ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                tmp = pd.DataFrame({"close": close.values, "ts": ts.values})
                tmp = tmp.dropna(subset=["ts", "close"])
                if len(tmp) >= self.htf_weekly_ma_period:
                    weekly = (
                        tmp.set_index("ts")["close"]
                        .resample("W-FRI", label="right", closed="right")
                        .last()
                        .dropna()
                    )
                    if len(weekly) >= self.htf_weekly_ma_period:
                        wk_ma = float(weekly.rolling(self.htf_weekly_ma_period).mean().iloc[-1])
                        if np.isfinite(wk_ma) and wk_ma > 0:
                            htf_long_ok = (cur_price >= wk_ma * (1 - self.htf_weekly_min_gap))
                            htf_short_ok = (cur_price <= wk_ma * (1 + self.htf_weekly_min_gap))
            except Exception:
                htf_long_ok = htf_short_ok = True

        # don 触发
        don_score, don_long, don_short, don_breakout_high, don_breakout_low = (
            self._don_signal(open_, close, high, low, vol, atr_val, adx_val, cur_price)
        )
        if don_score <= 0:
            return None
        if don_long and cur_price < ma50:
            return None
        if don_short and cur_price > ma50:
            return None
        if don_long and not htf_long_ok:
            return None
        if don_short and not htf_short_ok:
            return None

        # macd 触发
        macd_score, macd_long, macd_short = self._macd_signal(close, adx_val, atr_val)
        if macd_score <= 0:
            return None

        # 硬双触发 AND
        if self.require_dual_trigger:
            direction_conflict = (don_long and macd_short) or (don_short and macd_long)
            if direction_conflict:
                return None
            long_trigger = don_long and macd_long
            short_trigger = don_short and macd_short
            if not (long_trigger or short_trigger):
                return None
        else:
            long_trigger = don_long or macd_long
            short_trigger = don_short or macd_short
            if not (long_trigger or short_trigger):
                return None

        direction = "long" if long_trigger else "short"

        weighted = don_score * self.w_don + macd_score * self.w_macd
        if weighted < self.min_entry_conf:
            return None

        # 初始止损
        sl_dist = self._initial_sl_distance(
            direction=direction,
            cur_price=cur_price,
            atr_val=atr_val,
            natr_pct=natr_pct,
            high=high, low=low,
        )
        if sl_dist <= 0:
            return None
        R = sl_dist
        if direction == "long":
            stop = cur_price - R
        else:
            stop = cur_price + R

        # 仓位：Q = E * r / (R + 滑点 + 费用)
        #   R = 价格距离（USD per 1 unit underlying）
        #   1 张 BTC 涨 R 美元 = ctVal * R
        #   单笔风险 = size * ctVal * R + size * (slippage + fee) * (ctVal * price)
        #   → size = risk_dollars / (ctVal * R + ctVal * price * (slip + fee))
        ct_val = float(specs.get("ctVal") or specs.get("ct_val") or 0.01)
        if ct_val <= 0:
            return None
        lot_sz = float(specs.get("lotSz") or specs.get("lot_sz") or 0.01)
        min_sz = float(specs.get("minSz") or specs.get("min_sz") or 0.01)

        # 缩仓系数：conf_mult ∈ [0.75, 1.25] 温和调整
        score_pos = float(np.clip(
            (weighted - self.min_entry_conf) / max(1.0 - self.min_entry_conf, 1e-6),
            0.0, 1.0,
        ))
        conf_mult = 0.75 + 0.50 * score_pos
        vol_mult = 0.70 if natr_pct >= self.natr_shrink_pct else 1.0
        risk_dollars = capital * self.risk_pct * conf_mult * vol_mult

        cost_per_lot = ct_val * cur_price * (self.slippage_pct + self.fee_rt)
        risk_per_lot = ct_val * R + cost_per_lot
        if risk_per_lot <= 0:
            return None
        size = risk_dollars / risk_per_lot
        size = int(size / max(lot_sz, 1e-12)) * lot_sz
        if size < min_sz:
            return None
        planned_risk_usd = float(size) * ct_val * (
            R + cur_price * (self.slippage_pct + self.fee_rt)
        )
        planned_risk_pct = planned_risk_usd / max(capital, 1e-9)

        # 动态 max_bars：高 NATR 段减半
        effective_max_bars = self.timeout_max_bars
        if natr >= self.max_bars_vol_halve_natr:
            effective_max_bars = max(5, int(self.timeout_max_bars * self.max_bars_vol_halve_factor))

        # 止盈档：+2R 不再硬平仓（让右尾收益跑出去）
        rr_list = [1.0, 1.5, 2.0]
        batch_ratios = [0.0, self.r15_partial_pct, 0.0]
        tps = []
        for rr in rr_list:
            if direction == "long":
                tps.append(cur_price + sl_dist * rr)
            else:
                tps.append(cur_price - sl_dist * rr)

        return Signal(
            action="open_long" if direction == "long" else "open_short",
            direction=direction,
            confidence=float(weighted),
            strategy=self.name,
            regime=regime,
            stop_loss=float(stop),
            take_profit=None,
            tp_batches=list(tps),
            batch_ratios=list(batch_ratios),
            size=float(size),
            atr=float(atr_val),
            entry_reference_high=float(don_breakout_high if direction == "long" else cur_price),
            entry_reference_low=float(don_breakout_low if direction == "short" else cur_price),
            mfe_window_bars=int(self.timeout_mfe_bars),
            mfe_min_r=float(self.timeout_mfe_r),
            breakout_window_bars=int(self.timeout_breakout_bars),
            max_bars=int(effective_max_bars),
            panic_gap_atr=float(self.panic_gap_atr),
            r1_be_buffer_atr=float(self.r1_be_buffer_atr),
            r15_partial_pct=float(self.r15_partial_pct),
            r2_chandelier_N=int(self.r2_chandelier_N),
            r2_chandelier_k=float(self.r2_chandelier_k),
            reason=(
                f"[MING] {direction.upper()} don={don_score:.2f} macd={macd_score:.2f} "
                f"weighted={weighted:.2f} adx={adx_val:.1f} natr_pct={natr_pct:.0%} "
                f"R={R:.4f} risk_pct={planned_risk_pct*100:.2f}% sz={size:.4f} "
                f"sl={stop:.4f} max_bars={effective_max_bars}"
                f"{' (vol-halved)' if effective_max_bars != self.timeout_max_bars else ''}"
            ),
            rr_list=list(rr_list),
        )

    def _don_signal(
        self, open_, close, high, low, vol, atr_val, adx_val, cur_price,
    ) -> tuple[float, bool, bool, float, float]:
        """返回 (don_score, don_long, don_short, breakout_high, breakout_low)。

        Donchian 突破允许在最近 hold_bars 根内发生（窗口共振），突破发生后
        默认 3 天内只要当前价格仍站在突破位之上 + 量能/影线/ADX 满足，
        即可开仓。
        """
        period = self.don_period
        n = len(close)
        if n < period + 5:
            return 0.0, False, False, cur_price, cur_price

        prev_high = high.shift(1).rolling(period).max()
        prev_low = low.shift(1).rolling(period).min()
        broke_up_series = (close > prev_high) & (close.shift(1) <= prev_high.shift(1))
        broke_down_series = (close < prev_low) & (close.shift(1) >= prev_low.shift(1))

        hold_bars = max(1, self.don_min_break_hold_bars)
        recent_up = broke_up_series.iloc[-hold_bars:].fillna(False)
        recent_down = broke_down_series.iloc[-hold_bars:].fillna(False)
        don_long_breakout = bool(recent_up.any())
        don_short_breakout = bool(recent_down.any())

        don_breakout_high = 0.0
        don_breakout_low = 0.0
        if don_long_breakout:
            up_idx = recent_up[recent_up].index[-1]
            up_pos = close.index.get_loc(up_idx)
            if up_pos - period >= 0:
                don_breakout_high = float(prev_high.iloc[up_pos])
            else:
                don_breakout_high = float(high.iloc[-(period + 1):-1].max())
        if don_short_breakout:
            dn_idx = recent_down[recent_down].index[-1]
            dn_pos = close.index.get_loc(dn_idx)
            if dn_pos - period >= 0:
                don_breakout_low = float(prev_low.iloc[dn_pos])
            else:
                don_breakout_low = float(low.iloc[-(period + 1):-1].min())

        # 当前价仍站在突破位之上（防假突破）
        if don_long_breakout and don_breakout_high > 0:
            don_long_breakout = cur_price > don_breakout_high
        if don_short_breakout and don_breakout_low > 0:
            don_short_breakout = cur_price < don_breakout_low

        # 量能 + 影线过滤
        vol_ma = float(vol.iloc[-min(20, n):].mean())
        cur_vol = float(vol.iloc[-1])
        vol_ratio = cur_vol / max(vol_ma, 1e-8)
        vol_ok = vol_ratio >= self.don_vol_ratio

        high_now = float(high.iloc[-1])
        low_now = float(low.iloc[-1])
        open_now = float(open_.iloc[-1])
        body = max(abs(cur_price - open_now), 1e-12)
        if cur_price >= open_now:
            upper_wick = high_now - cur_price
            lower_wick = open_now - low_now
        else:
            upper_wick = high_now - open_now
            lower_wick = cur_price - low_now
        max_wick = max(upper_wick, lower_wick)
        wick_ok = (max_wick / body) <= self.don_max_wick_ratio

        don_long = don_long_breakout and vol_ok and adx_val >= self.don_min_adx and wick_ok
        don_short = don_short_breakout and vol_ok and adx_val >= self.don_min_adx and wick_ok

        if not (don_long or don_short):
            return 0.0, False, False, don_breakout_high, don_breakout_low

        adx_strength = float(np.clip((adx_val - self.don_min_adx) / 20.0, 0, 1))
        ref = don_breakout_high if don_long else don_breakout_low
        break_pct = abs(cur_price - ref) / max(atr_val, 1e-12) if ref > 0 else 0.0
        break_strength = float(np.clip(break_pct / 1.5, 0, 1))
        vol_strength = float(np.clip((vol_ratio - 1.0) / 1.5, 0, 1))
        don_score = 0.30 + 0.35 * adx_strength + 0.20 * break_strength + 0.15 * vol_strength
        don_score = float(np.clip(don_score, 0.0, 1.0))
        return don_score, don_long, don_short, don_breakout_high, don_breakout_low

    def _macd_signal(self, close, adx_val, atr_val) -> tuple[float, bool, bool]:
        """返回 (macd_score, macd_long, macd_short)。

        触发条件: MACD 同号 + hist 同方向 (cur_hist > 0 for long, < 0 for short)
        评分: ATR 归一化（hist / ATR），跨 BTC/ETH/DOGE 可比
        """
        macd_line, signal_line, hist = macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        if len(macd_line) < 3:
            return 0.0, False, False
        cur_macd = float(macd_line.iloc[-1])
        cur_hist = float(hist.iloc[-1])
        if not all(map(np.isfinite, [cur_macd, cur_hist, atr_val])):
            return 0.0, False, False

        macd_long = bool(cur_macd > 0 and cur_hist > 0 and adx_val >= self.macd_min_adx)
        macd_short = bool(cur_macd < 0 and cur_hist < 0 and adx_val >= self.macd_min_adx)
        if not (macd_long or macd_short):
            return 0.0, False, False

        hist_strength = abs(cur_hist) / max(atr_val, 1e-12)
        hist_strength_norm = float(np.clip(hist_strength / 0.10, 0.0, 1.0))
        adx_strength = float(np.clip((adx_val - self.macd_min_adx) / 20.0, 0.0, 1.0))
        macd_score = 0.40 + 0.30 * hist_strength_norm + 0.30 * adx_strength
        macd_score = float(np.clip(macd_score, 0.0, 1.0))
        return macd_score, macd_long, macd_short

    def _initial_sl_distance(
        self, direction: str, cur_price: float, atr_val: float, natr_pct: float,
        high: pd.Series, low: pd.Series,
    ) -> float:
        """初始止损距离 R。

        D = max(D_atr, D_floor, D_struct) — 趋势策略止损应在结构之外（更宽），
        否则容易被洗出去。
        """
        f_min = self.sl_floor_pct
        d_atr = self._k_vol_for_natr_pct(natr_pct) * atr_val
        d_floor = cur_price * f_min
        D = max(d_atr, d_floor)

        N = self.sl_struct_lookback
        if len(low) >= N + 1 and direction == "long":
            structure_low = float(low.iloc[-(N + 1):-1].min())
            structure_stop = structure_low - self.sl_struct_buffer_atr * atr_val
            structure_dist = cur_price - structure_stop
            if np.isfinite(structure_dist) and structure_dist > 0:
                D = max(D, structure_dist)
        elif len(high) >= N + 1 and direction == "short":
            structure_high = float(high.iloc[-(N + 1):-1].max())
            structure_stop = structure_high + self.sl_struct_buffer_atr * atr_val
            structure_dist = structure_stop - cur_price
            if np.isfinite(structure_dist) and structure_dist > 0:
                D = max(D, structure_dist)

        return float(max(D, d_floor))

    def _k_vol_for_natr_pct(self, natr_pct: float) -> float:
        """NATR 百分位 → ATR 倍数（三段插值，sl_k_vol_mid 实际生效）。

        natr_pct <= 0.25      → sl_k_vol_low
        0.25 < p <= 0.50     → low → mid 线性插值
        0.50 < p <  0.75     → mid → high 线性插值
        natr_pct >= 0.75      → sl_k_vol_high
        """
        p = float(natr_pct)
        if p <= 0.25:
            return self.sl_k_vol_low
        if p <= 0.50:
            t = (p - 0.25) / 0.25
            return self.sl_k_vol_low + t * (self.sl_k_vol_mid - self.sl_k_vol_low)
        if p < 0.75:
            t = (p - 0.50) / 0.25
            return self.sl_k_vol_mid + t * (self.sl_k_vol_high - self.sl_k_vol_mid)
        return self.sl_k_vol_high
