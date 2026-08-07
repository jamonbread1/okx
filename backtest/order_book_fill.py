# -*- coding: utf-8 -*-
"""
订单簿深度动态滑点（虚拟币合约回测，贴近实盘）

摒弃：固定比例滑点、收盘价必成交、纯 Tick 噪声模型。

核心：
  1. 信号时刻 + 执行延迟 → 基准中间价
  2. 用前 N 档合成/真实盘口，按下单量「吃单」得 VWAP
  3. 深度不足 → 部分成交或拒单
  4. 市价吃单；限价未触及对手价则不成交

无历史 L2 时：用 K 线 vol + 品种流动性档位合成指数衰减盘口（可复现、可压测）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from logger import setup_logger

log = setup_logger("ob_fill")

# 合约 tick / 默认点差（价差，价格单位）
TICK_SZ = {
    "BTC": 0.1, "ETH": 0.01, "SOL": 0.01, "BNB": 0.01,
    "XRP": 0.0001, "DOGE": 0.00001,
}
# 流动性档位：主流更紧、深度更大
LIQ_TIER = {
    "BTC": "major", "ETH": "major",
    "SOL": "mid", "BNB": "mid", "XRP": "mid",
    "DOGE": "alt",
}


def _base(symbol: str) -> str:
    return symbol.split("-")[0].upper()


def tick_size(symbol: str) -> float:
    return float(TICK_SZ.get(_base(symbol), 0.01))


def _round_adverse(px: float, tick: float, is_buy: bool) -> float:
    if tick <= 0:
        return px
    if is_buy:
        return math.ceil(px / tick - 1e-12) * tick
    return math.floor(px / tick + 1e-12) * tick


@dataclass
class BookLevel:
    price: float
    size: float  # 张数或币数量（与策略 sz 同单位）


@dataclass
class OBFillConfig:
    # 执行延迟（毫秒）——散户 API 常见 100~150
    latency_ms: float = 120.0
    latency_jitter_ms: float = 30.0
    # 盘口档数
    book_levels: int = 10
    # 合成盘口：第一档量 = bar_vol * level0_vol_frac（再指数衰减）
    level0_vol_frac: float = 0.08
    level_decay: float = 0.72
    # 点差：半边相对中间价（再按档位放大）
    half_spread_bps_major: float = 0.8   # 0.008%
    half_spread_bps_mid: float = 2.5
    half_spread_bps_alt: float = 8.0
    # 深度不足时：partial=部分成交 / reject=整单拒绝
    depth_policy: str = "partial"  # partial | reject
    min_fill_ratio: float = 0.15   # 部分成交低于此比例视为失败
    # 延迟后价格漂移：用 bar 振幅的一小部分模拟 120ms 内变动
    delay_drift_frac: float = 0.05
    # 当根成交量中视为「近端挂单」的比例（张）
    depth_participation: float = 0.02
    # 是否允许用下一根 open（默认关，消除 1-bar 成交前瞻）
    use_next_open: bool = False
    # 种子
    seed: Optional[int] = 42


@dataclass
class OBFillResult:
    filled: bool
    price: float          # VWAP 成交价
    filled_sz: float      # 实际成交张数
    requested_sz: float
    levels_eaten: int
    latency_ms: float
    slip_bps: float       # 相对 signal_px 的滑点（bp）
    reason: str
    book_mode: str = "synthetic"  # synthetic | real


class OrderBookFillEngine:
    def __init__(self, cfg: Optional[OBFillConfig] = None):
        self.cfg = cfg or OBFillConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        # 可选：外部注入真实盘口 {symbol: {"bids":[(px,sz)...], "asks":[...]}}
        self._real_books: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    def set_real_book(
        self, symbol: str, bids: Sequence[Tuple[float, float]], asks: Sequence[Tuple[float, float]]
    ) -> None:
        """实盘/回放注入 L2：bids 高价在前，asks 低价在前。"""
        self._real_books[symbol] = {
            "bids": [(float(p), float(s)) for p, s in bids],
            "asks": [(float(p), float(s)) for p, s in asks],
        }

    def clear_real_book(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._real_books.clear()
        else:
            self._real_books.pop(symbol, None)

    def _latency(self) -> float:
        j = float(self._rng.uniform(-self.cfg.latency_jitter_ms, self.cfg.latency_jitter_ms))
        return max(20.0, float(self.cfg.latency_ms) + j)

    def _half_spread(self, symbol: str, mid: float) -> float:
        tier = LIQ_TIER.get(_base(symbol), "alt")
        if tier == "major":
            bps = self.cfg.half_spread_bps_major
        elif tier == "mid":
            bps = self.cfg.half_spread_bps_mid
        else:
            bps = self.cfg.half_spread_bps_alt
        # 再加一点随机
        bps *= float(self._rng.uniform(0.85, 1.25))
        return mid * (bps / 10000.0)

    def _synthetic_book(
        self,
        symbol: str,
        mid: float,
        bar_vol: float,
        is_buy: bool,
    ) -> List[BookLevel]:
        """
        合成对手盘：买入吃 asks，卖出吃 bids。
        量按指数衰减分布在 N 档；价格按 tick 递进。
        """
        tick = tick_size(symbol)
        half = max(self._half_spread(symbol, mid), tick * 0.5)
        n = max(1, int(self.cfg.book_levels))
        # OKX SWAP history-candles 的 vol 通常为「张」；策略 sz 亦为张。
        # 若上游误传币数量级（极大/极小），用 mid 做粗校验并回退到保守深度。
        part = float(getattr(self.cfg, "depth_participation", 0.02))
        raw_vol = float(bar_vol or 0.0)
        if raw_vol < 0:
            raw_vol = 0.0
        # 异常保护：vol 相对 mid 极端离谱时退化为「中等流动性」地板
        # （避免币数量当张数 → 深度虚高，或单位错误 → 深度为 0 全拒单）
        total = max(raw_vol * part, 1e-8)
        if mid > 0 and raw_vol > 0:
            # 名义成交额粗估；若 implied notional 异常小/大，夹紧参与深度
            implied_notional = raw_vol * mid  # 若 vol 已是张且 ctVal≪1，会偏小，仅作相对保护
            if implied_notional < 1.0:
                total = max(total, 5.0)  # 至少 5 张近端深度
            elif raw_vol > 1e7:
                total = max(raw_vol * part * 0.01, 10.0)  # 疑似未换算的过大 tick 量
        level0 = total * float(self.cfg.level0_vol_frac)
        levels: List[BookLevel] = []
        for i in range(n):
            if is_buy:
                px = mid + half + i * tick
            else:
                px = mid - half - i * tick
            # 指数衰减份额
            share = (self.cfg.level_decay ** i)
            sz = level0 * share
            # 保证总深度大致收敛
            levels.append(BookLevel(price=float(px), size=float(max(sz, 0.0))))
        # 若总深度过小，放大到至少能吃小单
        ssum = sum(x.size for x in levels)
        if ssum < 1e-12:
            levels[0].size = max(1.0, total * 0.1)
        return levels

    def _real_or_synth_levels(
        self, symbol: str, mid: float, bar_vol: float, is_buy: bool
    ) -> Tuple[List[BookLevel], str]:
        rb = self._real_books.get(symbol)
        if rb:
            raw = rb["asks"] if is_buy else rb["bids"]
            if raw:
                levels = [BookLevel(price=p, size=s) for p, s in raw[: self.cfg.book_levels]]
                return levels, "real"
        return self._synthetic_book(symbol, mid, bar_vol, is_buy), "synthetic"

    def _walk_book(
        self, levels: List[BookLevel], need_sz: float
    ) -> Tuple[float, float, int, float]:
        """
        返回 (vwap, filled_sz, levels_eaten, available_total)
        """
        remain = float(need_sz)
        notional = 0.0
        filled = 0.0
        eaten = 0
        available = sum(max(0.0, lv.size) for lv in levels)
        for lv in levels:
            if remain <= 1e-15:
                break
            take = min(remain, max(0.0, lv.size))
            if take <= 0:
                continue
            notional += take * lv.price
            filled += take
            remain -= take
            eaten += 1
        vwap = (notional / filled) if filled > 0 else 0.0
        return vwap, filled, eaten, available

    def _mid_after_delay(
        self,
        signal_px: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        next_open: Optional[float] = None,
        use_next_open: bool = False,
    ) -> float:
        """
        延迟后基准中间价（默认无下一根前瞻）：
          - 默认：signal_px（已收盘确认价）或当根 open
          - use_next_open=True 才用下一根 open（旧行为，有 1-bar 前瞻）
          - 再叠加振幅 * delay_drift_frac * 随机（可关）
        """
        if use_next_open and next_open and next_open > 0:
            mid = float(next_open)
        elif signal_px and signal_px > 0:
            mid = float(signal_px)
        elif bar_open > 0:
            mid = float(bar_open)
        else:
            mid = float(signal_px or 0)
        rng = max(0.0, float(bar_high) - float(bar_low))
        drift = rng * float(self.cfg.delay_drift_frac) * float(self._rng.uniform(-1.0, 1.0))
        return mid + drift

    def fill_market(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        signal_px: float,
        bar_vol: float = 0.0,
        bar_open: float = 0.0,
        bar_high: float = 0.0,
        bar_low: float = 0.0,
        next_open: Optional[float] = None,
    ) -> OBFillResult:
        if size <= 0:
            return OBFillResult(False, signal_px, 0.0, size, 0, 0.0, 0.0, "size<=0")

        lat = self._latency()
        mid = self._mid_after_delay(signal_px, bar_open, bar_high, bar_low, next_open, use_next_open=bool(self.cfg.use_next_open))
        levels, book_mode = self._real_or_synth_levels(symbol, mid, max(bar_vol, 0.0), is_buy)
        vwap, filled, eaten, available = self._walk_book(levels, size)

        if filled <= 0 or vwap <= 0:
            return OBFillResult(
                False, signal_px, 0.0, size, 0, lat, 0.0, "empty_book", book_mode
            )

        ratio = filled / size
        policy = (self.cfg.depth_policy or "partial").lower()
        if ratio < 0.999:
            if policy == "reject" or ratio < float(self.cfg.min_fill_ratio):
                return OBFillResult(
                    False, signal_px, 0.0, size, eaten, lat, 0.0,
                    f"depth_reject avail={available:.4f} need={size:.4f} ratio={ratio:.2%}",
                    book_mode,
                )

        tick = tick_size(symbol)
        fill_px = _round_adverse(vwap, tick, is_buy)
        slip_bps = (fill_px / signal_px - 1.0) * 10000.0 if signal_px > 0 else 0.0
        if not is_buy:
            slip_bps = -slip_bps  # 卖出：更低价 = 正滑点成本
        # 统一成「成本为正」的 bp
        cost_bps = abs((fill_px - signal_px) / signal_px * 10000.0) if signal_px > 0 else 0.0

        reason = "ob_vwap"
        if ratio < 0.999:
            reason = f"ob_partial({filled:.4f}/{size:.4f})"

        return OBFillResult(
            filled=True,
            price=float(fill_px),
            filled_sz=float(filled),
            requested_sz=float(size),
            levels_eaten=int(eaten),
            latency_ms=lat,
            slip_bps=float(cost_bps),
            reason=reason,
            book_mode=book_mode,
        )

    def fill_limit(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        limit_px: float,
        signal_px: float,
        bar_high: float,
        bar_low: float,
        bar_vol: float = 0.0,
        bar_open: float = 0.0,
        next_open: Optional[float] = None,
    ) -> OBFillResult:
        """
        限价：延迟后若 bar 曾触及可成交方向（买限价 >= 最低可买 / 卖限价 <= 最高可卖）
        则按 min(limit, 对手价) 成交，且仍受深度约束；否则不成交。
        """
        lat = self._latency()
        mid = self._mid_after_delay(signal_px, bar_open, bar_high, bar_low, next_open, use_next_open=bool(self.cfg.use_next_open))
        # 是否触及：用当根 high/low 代理延迟窗口内价格轨迹
        if is_buy:
            # 买限价单：主动限价单 limit 通常高于 signal_px；只要 signal_px 本身不劣于限价，
            # 即视为可按限价保护吃单，避免缺少精确盘口时出现“本应可成交却未触价”的假拒单。
            touched = float(bar_low) <= float(limit_px) or mid <= float(limit_px) or float(signal_px) <= float(limit_px)
            if not touched:
                return OBFillResult(
                    False, limit_px, 0.0, size, 0, lat, 0.0, "limit_not_touched"
                )
            # 成交价不超过限价，且不优于对手一档太多
            levels, book_mode = self._real_or_synth_levels(symbol, mid, bar_vol, True)
            # 吃单但仍封顶 limit
            vwap, filled, eaten, _ = self._walk_book(levels, size)
            if filled <= 0:
                return OBFillResult(False, limit_px, 0.0, size, 0, lat, 0.0, "limit_no_depth", book_mode)
            fill_px = min(float(limit_px), vwap)
        else:
            # 卖限价单同理：主动限价单 limit 通常低于 signal_px。
            touched = float(bar_high) >= float(limit_px) or mid >= float(limit_px) or float(signal_px) >= float(limit_px)
            if not touched:
                return OBFillResult(
                    False, limit_px, 0.0, size, 0, lat, 0.0, "limit_not_touched"
                )
            levels, book_mode = self._real_or_synth_levels(symbol, mid, bar_vol, False)
            vwap, filled, eaten, _ = self._walk_book(levels, size)
            if filled <= 0:
                return OBFillResult(False, limit_px, 0.0, size, 0, lat, 0.0, "limit_no_depth", book_mode)
            fill_px = max(float(limit_px), vwap)

        ratio = filled / size
        if ratio < float(self.cfg.min_fill_ratio):
            return OBFillResult(
                False, limit_px, 0.0, size, eaten, lat, 0.0, "limit_depth_reject", book_mode
            )
        tick = tick_size(symbol)
        fill_px = _round_adverse(fill_px, tick, is_buy)
        cost_bps = abs((fill_px - signal_px) / signal_px * 10000.0) if signal_px > 0 else 0.0
        return OBFillResult(
            True, float(fill_px), float(filled), float(size), int(eaten), lat,
            float(cost_bps), "limit_fill", book_mode,
        )


def ob_config_from_strategy(st: Dict) -> OBFillConfig:
    return OBFillConfig(
        latency_ms=float(st.get("ob_latency_ms", 120)),
        latency_jitter_ms=float(st.get("ob_latency_jitter_ms", 30)),
        book_levels=int(st.get("ob_book_levels", 10)),
        level0_vol_frac=float(st.get("ob_level0_vol_frac", 0.08)),
        level_decay=float(st.get("ob_level_decay", 0.72)),
        half_spread_bps_major=float(st.get("ob_half_spread_bps_major", 0.8)),
        half_spread_bps_mid=float(st.get("ob_half_spread_bps_mid", 2.5)),
        half_spread_bps_alt=float(st.get("ob_half_spread_bps_alt", 8.0)),
        depth_policy=str(st.get("ob_depth_policy", "partial")),
        min_fill_ratio=float(st.get("ob_min_fill_ratio", 0.15)),
        delay_drift_frac=float(st.get("ob_delay_drift_frac", 0.05)),
        depth_participation=float(st.get("ob_depth_participation", 0.02)),
        use_next_open=bool(st.get("ob_use_next_open", False)),
        seed=st.get("ob_fill_seed", 42),
    )
