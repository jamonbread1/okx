# -*- coding: utf-8 -*-
"""
多币种组合回测引擎。

数据：SQLite bars.db / parquet（官方 Trade history 转换），
支持日期区间 + 自选主周期。

接口：直接调用 v3.engine.StrategyEngine 的原生接口
  generate_signal(inst_id, df, df_htf, capital, leverage, specs, funding_rate)，
  成交后由回测层调用 confirm_fill / confirm_partial_close / confirm_close
  把真实成交同步回引擎，引擎不再登记幻影仓位。

回测真实性：
  - 资金费：每逢结算时刻（默认 00/08/16 UTC）按 notional * rate * direction 计入 realized
  - intrabar 止损/止盈：由 v3 引擎按已收盘 bar 的 high/low 做保守触发，本层按最差价成交
  - 成交时点：默认用「下一根 bar 开盘价」成交（信号用上一根已收盘指标）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.metrics import check_overfitting, compute_metrics, metrics_to_frame
from backtest.progress import ProgressBar
from backtest.order_book_fill import OrderBookFillEngine, ob_config_from_strategy
from backtest.trade_pipeline import (
    COMMON_BARS,
    DB_PATH,
    PQ_ROOT,
    db_has_symbol_bar,
    load_bars,
)
from logger import setup_logger
from universe import allocate_capital

# v3 引擎直接引入
from v3.engine import StrategyEngine

log = setup_logger("bt_multi")

# 策略侧需要的辅助周期（相对主周期）
_HELPER_BARS = ("1m", "5m", "15m", "1H", "4H")

# BTC beta 折算
_BETA = {"BTC": 1.0, "ETH": 0.85, "SOL": 0.90, "DOGE": 0.75, "AVAX": 0.8}


class ParquetBarFetcher:
    """高速本地 K 线供给：用 searchsorted + iloc，避免每根 bar 全表布尔过滤。"""

    def __init__(self, store: Dict[str, Dict[str, pd.DataFrame]]):
        self.store = store
        self._i: Dict[str, int] = {}
        self._main_bar = "1m"
        self._inst_meta: Dict[str, Dict] = {}
        # 预计算时间戳 numpy 数组，加速对齐
        self._ts: Dict[str, Dict[str, np.ndarray]] = {}
        for sym, bars in store.items():
            self._ts[sym] = {}
            for b, df in bars.items():
                if df is None or df.empty:
                    continue
                ts = pd.to_datetime(df["ts"]).values.astype("datetime64[ns]")
                self._ts[sym][b] = ts

    def set_main_bar(self, bar: str):
        self._main_bar = bar

    def set_cursor(self, symbol: str, i: int):
        self._i[symbol] = int(i)

    def _slice(self, symbol: str, bar: str, limit: int) -> Optional[pd.DataFrame]:
        bars = self.store.get(symbol) or {}
        df = bars.get(bar)
        if df is None or df.empty:
            return None
        main = bars.get(self._main_bar)
        if main is None or main.empty:
            return None
        i = self._i.get(symbol, 0)
        if i < 0 or i >= len(main):
            return None
        ts_arr = self._ts.get(symbol, {}).get(self._main_bar)
        bar_ts = self._ts.get(symbol, {}).get(bar)
        if ts_arr is None or bar_ts is None:
            # fallback
            ts = main["ts"].iloc[i]
            sub = df[df["ts"] <= ts].tail(limit)
            return None if len(sub) < 5 else sub.reset_index(drop=True)
        t = ts_arr[i]
        # 右边界：<= 当前主周期时间的最后一根
        j = int(np.searchsorted(bar_ts, t, side="right") - 1)
        if j < 4:
            return None
        start = max(0, j - limit + 1)
        return df.iloc[start : j + 1].reset_index(drop=True)

    def get_candles_df(self, inst_id: str, bar: str = "15m", limit: int = 100):
        """
        signal_on_closed_bar=True（默认）：
          - 指标只用「已收盘」K 线：丢弃与主周期当前时间对齐的最后一根未确认 bar
          - 避免用当根完整 OHLC 做开仓决策（消除 bar 级未来信息）
        """
        df = self._slice(inst_id, bar, limit + (1 if getattr(self, "signal_on_closed_bar", True) else 0))
        if df is None or len(df) < 5:
            return df
        if getattr(self, "signal_on_closed_bar", True) and bar != getattr(self, "_main_bar", "1m"):
            # 高周期：最后一根若 ts 对齐到当前主周期时刻，视为未收盘，去掉
            main = (self.store.get(inst_id) or {}).get(self._main_bar)
            i = self._i.get(inst_id, 0)
            if main is not None and 0 <= i < len(main):
                cur_ts = pd.Timestamp(main["ts"].iloc[i])
                last_ts = pd.Timestamp(df["ts"].iloc[-1])
                # 若最后一根 bar 的开始时间 >= 当前主周期所在的该周期边界，丢弃
                if last_ts >= cur_ts:
                    df = df.iloc[:-1].reset_index(drop=True)
        elif getattr(self, "signal_on_closed_bar", True) and bar == getattr(self, "_main_bar", "1m"):
            # 主周期：开仓信号用上一根已收盘 close，故蜡烛序列去掉最后一根
            if len(df) > 5:
                df = df.iloc[:-1].reset_index(drop=True)
        return df if df is not None and len(df) >= 5 else df

    def get_last_price(self, inst_id: str) -> float:
        """盯市价 = 当前主周期 close（持仓 PnL / 风控）。开仓指标见 get_candles_df（已收盘）。"""
        main = (self.store.get(inst_id) or {}).get(self._main_bar)
        if main is None:
            return 0.0
        i = self._i.get(inst_id, 0)
        if i < 0 or i >= len(main):
            return 0.0
        return float(main["close"].iloc[i])

    def get_bar_open(self, inst_id: str) -> float:
        """当前主周期 bar 的开盘价。"""
        main = (self.store.get(inst_id) or {}).get(self._main_bar)
        if main is None:
            return 0.0
        i = self._i.get(inst_id, 0)
        if i < 0 or i >= len(main):
            return 0.0
        return float(main["open"].iloc[i])

    def get_instrument_info(self, inst_id: str) -> Dict:
        if inst_id in self._inst_meta:
            return self._inst_meta[inst_id]
        base = inst_id.split("-")[0]
        # OKX 风格 camelCase + snake_case 双写，避免 sizing 读不到 ct_val
        if base == "BTC":
            ct, lot, tick = "0.01", "0.01", "0.1"
        elif base == "ETH":
            ct, lot, tick = "0.1", "0.01", "0.01"
        elif base == "SOL":
            ct, lot, tick = "1", "1", "0.01"
        else:
            ct, lot, tick = "1", "1", "0.01"
        meta = {
            "tickSz": tick, "lotSz": lot, "minSz": lot, "ctVal": ct,
            "tick_sz": tick, "lot_sz": lot, "min_sz": lot, "ct_val": ct,
        }
        self._inst_meta[inst_id] = meta
        return meta

    def get_order_book_imbalance(self, inst_id: str, depth: int = 5) -> float:
        df = self._slice(inst_id, "1m", 20)
        if df is None or "buy_vol" not in df.columns:
            return 0.0
        b = float(df["buy_vol"].iloc[-5:].sum())
        s = float(df["sell_vol"].iloc[-5:].sum())
        tot = b + s
        return 0.0 if tot <= 0 else (b - s) / tot

    def get_cvd_proxy(self, inst_id: str, limit: int = 60) -> float:
        df = self._slice(inst_id, "1m", limit)
        if df is None or "cvd" not in df.columns or len(df) < 3:
            return 0.0
        cvd = df["cvd"].astype(float)
        delta = float(cvd.iloc[-1] - cvd.iloc[0])
        scale = float(df["vol"].sum()) + 1e-8
        return float(np.clip(delta / scale, -1, 1))

    def get_account_usdt_equity(self) -> float:
        return 0.0


@dataclass
class SimPos:
    long_sz: float = 0.0
    short_sz: float = 0.0
    entry_long: float = 0.0
    entry_short: float = 0.0
    realized: float = 0.0
    rt_start_realized: float = 0.0  # 本轮开仓前 realized，用于 Kelly 真实 $PnL
    strategy: str = ""              # 当前持仓来源策略，用于强制平仓归因


@dataclass
class MultiBacktestResult:
    metrics: Dict
    equity_curve: pd.DataFrame
    trades: List[Dict] = field(default_factory=list)
    phase: str = ""


class MultiSymbolBacktester:
    """多币种同步走步，资金池共享，逻辑贴近 PortfolioManager。"""

    def __init__(
        self,
        config: Dict,
        symbols: List[str],
        main_bar: str = "1m",
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ):
        self.config = config
        self.symbols = symbols
        self.main_bar = main_bar
        self.start = pd.Timestamp(start) if start is not None else None
        self.end = pd.Timestamp(end) if end is not None else None
        st = config.get("strategy") or {}
        # 默认往返费率（开 maker 平 taker 近似）
        m = float(st.get("mom_maker_fee", st.get("ema_break_maker_fee", 0.0002)))
        tk = float(st.get("mom_taker_fee", st.get("ema_break_taker_fee", 0.0005)))
        self.fee_rt = m + tk
        # 滑点：取各策略中最保守者，并可 stress 放大（防过拟合）
        slip_m = float(st.get("mom_slippage_pct", 0.00040))
        slip_e = float(st.get("ema_break_slippage_pct", 0.00030))
        slip_v = float(st.get("vol_break_slippage_pct", 0.00035))
        self.slip = max(slip_m, slip_e, slip_v)
        stress = float(st.get("bt_slippage_stress", 1.5))
        self.slip = self.slip * stress
        # 按模式细分滑点/费率（与 _apply 配合）
        self.slip_by_mode = {
            "default": self.slip,
            "mom": slip_m * stress,
            "ema": slip_e * stress,
            "vol": slip_v * stress,
        }
        self.fee_rt_by_mode = {
            "default": self.fee_rt,
            "mom": self.fee_rt,
            "ema": self.fee_rt,
            "vol": self.fee_rt,
        }
        # 订单簿深度动态滑点（延迟 + 吃单 VWAP + 部分成交/拒单）
        self.use_ob_fill = bool(st.get("enable_ob_fill", True))
        self.ob_fill = OrderBookFillEngine(ob_config_from_strategy(st)) if self.use_ob_fill else None
        # 默认使用“主动限价单”：买单 = 信号价 + N*tick，卖单 = 信号价 - N*tick。
        # 它保留限价保护，同时足够 aggressive，通常可成交。
        self.execution_order_type = str(st.get("execution_order_type", "aggressive_limit")).strip().lower()
        self.aggressive_limit_ticks = float(st.get("aggressive_limit_ticks", 5.0))
        self.aggressive_limit_tick_fallback = float(st.get("aggressive_limit_tick_fallback", 0.1))
        self._bar_cursor: Dict[str, int] = {}
        self._main_dfs: Dict[str, pd.DataFrame] = {}

    def _load_store(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        store: Dict[str, Dict[str, pd.DataFrame]] = {}
        need = [self.main_bar] + list(_HELPER_BARS)
        bars = list(dict.fromkeys(need))
        for sym in self.symbols:
            store[sym] = {}
            for b in bars:
                try:
                    df = load_bars(sym, bar=b, start=self.start, end=self.end)
                    if df is None or len(df) < 5:
                        if b == self.main_bar:
                            raise FileNotFoundError(f"{sym} {b} 在选定日期内无数据")
                        log.warning(f"{sym} {b} 数据过少，跳过辅助周期")
                        continue
                    store[sym][b] = df
                    t0, t1 = df["ts"].iloc[0], df["ts"].iloc[-1]
                    log.info(f"加载 {sym} {b} rows={len(df)} | {t0} → {t1}")
                except FileNotFoundError as e:
                    if b == self.main_bar:
                        raise
                    log.warning(str(e))
            if self.main_bar not in store[sym]:
                raise FileNotFoundError(f"{sym} 缺少主周期 {self.main_bar}")
        if self.ob_fill is not None:
            log.info("订单簿深度动态滑点已启用（合成 L2 / 可注入真实盘口）")
        return store

    def _aligned_index(self, store) -> pd.DatetimeIndex:
        """取所有品种主周期时间交集，保证同步。

        智能剔除（v7.1）：若某品种主周期数据 < 全部品种主周期长度的 50%，
        自动从 store 弹出并 warn（避免 1 个新币种拖垮整个回测）。
        """
        # 先算每个品种主周期长度，按短到长排序
        sym_lens = []
        for sym in self.symbols:
            df = store.get(sym, {}).get(self.main_bar)
            if df is None or df.empty:
                continue
            sym_lens.append((sym, len(df)))
        if not sym_lens:
            raise RuntimeError("多币种时间交集过短，请检查各品种 parquet 时间范围是否重叠")
        max_len = max(n for _, n in sym_lens)
        drop_short = [
            sym for sym, n in sym_lens
            if n < max_len * 0.5 and max_len > 0
        ]
        if drop_short:
            log.warning(
                f"[aligned_index] 自动剔除数据过短品种 {drop_short} "
                f"（主周期 {self.main_bar} bars < 最长品种的 50%）"
            )
            for sym in drop_short:
                store.pop(sym, None)
                self.symbols = [s for s in self.symbols if s != sym]
        # 取交集
        common = None
        for sym in self.symbols:
            ts = pd.DatetimeIndex(store[sym][self.main_bar]["ts"])
            common = ts if common is None else common.intersection(ts)
        if common is None or len(common) < 100:
            raise RuntimeError("多币种时间交集过短，请检查各品种 parquet 时间范围是否重叠")
        return common.sort_values()

    def _ct_val(self, sym: str) -> float:
        base = sym.split("-")[0]
        return {"BTC": 0.01, "ETH": 0.1}.get(base, 1.0)

    def _tick_sz(self, sym: str) -> float:
        base = sym.split("-")[0]
        return {"BTC": 0.1, "ETH": 0.01, "SOL": 0.01, "DOGE": 0.00001, "AVAX": 0.01}.get(
            base,
            float(getattr(self, "aggressive_limit_tick_fallback", 0.1)),
        )

    def _execution_params(self, sym: str, is_buy: bool, signal_px: float) -> tuple[str, float]:
        """返回执行订单类型和限价。默认是主动限价单。

        买单：limit = signal_px + aggressive_limit_ticks * tick
        卖单：limit = signal_px - aggressive_limit_ticks * tick
        """
        if str(getattr(self, "execution_order_type", "market")).lower() not in (
            "aggressive_limit", "limit", "limit_aggressive"
        ):
            return "market", 0.0
        px = float(signal_px or 0.0)
        if px <= 0:
            return "market", 0.0
        tick = float(self._tick_sz(sym) or 0.0)
        if tick <= 0:
            tick = float(getattr(self, "aggressive_limit_tick_fallback", 0.1))
        offset = max(tick, float(getattr(self, "aggressive_limit_ticks", 5.0)) * tick)
        limit_px = px + offset if is_buy else max(tick, px - offset)
        return "limit", float(limit_px)

    def _resolve_fill(
        self,
        sym: str,
        is_buy: bool,
        sz: float,
        px: float,
        ts,
        mode: str,
        ord_type: str = "market",
        limit_px: float = 0.0,
    ) -> tuple:
        """
        订单簿深度动态成交。
        返回 (fill_price, filled_sz, meta)。
        filled_sz 可能 < sz（部分成交）；0 表示拒单。
        """
        meta = {
            "mode": "pct_legacy",
            "slip_bps": 0.0,
            "latency_ms": 0.0,
            "reason": "",
            "levels_eaten": 0,
            "book_mode": "",
            "filled_sz": float(sz),
        }
        if self.ob_fill is None:
            mkey = (mode or "default").lower()
            slip = float(self.slip_by_mode.get(mkey, self.slip))
            fill = px * (1 + slip) if is_buy else px * (1 - slip)
            meta["reason"] = "legacy_pct"
            return fill, float(sz), meta

        bar_open = bar_high = bar_low = bar_vol = 0.0
        next_open = None
        main = self._main_dfs.get(sym)
        if main is not None and len(main) > 0:
            i = int(self._bar_cursor.get(sym, 0))
            i = max(0, min(i, len(main) - 1))
            row = main.iloc[i]
            bar_open = float(row.get("open", px) or px)
            bar_high = float(row.get("high", px) or px)
            bar_low = float(row.get("low", px) or px)
            bar_vol = float(row.get("vol", 0) or 0)
            if i + 1 < len(main):
                next_open = float(main.iloc[i + 1].get("open", 0) or 0) or None

        if (ord_type or "market").lower() == "limit" and limit_px > 0:
            fr = self.ob_fill.fill_limit(
                symbol=sym, is_buy=is_buy, size=float(sz), limit_px=float(limit_px),
                signal_px=float(px), bar_high=bar_high, bar_low=bar_low,
                bar_vol=bar_vol, bar_open=bar_open, next_open=next_open,
            )
        else:
            fr = self.ob_fill.fill_market(
                symbol=sym, is_buy=is_buy, size=float(sz), signal_px=float(px),
                bar_vol=bar_vol, bar_open=bar_open, bar_high=bar_high, bar_low=bar_low,
                next_open=next_open,
            )

        if not fr.filled or fr.filled_sz <= 0:
            meta.update({
                "mode": "rejected",
                "reason": fr.reason,
                "latency_ms": fr.latency_ms,
                "filled_sz": 0.0,
                "book_mode": fr.book_mode,
            })
            return float(px), 0.0, meta

        fill_price = float(fr.price)
        fill_reason = fr.reason
        # 合成订单簿可能在低流动性/大滑点时把 VWAP 推到当前 K 线 OHLC 外。
        # bar 级回测无法证明 K 线范围外成交存在，因此统一夹到 [low, high]。
        # 注意：close_long 是卖出，价格高于 high 也同样不可成交，不能只夹 adverse 方向。
        if bar_high > 0 and fill_price > bar_high:
            fill_price = float(bar_high)
            fill_reason = f"{fill_reason}|bar_high_clamp"
        if bar_low > 0 and fill_price < bar_low:
            fill_price = float(bar_low)
            fill_reason = f"{fill_reason}|bar_low_clamp"
        slip_bps = abs((fill_price - float(px)) / float(px) * 10000.0) if px and px > 0 else float(fr.slip_bps)
        meta.update({
            "mode": "orderbook",
            "slip_bps": slip_bps,
            "latency_ms": fr.latency_ms,
            "reason": fill_reason,
            "levels_eaten": fr.levels_eaten,
            "book_mode": fr.book_mode,
            "filled_sz": float(fr.filled_sz),
        })
        return fill_price, float(fr.filled_sz), meta

    def _apply(self, pos: SimPos, side: str, action: str, sz: float, px: float, note: str, trades: list, sym: str, mode: str = "default", ts=None, ord_type: str = "market", limit_px: float = 0.0):
        """sz 为张数；成交价由订单簿深度引擎决定；深度不足可部分成交或拒单。

        返回 (filled_sz, fill_price)；拒单/零成交返回 (0, px)。
        """
        if sz <= 0:
            return 0.0, float(px)
        _ts_str = str(pd.Timestamp(ts)) if ts is not None else ""
        def _stamp_ts():
            if not trades:
                return
            if _ts_str and not trades[-1].get("ts"):
                trades[-1]["ts"] = _ts_str
            trades[-1]["ord_type"] = str(ord_type or "market")
            trades[-1]["limit_px"] = float(limit_px or 0.0)
        ctv = self._ct_val(sym)
        mkey = (mode or "default").lower()
        if not hasattr(self, "slip_by_mode") or not self.slip_by_mode:
            self.slip_by_mode = {"default": float(getattr(self, "slip", 0.0004))}
            self.fee_rt_by_mode = {"default": float(getattr(self, "fee_rt", 0.0007))}
        if mkey not in getattr(self, "slip_by_mode", {}):
            n = (note or "").upper()
            if "SCALP" in n or "IMPULSE" in n:
                mkey = "scalp"
            elif "[MR]" in n or n.startswith("MR"):
                mkey = "mr"
            elif "MOM" in n:
                mkey = "mom"
            else:
                mkey = "default"
        fee_rt = float(self.fee_rt_by_mode.get(mkey, self.fee_rt))

        def _fee(notional_sz, fill_px):
            return abs(notional_sz) * ctv * fill_px * (fee_rt / 2.0)

        strat = (mode or "default")
        if action in ("open_long", "hft_long", "scale_in") and side == "long":
            fill, fsz, fmeta = self._resolve_fill(sym, True, sz, px, ts, mkey, ord_type, limit_px)
            if fsz <= 0:
                trades.append({
                    "symbol": sym, "side": "reject_open_long", "sz": 0, "px": px, "pnl": 0,
                    "note": note, "signal_px": px, "fill_mode": fmeta.get("mode"),
                    "slip_bps": fmeta.get("slip_bps"), "latency_ms": fmeta.get("latency_ms"),
                    "fill_reason": fmeta.get("reason"), "levels_eaten": fmeta.get("levels_eaten"),
                    "strategy": strat, "dir": "long",
                })
                return 0.0, float(px)
            fee = _fee(fsz, fill)
            new_sz = pos.long_sz + fsz
            if pos.long_sz > 0:
                pos.entry_long = (pos.entry_long * pos.long_sz + fill * fsz) / new_sz
            else:
                pos.entry_long = fill
            pos.long_sz = new_sz
            pos.strategy = strat
            pos.realized -= fee
            trades.append({
                "symbol": sym, "side": "open_long", "sz": fsz, "px": fill, "pnl": -fee, "note": note,
                "signal_px": px, "fill_mode": fmeta.get("mode"), "slip_bps": fmeta.get("slip_bps"),
                "latency_ms": fmeta.get("latency_ms"), "fill_reason": fmeta.get("reason"),
                "levels_eaten": fmeta.get("levels_eaten"), "book_mode": fmeta.get("book_mode"),
                "strategy": strat, "dir": "long",
            })
            _stamp_ts()
            return float(fsz), float(fill)
        elif action in ("open_short", "hft_short", "scale_in") and side == "short":
            fill, fsz, fmeta = self._resolve_fill(sym, False, sz, px, ts, mkey, ord_type, limit_px)
            if fsz <= 0:
                trades.append({
                    "symbol": sym, "side": "reject_open_short", "sz": 0, "px": px, "pnl": 0,
                    "note": note, "signal_px": px, "fill_mode": fmeta.get("mode"),
                    "fill_reason": fmeta.get("reason"), "latency_ms": fmeta.get("latency_ms"),
                    "strategy": strat, "dir": "short",
                })
                return 0.0, float(px)
            fee = _fee(fsz, fill)
            new_sz = pos.short_sz + fsz
            if pos.short_sz > 0:
                pos.entry_short = (pos.entry_short * pos.short_sz + fill * fsz) / new_sz
            else:
                pos.entry_short = fill
            pos.short_sz = new_sz
            pos.strategy = strat
            pos.realized -= fee
            trades.append({
                "symbol": sym, "side": "open_short", "sz": fsz, "px": fill, "pnl": -fee, "note": note,
                "signal_px": px, "fill_mode": fmeta.get("mode"), "slip_bps": fmeta.get("slip_bps"),
                "latency_ms": fmeta.get("latency_ms"), "fill_reason": fmeta.get("reason"),
                "levels_eaten": fmeta.get("levels_eaten"), "book_mode": fmeta.get("book_mode"),
                "strategy": strat, "dir": "short",
            })
            _stamp_ts()
            return float(fsz), float(fill)
        elif action in ("close", "partial_close") and side == "long" and pos.long_sz > 0:
            close_sz = min(sz, pos.long_sz)
            fill, fsz, fmeta = self._resolve_fill(sym, False, close_sz, px, ts, mkey, ord_type, limit_px)
            if fsz <= 0:
                trades.append({
                    "symbol": sym, "side": "reject_close_long", "sz": 0, "px": px, "pnl": 0,
                    "note": note, "fill_reason": fmeta.get("reason"), "latency_ms": fmeta.get("latency_ms"),
                    "strategy": strat, "dir": "long",
                })
                return 0.0, float(px)
            fee = _fee(fsz, fill)
            pnl = (fill - pos.entry_long) * fsz * ctv - fee
            pos.realized += pnl
            pos.long_sz -= fsz
            if pos.long_sz <= 1e-12:
                pos.long_sz = 0.0
                pos.entry_long = 0.0
                if pos.short_sz <= 1e-12:
                    pos.strategy = ""
            trades.append({
                "symbol": sym, "side": "close_long", "sz": fsz, "px": fill, "pnl": pnl, "note": note,
                "signal_px": px, "fill_mode": fmeta.get("mode"), "slip_bps": fmeta.get("slip_bps"),
                "latency_ms": fmeta.get("latency_ms"), "fill_reason": fmeta.get("reason"),
                "levels_eaten": fmeta.get("levels_eaten"), "book_mode": fmeta.get("book_mode"),
                "strategy": strat, "dir": "long",
            })
            _stamp_ts()
            return float(fsz), float(fill)
        elif action in ("close", "partial_close") and side == "short" and pos.short_sz > 0:
            close_sz = min(sz, pos.short_sz)
            fill, fsz, fmeta = self._resolve_fill(sym, True, close_sz, px, ts, mkey, ord_type, limit_px)
            if fsz <= 0:
                trades.append({
                    "symbol": sym, "side": "reject_close_short", "sz": 0, "px": px, "pnl": 0,
                    "note": note, "fill_reason": fmeta.get("reason"), "latency_ms": fmeta.get("latency_ms"),
                    "strategy": strat, "dir": "short",
                })
                return 0.0, float(px)
            fee = _fee(fsz, fill)
            pnl = (pos.entry_short - fill) * fsz * ctv - fee
            pos.realized += pnl
            pos.short_sz -= fsz
            if pos.short_sz <= 1e-12:
                pos.short_sz = 0.0
                pos.entry_short = 0.0
                if pos.long_sz <= 1e-12:
                    pos.strategy = ""
            trades.append({
                "symbol": sym, "side": "close_short", "sz": fsz, "px": fill, "pnl": pnl, "note": note,
                "signal_px": px, "fill_mode": fmeta.get("mode"), "slip_bps": fmeta.get("slip_bps"),
                "latency_ms": fmeta.get("latency_ms"), "fill_reason": fmeta.get("reason"),
                "levels_eaten": fmeta.get("levels_eaten"), "book_mode": fmeta.get("book_mode"),
                "strategy": strat, "dir": "short",
            })
            _stamp_ts()
            return float(fsz), float(fill)
        return 0.0, float(px)

    def _apply_funding(self, ts, positions, fetcher, trades) -> None:
        """——资金费。每逢结算时刻，按 notional * rate * direction 计入 realized。"""
        for s in self.symbols:
            pos = positions[s]
            if pos.long_sz <= 0 and pos.short_sz <= 0:
                continue
            try:
                rate = float(fetcher.get_funding_rate(s) or 0.0)
            except Exception:
                rate = 0.0
            if abs(rate) < 1e-12:
                continue
            ctv = self._ct_val(s)
            px = fetcher.get_last_price(s)
            if px <= 0:
                continue
            if pos.long_sz > 0:
                notional = pos.long_sz * ctv * px
                cost = notional * rate          # 正费率 → 多方向付费
                pos.realized -= cost
                trades.append({"symbol": s, "side": "funding", "sz": pos.long_sz, "px": px,
                               "pnl": -cost, "note": "funding_long", "strategy": "funding", "dir": "funding",
                               "ts": str(pd.Timestamp(ts))})
            if pos.short_sz > 0:
                notional = pos.short_sz * ctv * px
                gain = notional * rate           # 正费率 → 空方向收
                pos.realized += gain
                trades.append({"symbol": s, "side": "funding", "sz": pos.short_sz, "px": px,
                               "pnl": gain, "note": "funding_short", "strategy": "funding", "dir": "funding",
                               "ts": str(pd.Timestamp(ts))})

    def run(self, is_ratio: float = 0.7, phase_filter: str = "all") -> Tuple[MultiBacktestResult, MultiBacktestResult]:
        store = self._load_store()
        self._store = store
        timeline = self._aligned_index(store)
        n = len(timeline)
        split = int(n * is_ratio)
        split = max(80, min(split, n - 40))

        # 防过拟合：IS/OOS 之间留 embargo 空隙
        st_cfg = self.config.get("strategy") or {}
        if "bt_embargo_bars" in st_cfg:
            embargo_bars = max(0, int(st_cfg.get("bt_embargo_bars") or 0))
        else:
            day_map = {
                "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
                "30m": 48, "1H": 24, "2H": 12, "4H": 6, "1D": 1,
            }
            embargo_bars = max(1, int(day_map.get(self.main_bar, 96) * 0.5))

        is_end_i = max(0, split - 1)
        oos_start_i = min(n - 1, split + embargo_bars)
        log.info(
            f"走步分割 IS→{timeline[is_end_i]} | embargo={embargo_bars} bars | "
            f"OOS←{timeline[oos_start_i]} → {timeline[-1]}"
        )
        res_is = self._run_phase(timeline[0], timeline[is_end_i], "in_sample_70pct")
        res_oos = self._run_phase(timeline[oos_start_i], timeline[-1], "out_of_sample_30pct")
        return res_is, res_oos

    def _run_phase(self, t0, t1, phase: str) -> MultiBacktestResult:
        """跑单个时间段（P3 walk-forward 复用）。"""
        store = self._store
        st = (self.config.get("strategy") or {})
        fetcher = ParquetBarFetcher(store)
        fetcher.set_main_bar(self.main_bar)
        fetcher.signal_on_closed_bar = bool(st.get("signal_on_closed_bar", True))
        # 资金费率缓存（需 tools/build_funding_history.py 入库）
        try:
            from backtest.funding_store import preload_cache, rate_at, DB_PATH as FDB
            if not hasattr(self, "_funding_cache") or self._funding_cache is None:
                self._funding_cache = preload_cache(self.symbols, db_path=FDB)
            def _gf(inst_id, _cache=self._funding_cache):
                main = store[inst_id][self.main_bar]
                ii = fetcher._i.get(inst_id, 0)
                ts = main["ts"].iloc[min(max(ii, 0), len(main) - 1)]
                return rate_at(inst_id, pd.Timestamp(ts), cache=_cache)
            fetcher.get_funding_rate = _gf  # type: ignore
        except Exception as _e:
            log.warning(f"资金费率缓存未加载: {_e}")
        # 每品种独立 v3 引擎（状态隔离）——直接使用 v3.engine.StrategyEngine
        cfg_bt = dict(self.config)
        st_bt = dict(cfg_bt.get("strategy") or {})
        st_bt["signal_bar"] = self.main_bar
        st_bt["main_bar"] = self.main_bar
        st_bt["_main_bar"] = self.main_bar
        cfg_bt["strategy"] = st_bt
        engines = {s: StrategyEngine(cfg_bt) for s in self.symbols}
        positions = {s: SimPos() for s in self.symbols}
        total_cap = float(self.config.get("capital_usdt", 10000))
        # ---- v7 风控：把 start_equity 注入每个引擎（首根 K 线 update_equity 也会再次确认）----
        for _s, _e in engines.items():
            if hasattr(_e, "start_equity"):
                _e.start_equity = total_cap
        alloc = allocate_capital(
            total_cap * (1 - float(self.config.get("risk", {}).get("cash_reserve_ratio", 0.18))),
            self.symbols,
            mode=(self.config.get("universe") or {}).get("alloc_mode", "equal"),
        )
        equity0 = total_cap
        curve = []
        trades: List[Dict] = []
        signal_stats = {"hold": 0, "open": 0, "close": 0, "other": 0}
        reason_top = {}
        # 索引映射
        main_dfs = {s: store[s][self.main_bar] for s in self.symbols}
        self._main_dfs = main_dfs
        ts_arrays = {
            s: pd.to_datetime(main_dfs[s]["ts"]).values.astype("datetime64[ns]")
            for s in self.symbols
        }
        t0ns = np.datetime64(pd.Timestamp(t0).to_datetime64())
        t1ns = np.datetime64(pd.Timestamp(t1).to_datetime64())
        ref = self.symbols[0]
        ref_ts = ts_arrays[ref]
        i0 = int(np.searchsorted(ref_ts, t0ns, side="left"))
        i1 = int(np.searchsorted(ref_ts, t1ns, side="right") - 1)
        i0 = max(0, i0)
        i1 = min(len(ref_ts) - 1, i1)

        only_trend = (not bool(st.get("enable_vol_break", True)))
        signal_stride = 15 if only_trend and self.main_bar.endswith("1m") else 1

        # —— 回测真实性配置（P1）——
        exec_at_open = bool(st.get("bt_exec_at_open", True))   # next-bar-open 成交
        funding_hours = list(st.get("funding_settlement_hours", [0, 8, 16]))
        # —— 组合净敞口——
        net_delta_enabled = bool(st.get("net_delta_enabled", False))
        net_delta_limit_mult = float(st.get("net_delta_limit_mult", 1.8))
        # —— 信号周期 / HTF 回看长度 ——
        signal_lookback = int(st.get("signal_lookback", 160))
        htf_bar = str(st.get("htf_bar", "4H"))
        htf_lookback = int(st.get("htf_lookback", 90))
        leverage = float(self.config.get("leverage", 6.0))

        total_steps = max(1, i1 - i0 + 1)
        pbar = ProgressBar(total=total_steps, desc=f"BT {phase}")
        log_every = max(1, total_steps // 20)

        for step_i, i_ref in enumerate(range(i0, i1 + 1)):
            ts = ref_ts[i_ref]
            if step_i % log_every == 0 or step_i == total_steps - 1:
                pbar.set(step_i + 1, suffix=f"eq≈{equity0:.0f}")
            for s in self.symbols:
                j = int(np.searchsorted(ts_arrays[s], ts, side="right") - 1)
                if j < 0:
                    continue
                fetcher.set_cursor(s, j)
                self._bar_cursor[s] = j

            # 本根K线开盘前权益（已实现+浮动）
            mark_pre = 0.0
            for s in self.symbols:
                pos = positions[s]
                px0 = fetcher.get_last_price(s)
                ctv = self._ct_val(s)
                mark_pre += pos.realized
                if px0 > 0 and pos.long_sz > 0:
                    mark_pre += (px0 - pos.entry_long) * pos.long_sz * ctv
                if px0 > 0 and pos.short_sz > 0:
                    mark_pre += (pos.entry_short - px0) * pos.short_sz * ctv
            live_eq_bar = equity0 + mark_pre

            # 组合净敞口（beta 折算名义 USD）
            cur_delta = 0.0
            for s in self.symbols:
                pos = positions[s]
                ctv = self._ct_val(s)
                px = fetcher.get_last_price(s)
                b = _BETA.get(s.split("-")[0], 1.0)
                if px > 0 and pos.long_sz > 0:
                    cur_delta += b * pos.long_sz * ctv * px
                if px > 0 and pos.short_sz > 0:
                    cur_delta -= b * pos.short_sz * ctv * px

            mark = 0.0
            for s in self.symbols:
                eng = engines[s]
                pos = positions[s]
                px = fetcher.get_last_price(s)
                if px <= 0:
                    continue
                cap = float(alloc.get(s, total_cap / len(self.symbols)))
                if hasattr(eng, "set_sim_time"):
                    eng.set_sim_time(float(pd.Timestamp(ts).timestamp()))
                has_pos = pos.long_sz > 0 or pos.short_sz > 0
                # 空仓且非评估点：跳过信号计算（大加速）
                if (not has_pos) and signal_stride > 1 and (i_ref % signal_stride != 0):
                    ctv = self._ct_val(s)
                    if px > 0 and pos.long_sz > 0:
                        mark += (px - pos.entry_long) * pos.long_sz * ctv
                    if px > 0 and pos.short_sz > 0:
                        mark += (pos.entry_short - px) * pos.short_sz * ctv
                    mark += pos.realized
                    continue

                # —— 组装 v3 原生入参——
                df = fetcher.get_candles_df(s, self.main_bar, limit=signal_lookback)
                df_htf = None
                try:
                    df_htf = fetcher.get_candles_df(s, htf_bar, limit=htf_lookback)
                except Exception:
                    df_htf = None
                specs = fetcher.get_instrument_info(s)
                try:
                    funding_rate = float(fetcher.get_funding_rate(s) or 0.0)
                except Exception:
                    funding_rate = 0.0

                sig = eng.generate_signal(
                    s, df, df_htf, cap, leverage, specs, funding_rate=funding_rate,
                )
                act = sig.action
                if act == "hold":
                    signal_stats["hold"] += 1
                    r = (sig.reason or "")[:60]
                    reason_top[r] = reason_top.get(r, 0) + 1
                elif act in ("open_long", "open_short", "hft_long", "hft_short", "scale_in"):
                    signal_stats["open"] += 1
                elif act in ("close", "partial_close"):
                    signal_stats["close"] += 1
                else:
                    signal_stats["other"] += 1

                # 权益过低只允许平仓
                live_eq = live_eq_bar
                if live_eq < equity0 * 0.5 and act not in ("close", "partial_close", "hold"):
                    continue

                # 成交基准价：intrabar 保守价 > next-bar-open > 当前收盘
                bar_open = fetcher.get_bar_open(s)
                if sig.fill_price and sig.fill_price > 0:
                    exec_px = sig.fill_price
                elif exec_at_open and bar_open > 0:
                    exec_px = bar_open
                else:
                    exec_px = px

                if act in ("open_long", "hft_long", "open_short", "hft_short", "scale_in"):
                    ctv = self._ct_val(s)
                    max_notional = max(live_eq, 100) * 0.15
                    max_sz = max_notional / max(ctv * px, 1e-8)
                    sz = min(float(sig.size or 0), max_sz)
                    if sz <= 0:
                        continue
                    side = "long" if act in ("open_long", "hft_long") else "short"
                    # 净敞口约束
                    if net_delta_enabled:
                        b = _BETA.get(s.split("-")[0], 1.0)
                        dsign = 1.0 if side == "long" else -1.0
                        new_delta = dsign * b * sz * ctv * exec_px
                        limit = net_delta_limit_mult * (max(live_eq, 100) * 0.15)
                        if abs(cur_delta + new_delta) > limit:
                            continue
                    # 新开仓（无旧仓）时记录本轮 realized 起点（开仓费之前）
                    was_flat = pos.long_sz <= 1e-12 and pos.short_sz <= 1e-12
                    realized_before = float(pos.realized)
                    ord_type, limit_px = self._execution_params(s, side == "long", exec_px)
                    fsz, fill = self._apply(pos, side, act, sz, exec_px, sig.reason,
                                            trades, s, mode=sig.strategy, ts=ts,
                                            ord_type=ord_type, limit_px=limit_px)
                    if fsz > 0:
                        if was_flat:
                            pos.rt_start_realized = realized_before
                        if trades and str(trades[-1].get("side", "")).startswith("open"):
                            ctv_a = self._ct_val(s)
                            sl = float(getattr(sig, "stop_loss", 0) or 0)
                            tp = float(getattr(sig, "take_profit", 0) or 0)
                            trades[-1]["ts"] = str(pd.Timestamp(ts))
                            trades[-1]["stop_loss"] = sl
                            trades[-1]["take_profit"] = tp
                            trades[-1]["ct_val"] = ctv_a
                            trades[-1]["signal_conf"] = float(getattr(sig, "confidence", 0) or 0)
                            trades[-1]["regime"] = str(getattr(sig, "regime", "") or "")
                            if sl > 0 and fill > 0:
                                sl_dist = abs(float(fill) - sl)
                                trades[-1]["sl_dist"] = sl_dist
                                trades[-1]["sl_pct"] = sl_dist / float(fill)
                                trades[-1]["planned_loss"] = float(fsz) * sl_dist * ctv_a
                                trades[-1]["notional"] = float(fsz) * float(fill) * ctv_a
                        eng.confirm_fill(s, fill, fsz, sig)
                        # 同 bar 滚动净敞口，避免多品种连续开仓超限
                        if net_delta_enabled:
                            b = _BETA.get(s.split("-")[0], 1.0)
                            dsign = 1.0 if side == "long" else -1.0
                            cur_delta += dsign * b * fsz * ctv * fill
                elif act == "partial_close":
                    ratio = float(getattr(sig, "close_ratio", 0) or 0.4)
                    if ratio <= 0:
                        ratio = 0.4
                    sz = (pos.long_sz if sig.direction == "long" else pos.short_sz) * ratio
                    # 平多是卖出；平空是买入。
                    ord_type, limit_px = self._execution_params(s, sig.direction == "short", exec_px)
                    fsz, fill = self._apply(pos, sig.direction, "partial_close", sz, exec_px,
                                            sig.reason, trades, s, mode=sig.strategy, ts=ts,
                                            ord_type=ord_type, limit_px=limit_px)
                    if fsz > 0:
                        if trades and "close" in str(trades[-1].get("side", "")):
                            trades[-1]["ts"] = str(pd.Timestamp(ts))
                            trades[-1]["close_reason"] = str(sig.reason or "")
                            # 写入 v3 诊断字段（mfe_r / mae_r）—— 来自引擎 v3 engine
                            try:
                                _e_pos = eng.positions.get(s)
                                if _e_pos is not None:
                                    trades[-1]["mfe_r"] = float(_e_pos.mfe_r)
                                    trades[-1]["mae_r"] = float(_e_pos.mae_r)
                                    if _e_pos.sl_dist > 0:
                                        # realized R = (fill - entry) / sl_dist (long) 反之 short
                                        _entry = _e_pos.entry_long if sig.direction == "long" else _e_pos.entry_short
                                        if _entry > 0:
                                            if sig.direction == "long":
                                                trades[-1]["pnl_r"] = (float(fill) - _entry) / _e_pos.sl_dist
                                            else:
                                                trades[-1]["pnl_r"] = (_entry - float(fill)) / _e_pos.sl_dist
                            except Exception:
                                pass

                        eng.confirm_partial_close(s, fsz)
                        if net_delta_enabled:
                            b = _BETA.get(s.split("-")[0], 1.0)
                            ctv_p = self._ct_val(s)
                            if sig.direction == "long":
                                cur_delta -= b * fsz * ctv_p * fill
                            else:
                                cur_delta += b * fsz * ctv_p * fill
                elif act == "close":
                    sz = pos.long_sz if sig.direction == "long" else pos.short_sz
                    side_dir = sig.direction
                    ord_type, limit_px = self._execution_params(s, side_dir == "short", exec_px)
                    fsz, fill = self._apply(pos, side_dir, "close", sz, exec_px,
                                            sig.reason, trades, s, mode=sig.strategy, ts=ts,
                                            ord_type=ord_type, limit_px=limit_px)
                    if fsz > 0:
                        if trades and str(trades[-1].get("side", "")).startswith("close"):
                            trades[-1]["ts"] = str(pd.Timestamp(ts))
                            trades[-1]["close_reason"] = str(sig.reason or "")
                            # 写入 v3 诊断字段（mfe_r / mae_r / pnl_r）—— 来自 v3 engine 的 Position 状态
                            try:
                                _e_pos = eng.positions.get(s)
                                if _e_pos is not None:
                                    trades[-1]["mfe_r"] = float(_e_pos.mfe_r)
                                    trades[-1]["mae_r"] = float(_e_pos.mae_r)
                                    if _e_pos.sl_dist > 0:
                                        _entry = _e_pos.entry_long if side_dir == "long" else _e_pos.entry_short
                                        if _entry > 0:
                                            if side_dir == "long":
                                                trades[-1]["pnl_r"] = (float(fill) - _entry) / _e_pos.sl_dist
                                            else:
                                                trades[-1]["pnl_r"] = (_entry - float(fill)) / _e_pos.sl_dist
                            except Exception:
                                pass

                        fully_closed = pos.long_sz <= 1e-12 and pos.short_sz <= 1e-12
                        if fully_closed and hasattr(eng, "note_trade_result"):
                            # 真实美元 PnL（整轮：开仓费+分批+最终平仓）
                            rt_pnl = float(pos.realized) - float(pos.rt_start_realized)
                            eng.note_trade_result(s, rt_pnl)
                        eng.confirm_close(s)
                        if net_delta_enabled and fsz > 0:
                            b = _BETA.get(s.split("-")[0], 1.0)
                            # 平多减少正敞口，平空减少负敞口
                            if side_dir == "long":
                                cur_delta -= b * fsz * self._ct_val(s) * fill
                            else:
                                cur_delta += b * fsz * self._ct_val(s) * fill

                # 未实现
                ctv = self._ct_val(s)
                upnl = 0.0
                if pos.long_sz > 0:
                    upnl += (px - pos.entry_long) * pos.long_sz * ctv
                if pos.short_sz > 0:
                    upnl += (pos.entry_short - px) * pos.short_sz * ctv
                mark += pos.realized + upnl

            # 资金费：结算时刻（默认 00/08/16 UTC）——先入账再记权益，避免晚一根 bar
            try:
                t_dt = pd.Timestamp(ts)
                is_settle = int(t_dt.minute) == 0 and int(t_dt.hour) in funding_hours
            except Exception:
                is_settle = False
            if is_settle:
                self._apply_funding(ts, positions, fetcher, trades)
                # funding 写入 realized 后重算 mark
                mark = 0.0
                for s in self.symbols:
                    pos = positions[s]
                    px = fetcher.get_last_price(s)
                    ctv = self._ct_val(s)
                    upnl = 0.0
                    if px > 0 and pos.long_sz > 0:
                        upnl += (px - pos.entry_long) * pos.long_sz * ctv
                    if px > 0 and pos.short_sz > 0:
                        upnl += (pos.entry_short - px) * pos.short_sz * ctv
                    mark += pos.realized + upnl

            equity = max(0.0, equity0 + mark)  # 回测不允许负权益展示（实盘会强平）
            # ---- v7 风控：每根 K 线末尾把当前权益推给引擎，触发 max_dd / equity_lock 检查 ----
            for _s, _e in engines.items():
                if hasattr(_e, "update_equity"):
                    _e.update_equity(float(equity))
            curve.append({"ts": str(pd.Timestamp(ts)), "equity": equity})
            if step_i % log_every == 0:
                pbar.set(step_i + 1, suffix=f"eq={equity:.0f} trades={len(trades)}")

        pbar.set(total_steps, suffix=f"eq={curve[-1]['equity']:.0f}" if curve else "")
        eq = pd.DataFrame(curve)
        if eq.empty:
            eq = pd.DataFrame([{"ts": t0, "equity": equity0}])
        eq_series = eq["equity"].astype(float)
        if len(eq_series) > 0 and abs(float(eq_series.iloc[0]) - equity0) > 1e-6:
            eq_series = pd.concat([pd.Series([equity0]), eq_series], ignore_index=True)
        # 样本期末强制平仓，使 open/close 与 PnL 对齐
        ts_last = ts
        for s in self.symbols:
            pos = positions[s]
            px = fetcher.get_last_price(s)
            if px <= 0:
                continue
            if pos.long_sz > 0:
                eod_mode = pos.strategy or "default"
                ord_type, limit_px = self._execution_params(s, False, px)
                fsz, fill = self._apply(pos, "long", "close", float(pos.long_sz), px, "eod_force_close",
                                        trades, s, mode=eod_mode, ts=ts_last,
                                        ord_type=ord_type, limit_px=limit_px)
                if fsz > 0 and s in engines:
                    if pos.long_sz <= 1e-12 and pos.short_sz <= 1e-12:
                        rt_pnl = float(pos.realized) - float(pos.rt_start_realized)
                        engines[s].note_trade_result(s, rt_pnl)
                    engines[s].confirm_close(s)
                signal_stats["close"] = signal_stats.get("close", 0) + 1
            if pos.short_sz > 0:
                eod_mode = pos.strategy or "default"
                ord_type, limit_px = self._execution_params(s, True, px)
                fsz, fill = self._apply(pos, "short", "close", float(pos.short_sz), px, "eod_force_close",
                                        trades, s, mode=eod_mode, ts=ts_last,
                                        ord_type=ord_type, limit_px=limit_px)
                if fsz > 0 and s in engines:
                    if pos.long_sz <= 1e-12 and pos.short_sz <= 1e-12:
                        rt_pnl = float(pos.realized) - float(pos.rt_start_realized)
                        engines[s].note_trade_result(s, rt_pnl)
                    engines[s].confirm_close(s)
                signal_stats["close"] = signal_stats.get("close", 0) + 1
        mark_end = sum(positions[s].realized for s in self.symbols)
        if len(eq_series) > 0:
            eq_series = eq_series.copy()
            eq_series.iloc[-1] = equity0 + mark_end
        else:
            eq_series = pd.Series([equity0, equity0 + mark_end])
        bpy_map = {
            "1s": 365 * 24 * 3600, "1m": 365 * 24 * 60, "3m": 365 * 24 * 20,
            "5m": 365 * 24 * 12, "15m": 365 * 24 * 4, "30m": 365 * 24 * 2,
            "1H": 365 * 24, "2H": 365 * 12, "4H": 365 * 6, "1D": 365,
        }
        bpy = float(bpy_map.get(self.main_bar, 365 * 24 * 60))
        m = compute_metrics(eq_series, trades, bars_per_year=bpy)
        m["phase"] = phase
        m["symbols"] = ",".join(self.symbols)
        # trade_count 已由 compute_metrics 按 round-trip 统计；此处保留 close leg 诊断
        m["close_leg_count"] = m.get("close_leg_count") or len(
            [x for x in trades if str(x.get("side", "")).startswith("close")]
        )
        m["signal_stats"] = signal_stats
        m["main_bar"] = self.main_bar
        # P3 归因拆分：per-strategy / long-short
        m["attribution"] = self._attribution(trades)
        top_reasons = sorted(reason_top.items(), key=lambda x: -x[1])[:5]
        log.info(f"[{phase}] 信号统计 {signal_stats} | hold原因TOP: {top_reasons}")
        return MultiBacktestResult(metrics=m, equity_curve=eq, trades=trades, phase=phase)

    @staticmethod
    def _attribution(trades: List[Dict]) -> Dict:
        """P3 归因：按策略 / 方向 / 是否 funding 拆分已实现 PnL 与笔数。"""
        out = {"per_strategy": {}, "per_direction": {"long": 0.0, "short": 0.0, "funding": 0.0}}
        for t in trades:
            side = str(t.get("side", ""))
            if side.startswith("open") or side.startswith("reject"):
                continue
            pnl = float(t.get("pnl", 0) or 0)
            strat = str(t.get("strategy", "default"))
            d = str(t.get("dir", "flat"))
            ps = out["per_strategy"].setdefault(strat, {"pnl": 0.0, "n": 0})
            ps["pnl"] += pnl
            ps["n"] += 1
            if d == "funding":
                out["per_direction"]["funding"] += pnl
            elif d == "long":
                out["per_direction"]["long"] += pnl
            elif d == "short":
                out["per_direction"]["short"] += pnl
        for k in out["per_strategy"]:
            v = out["per_strategy"][k]
            v["pnl"] = round(v["pnl"], 4)
        return out


def _symbol_data_ready(symbol: str, main_bar: str) -> bool:
    if db_has_symbol_bar(symbol, main_bar, DB_PATH):
        return True
    bar_file = f"bars_{main_bar}.parquet"
    p = os.path.join(PQ_ROOT, symbol, bar_file)
    return os.path.isfile(p) or os.path.isfile(p.replace(".parquet", ".pkl"))


def _db_inventory_msg() -> str:
    """诊断：绝对路径 + meta 里已有的 symbol/bar。"""
    abs_db = os.path.abspath(DB_PATH)
    lines = [f"  DB 绝对路径: {abs_db}", f"  DB 存在: {os.path.isfile(DB_PATH)}"]
    try:
        from backtest.trade_pipeline import list_db_coverage
        cov = list_db_coverage(DB_PATH)
        if cov is None or cov.empty:
            lines.append("  meta 表为空（或无法读取）")
        else:
            lines.append("  meta 已有:")
            for _, r in cov.iterrows():
                lines.append(
                    f"    {r.get('symbol')} {r.get('bar')} n={r.get('n_bars')} "
                    f"{r.get('ts_min')} → {r.get('ts_max')}"
                )
    except Exception as e:
        lines.append(f"  读取 meta 失败: {e}")
    lines.append(f"  parquet 根: {os.path.abspath(PQ_ROOT)}")
    return "\n".join(lines)


def run_multi_from_parquet(
    config: Dict,
    symbols: Optional[List[str]] = None,
    is_ratio: float = 0.7,
    main_bar: str = "1m",
    out_dir: str = "data/backtest_results",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    # out_dir 相对项目根，避免 cwd 漂移
    if not os.path.isabs(out_dir):
        from backtest.trade_pipeline import _ROOT
        out_dir = os.path.join(_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    u = config.get("universe") or {}
    if not symbols:
        prefer = list(u.get("prefer_only") or [])
        max_n = int(u.get("max_symbols", 4))
        symbols = (prefer or ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"])[:max_n]

    log.info(f"数据探测 DB={os.path.abspath(DB_PATH)} exists={os.path.isfile(DB_PATH)} bar={main_bar}")
    ready = []
    for s in symbols:
        if _symbol_data_ready(s, main_bar):
            ready.append(s)
        else:
            log.warning(f"跳过无数据品种 {s}（DB/parquet 均无 {main_bar}）")
    if not ready:
        raise FileNotFoundError(
            "没有任何品种的 K 线数据。请先：\n"
            "  python tools/data_manager.py --symbols BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP "
            "--bars 1m,5m,15m,1H,4H --days 180\n"
            f"  支持周期: {', '.join(COMMON_BARS)}\n"
            + _db_inventory_msg()
        )

    ts_start = pd.Timestamp(start) if start else None
    ts_end = pd.Timestamp(end) if end else None
    log.info(
        f"多币种回测 symbols={ready} bar={main_bar} "
        f"range={ts_start or 'min'}→{ts_end or 'max'} IS={is_ratio:.0%}"
    )
    bt = MultiSymbolBacktester(
        config, ready, main_bar=main_bar, start=ts_start, end=ts_end
    )
    res_is, res_oos = bt.run(is_ratio=is_ratio)
    is_m = res_is.metrics
    oos_m = res_oos.metrics

    overfit = check_overfitting(is_m, oos_m)
    log.info(
        f"过拟合检查 ok={overfit['ok']} score={overfit['overfit_score']} | {overfit['advice']}"
    )
    for w in overfit.get("warnings") or []:
        log.warning(f"[overfit] {w}")

    res_is.equity_curve.to_csv(os.path.join(out_dir, "MULTI_IS_equity.csv"), index=False)
    res_oos.equity_curve.to_csv(os.path.join(out_dir, "MULTI_OOS_equity.csv"), index=False)
    pd.DataFrame(res_is.trades).to_csv(os.path.join(out_dir, "MULTI_IS_trades.csv"), index=False)
    pd.DataFrame(res_oos.trades).to_csv(os.path.join(out_dir, "MULTI_OOS_trades.csv"), index=False)
    cmp = metrics_to_frame(res_is.metrics, res_oos.metrics)
    cmp.to_csv(os.path.join(out_dir, "MULTI_metrics_compare.csv"), index=False)
    with open(os.path.join(out_dir, "MULTI_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "in_sample": res_is.metrics,
                "out_of_sample": res_oos.metrics,
                "overfit": overfit,
                "range": {"start": str(ts_start) if ts_start else None, "end": str(ts_end) if ts_end else None},
                "main_bar": main_bar,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    from v3.utils.run_meta import new_run_meta, write_run_meta
    meta = new_run_meta(
        config,
        strategy=",".join((config.get("strategy") or {}).get("enabled_strategies") or []),
        symbol=",".join(symbols or []),
        bar=main_bar,
        extra={
            "out_dir": out_dir,
            "is_trade_count": int(is_m.get("trade_count") or 0),
            "oos_trade_count": int(oos_m.get("trade_count") or 0),
            "is_total_pnl": float(is_m.get("total_pnl") or 0),
            "oos_total_pnl": float(oos_m.get("total_pnl") or 0),
        },
    )
    write_run_meta(os.path.join(out_dir, "MULTI_run_meta.json"), meta)
    is_m["run_id"] = meta["run_id"]
    oos_m["run_id"] = meta["run_id"]
    is_m["config_hash"] = meta["config_hash"]
    oos_m["config_hash"] = meta["config_hash"]

    return is_m, oos_m, cmp
