# -*- coding: utf-8 -*-
"""回测用 DataFetcher：按时间游标切片历史K线，不访问网络。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.data_store import CandleStore, default_spec


class HistoricalDataFetcher:
    def __init__(self, store: CandleStore, equity0: float = 10000.0):
        self.store = store
        self.cursor: Optional[pd.Timestamp] = None
        self.equity = float(equity0)
        self._pos: Dict[str, Dict[str, float]] = {}  # inst -> long/short size

    def set_cursor(self, ts: pd.Timestamp) -> None:
        self.cursor = pd.Timestamp(ts)

    def set_equity(self, eq: float) -> None:
        self.equity = float(eq)

    def set_position(self, inst_id: str, long_sz: float = 0.0, short_sz: float = 0.0) -> None:
        self._pos[inst_id] = {"long": float(long_sz), "short": float(short_sz)}

    def _slice(self, inst_id: str, bar: str, limit: int) -> pd.DataFrame:
        df = self.store.get(inst_id, bar)
        if df is None or df.empty or self.cursor is None:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
        sub = df[df["ts"] <= self.cursor]
        if sub.empty:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
        return sub.tail(limit).reset_index(drop=True)

    def get_instrument_info(self, inst_id: str) -> Dict:
        spec = default_spec(inst_id)
        return {
            "instId": inst_id,
            "ctVal": str(spec["ctVal"]),
            "lotSz": str(spec["lotSz"]),
            "minSz": str(spec["minSz"]),
            "tickSz": str(spec["tickSz"]),
            "state": "live",
        }

    def get_ticker(self, inst_id: str) -> Dict:
        px = self.get_last_price(inst_id)
        return {"last": str(px), "instId": inst_id}

    def get_last_price(self, inst_id: str) -> float:
        # 优先 15m，再 1m
        for bar in ("15m", "1m", "1H", "5m"):
            df = self._slice(inst_id, bar, 2)
            if not df.empty:
                return float(df["close"].iloc[-1])
        return 0.0

    def get_candles_df(self, inst_id: str, bar: str = "15m", limit: int = 150) -> pd.DataFrame:
        return self._slice(inst_id, bar, limit)

    def get_multi_timeframe_candles(
        self, inst_id: str, bars: List[str] = None, limit: int = 120
    ) -> Dict[str, pd.DataFrame]:
        bars = bars or ["5m", "15m", "1H", "4H"]
        return {b: self.get_candles_df(inst_id, bar=b, limit=limit) for b in bars}

    def get_fee_rates(self, inst_id: str) -> Tuple[float, float]:
        return 0.0002, 0.0005

    def get_funding_rate(self, inst_id: str) -> float:
        # 回测无资金费率序列时返回中性
        return 0.0

    def get_account_usdt_equity(self) -> float:
        return self.equity

    def get_position(self, inst_id: str) -> Dict:
        p = self._pos.get(inst_id, {})
        lo, sh = p.get("long", 0.0), p.get("short", 0.0)
        if lo > 0:
            return {"instId": inst_id, "pos": str(lo), "posSide": "long"}
        if sh > 0:
            return {"instId": inst_id, "pos": str(-sh), "posSide": "short"}
        return {}

    def get_order_book_imbalance(self, inst_id: str, depth: int = 5) -> float:
        """用近几根成交量与涨跌近似不平衡（回测代理）。"""
        df = self._slice(inst_id, "1m", 10)
        if len(df) < 5:
            return 0.0
        chg = df["close"].pct_change().fillna(0)
        vol = df["vol"].replace(0, np.nan)
        up = float((vol * (chg > 0)).sum())
        dn = float((vol * (chg < 0)).sum())
        s = up + dn
        if s <= 0:
            return 0.0
        return float(np.clip((up - dn) / s, -1, 1))

    def get_cvd_proxy(self, inst_id: str, limit: int = 100) -> float:
        df = self._slice(inst_id, "1m", min(limit, 80))
        if len(df) < 5:
            return 0.0
        chg = df["close"].diff().fillna(0)
        signed = np.sign(chg) * df["vol"]
        s = float(df["vol"].sum())
        if s <= 0:
            return 0.0
        return float(np.clip(signed.sum() / s, -1, 1))

    # 兼容 strategy 里偶发的 self.fetcher.client
    @property
    def client(self):
        return self

    def get_open_interest(self, inst_type: str = "SWAP", inst_id: str = "") -> List[Dict]:
        return [{"oi": "0", "instId": inst_id}]
