# -*- coding: utf-8 -*-
"""
[已弃用] 单品种回测引擎。

主路径请使用 backtest.multi_engine.MultiSymbolBacktester /
`python -m v2.run_backtest`。本文件仅保留兼容，不再修复新功能。

按 15m 主轴推进时间，调用 StrategyEngine 生成信号并模拟成交。
默认手续费 taker 0.05%，滑点 1 tick。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtest.data_store import CandleStore, default_spec, prepare_store
from backtest.hist_fetcher import HistoricalDataFetcher
from backtest.metrics import compute_metrics, metrics_to_frame
from logger import setup_logger
from strategy import StrategyEngine  # v2 引擎（P0-1 统一接口）
from v2.strategies.base import Signal as TradeSignal

log = setup_logger("bt_engine")


@dataclass
class Position:
    long_sz: float = 0.0
    short_sz: float = 0.0
    entry_long: float = 0.0
    entry_short: float = 0.0


@dataclass
class BacktestResult:
    phase: str
    equity_curve: pd.DataFrame
    trades: List[Dict]
    metrics: Dict
    config_snapshot: Dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        config: Dict,
        store: CandleStore,
        symbol: str,
        fee_rate: float = 0.0005,
        slippage_bps: float = 1.0,
    ):
        log.warning(
            "BacktestEngine(单品种) 已弃用，请改用 multi_engine / python -m v2.run_backtest"
        )
        self.config = config
        self.store = store
        self.symbol = symbol
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.capital0 = float(config.get("capital_usdt", 10000))
        self.fetcher = HistoricalDataFetcher(store, equity0=self.capital0)
        # v2 引擎（P0-1）：不再传入 fetcher，df 由 run() 组装后直传
        self.strategy = StrategyEngine(config)
        self.signal_bar = str((config.get("strategy") or {}).get("signal_bar", "1H"))
        self.htf_bar = str((config.get("strategy") or {}).get("htf_bar", "4H"))
        self.leverage = float(config.get("leverage", 6.0))
        self.spec = default_spec(symbol)
        self.pos = Position()
        self.equity = self.capital0
        self.trades: List[Dict] = []
        self.equity_rows: List[Dict] = []

    def _price_at(self, ts: pd.Timestamp) -> float:
        self.fetcher.set_cursor(ts)
        return self.fetcher.get_last_price(self.symbol)

    def _apply_slip(self, price: float, side: str) -> float:
        # buy 买贵，sell 卖便宜
        slip = price * (self.slippage_bps / 10000.0)
        return price + slip if side == "buy" else price - slip

    def _notional(self, sz: float, price: float) -> float:
        return abs(sz) * self.spec["ctVal"] * price

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

    def _close_long(self, price: float, reason: str, ts) -> None:
        if self.pos.long_sz <= 0:
            return
        px = self._apply_slip(price, "sell")
        n = self._notional(self.pos.long_sz, px)
        fee = self._fee(n)
        pnl = (px - self.pos.entry_long) / self.pos.entry_long * n - fee if self.pos.entry_long > 0 else -fee
        self.equity += pnl
        self.trades.append(
            {
                "ts": str(ts),
                "side": "sell",
                "posSide": "long",
                "price": px,
                "size": self.pos.long_sz,
                "fee": fee,
                "pnl": pnl,
                "reason": reason,
                "equity": self.equity,
            }
        )
        self.pos.long_sz = 0.0
        self.pos.entry_long = 0.0
        try:
            self.strategy.note_trade_result(self.symbol, pnl)
        except Exception:
            pass

    def _close_short(self, price: float, reason: str, ts) -> None:
        if self.pos.short_sz <= 0:
            return
        px = self._apply_slip(price, "buy")
        n = self._notional(self.pos.short_sz, px)
        fee = self._fee(n)
        pnl = (self.pos.entry_short - px) / self.pos.entry_short * n - fee if self.pos.entry_short > 0 else -fee
        self.equity += pnl
        self.trades.append(
            {
                "ts": str(ts),
                "side": "buy",
                "posSide": "short",
                "price": px,
                "size": self.pos.short_sz,
                "fee": fee,
                "pnl": pnl,
                "reason": reason,
                "equity": self.equity,
            }
        )
        self.pos.short_sz = 0.0
        self.pos.entry_short = 0.0
        try:
            self.strategy.note_trade_result(self.symbol, pnl)
        except Exception:
            pass

    def _open_long(self, size: float, price: float, reason: str, ts) -> None:
        if size <= 0:
            return
        px = self._apply_slip(price, "buy")
        n = self._notional(size, px)
        fee = self._fee(n)
        self.equity -= fee
        # 加仓均价
        if self.pos.long_sz > 0 and self.pos.entry_long > 0:
            total = self.pos.long_sz + size
            self.pos.entry_long = (self.pos.entry_long * self.pos.long_sz + px * size) / total
            self.pos.long_sz = total
        else:
            self.pos.long_sz = size
            self.pos.entry_long = px
        self.trades.append(
            {
                "ts": str(ts),
                "side": "buy",
                "posSide": "long",
                "price": px,
                "size": size,
                "fee": fee,
                "pnl": -fee,
                "reason": reason,
                "equity": self.equity,
            }
        )

    def _open_short(self, size: float, price: float, reason: str, ts) -> None:
        if size <= 0:
            return
        px = self._apply_slip(price, "sell")
        n = self._notional(size, px)
        fee = self._fee(n)
        self.equity -= fee
        if self.pos.short_sz > 0 and self.pos.entry_short > 0:
            total = self.pos.short_sz + size
            self.pos.entry_short = (self.pos.entry_short * self.pos.short_sz + px * size) / total
            self.pos.short_sz = total
        else:
            self.pos.short_sz = size
            self.pos.entry_short = px
        self.trades.append(
            {
                "ts": str(ts),
                "side": "sell",
                "posSide": "short",
                "price": px,
                "size": size,
                "fee": fee,
                "pnl": -fee,
                "reason": reason,
                "equity": self.equity,
            }
        )

    def _mark_equity(self, price: float) -> float:
        """权益 = 现金口径 + 未实现盈亏（简化：开仓费已扣，按均价浮盈）"""
        upnl = 0.0
        if self.pos.long_sz > 0 and self.pos.entry_long > 0:
            n = self._notional(self.pos.long_sz, price)
            upnl += (price - self.pos.entry_long) / self.pos.entry_long * n
        if self.pos.short_sz > 0 and self.pos.entry_short > 0:
            n = self._notional(self.pos.short_sz, price)
            upnl += (self.pos.entry_short - price) / self.pos.entry_short * n
        return self.equity + upnl

    def _execute(self, sig: TradeSignal, price: float, ts) -> None:
        """v2 Signal 执行（P0-1）。成交后同步回 v2 引擎（confirm_*）。"""
        act = sig.action
        if act == "hold":
            return
        if act == "close":
            if sig.direction == "long":
                self._close_long(price, sig.reason, ts)
            elif sig.direction == "short":
                self._close_short(price, sig.reason, ts)
            self.strategy.confirm_close(self.symbol)
            return
        if act == "partial_close":
            ratio = float(getattr(sig, "close_ratio", 0) or 0.4)
            if ratio <= 0:
                ratio = 0.4
            sz = (self.pos.long_sz if sig.direction == "long" else self.pos.short_sz) * ratio
            if sig.direction == "long":
                self._close_long_ratio(sz, price, sig.reason, ts)
            else:
                self._close_short_ratio(sz, price, sig.reason, ts)
            self.strategy.confirm_partial_close(self.symbol, sz)
            return
        if act in ("open_long", "hft_long"):
            if self.pos.short_sz > 0 and getattr(sig, "strategy", "") != "hedge":
                self._close_short(price, "开多前平空", ts)
            sz = float(sig.size or 0)
            if sz > 0:
                self._open_long(sz, price, sig.reason, ts)
                self.strategy.confirm_fill(self.symbol, price, sz, sig)
            return
        if act in ("open_short", "hft_short"):
            if self.pos.long_sz > 0 and getattr(sig, "strategy", "") != "hedge":
                self._close_long(price, "开空前平多", ts)
            sz = float(sig.size or 0)
            if sz > 0:
                self._open_short(sz, price, sig.reason, ts)
                self.strategy.confirm_fill(self.symbol, price, sz, sig)

    def _close_long_ratio(self, size: float, price: float, reason: str, ts) -> None:
        if size <= 0 or self.pos.long_sz <= 0:
            return
        size = min(size, self.pos.long_sz)
        px = self._apply_slip(price, "sell")
        n = self._notional(size, px)
        fee = self._fee(n)
        pnl = (px - self.pos.entry_long) / self.pos.entry_long * n - fee if self.pos.entry_long > 0 else -fee
        self.equity += pnl
        self.trades.append({"ts": str(ts), "side": "sell", "posSide": "long", "price": px,
                            "size": size, "fee": fee, "pnl": pnl, "reason": reason, "equity": self.equity})
        self.pos.long_sz -= size
        if self.pos.long_sz <= 1e-12:
            self.pos.long_sz = 0.0
            self.pos.entry_long = 0.0

    def _close_short_ratio(self, size: float, price: float, reason: str, ts) -> None:
        if size <= 0 or self.pos.short_sz <= 0:
            return
        size = min(size, self.pos.short_sz)
        px = self._apply_slip(price, "buy")
        n = self._notional(size, px)
        fee = self._fee(n)
        pnl = (self.pos.entry_short - px) / self.pos.entry_short * n - fee if self.pos.entry_short > 0 else -fee
        self.equity += pnl
        self.trades.append({"ts": str(ts), "side": "buy", "posSide": "short", "price": px,
                            "size": size, "fee": fee, "pnl": pnl, "reason": reason, "equity": self.equity})
        self.pos.short_sz -= size
        if self.pos.short_sz <= 1e-12:
            self.pos.short_sz = 0.0
            self.pos.entry_short = 0.0

    def run(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        phase: str = "backtest",
        step_bar: str = "15m",
        warmup: int = 80,
    ) -> BacktestResult:
        df = self.store.get(self.symbol, step_bar)
        if df is None or len(df) < warmup + 10:
            raise RuntimeError(f"主周期数据不足: {self.symbol} {step_bar}")

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df["ts"] >= pd.Timestamp(start)
        if end is not None:
            mask &= df["ts"] <= pd.Timestamp(end)
        axis = df.loc[mask].reset_index(drop=True)
        if len(axis) < warmup + 5:
            raise RuntimeError(f"{phase} 区间K线过少: {len(axis)}")

        self.pos = Position()
        self.equity = self.capital0
        self.trades = []
        self.equity_rows = []
        self.fetcher.set_equity(self.equity)

        log.info(f"=== {phase} {self.symbol} bars={len(axis)} "
                 f"{axis['ts'].iloc[0]} → {axis['ts'].iloc[-1]} ===")

        for i in range(warmup, len(axis)):
            ts = axis["ts"].iloc[i]
            price = float(axis["close"].iloc[i])
            self.fetcher.set_cursor(ts)
            self.fetcher.set_equity(self.equity)
            self.fetcher.set_position(self.symbol, self.pos.long_sz, self.pos.short_sz)

            try:
                # P0-1：v2 原生接口
                df = self.fetcher.get_candles_df(self.symbol, bar=self.signal_bar, limit=160)
                df_htf = None
                try:
                    df_htf = self.fetcher.get_candles_df(self.symbol, bar=self.htf_bar, limit=90)
                except Exception:
                    df_htf = None
                specs = self.fetcher.get_instrument_info(self.symbol)
                try:
                    funding_rate = float(self.fetcher.get_funding_rate(self.symbol) or 0.0)
                except Exception:
                    funding_rate = 0.0
                sig = self.strategy.generate_signal(
                    self.symbol, df, df_htf, self.capital0, self.leverage, specs,
                    funding_rate=funding_rate,
                )
            except Exception as e:
                log.debug(f"信号异常 @ {ts}: {e}")
                sig = TradeSignal(action="hold", direction="flat", confidence=0.0, reason=str(e))

            exec_px = sig.fill_price if (getattr(sig, "fill_price", 0) or 0) > 0 else price
            self._execute(sig, exec_px, ts)
            mark = self._mark_equity(price)
            self.equity_rows.append(
                {
                    "ts": ts,
                    "equity": mark,
                    "cash_equity": self.equity,
                    "price": price,
                    "long": self.pos.long_sz,
                    "short": self.pos.short_sz,
                    "action": sig.action,
                    "score": getattr(sig, "confidence", 0.0) or 0.0,
                }
            )

        # 结束强平
        if len(axis):
            last_ts = axis["ts"].iloc[-1]
            last_px = float(axis["close"].iloc[-1])
            self._close_long(last_px, "回测结束平多", last_ts)
            self._close_short(last_px, "回测结束平空", last_ts)
            self.strategy.confirm_close(self.symbol)

        eq_df = pd.DataFrame(self.equity_rows)
        if eq_df.empty:
            eq_df = pd.DataFrame(columns=["ts", "equity", "price"])
        mets = compute_metrics(eq_df["equity"] if "equity" in eq_df else pd.Series([self.capital0]), self.trades)
        mets["phase"] = phase
        mets["symbol"] = self.symbol
        log.info(f"{phase} 完成: ret={mets.get('total_return')} maxDD={mets.get('max_drawdown')} "
                 f"trades={mets.get('trade_count', 0)}")
        return BacktestResult(
            phase=phase,
            equity_curve=eq_df,
            trades=self.trades,
            metrics=mets,
            config_snapshot={
                "symbol": self.symbol,
                "capital": self.capital0,
                "fee": self.fee_rate,
                "slippage_bps": self.slippage_bps,
            },
        )


def run_walk_forward(
    config: Dict,
    symbol: str = "BTC-USDT-SWAP",
    is_ratio: float = 0.7,
    data_dir: str = "data/backtest",
    out_dir: str = "data/backtest_results",
    use_synthetic: bool = False,
    fetch: bool = False,
    step_bar: str = "15m",
    max_batches: int = 0,
    days: float = 30.0,
    force_refetch: bool = False,
) -> Tuple[BacktestResult, BacktestResult, pd.DataFrame]:
    """
    70% 样本内回测 + 30% 样本外模拟实盘（同一套参数，不做偷窥优化）。
    """
    os.makedirs(out_dir, exist_ok=True)
    bars = ["5m", "15m", "1H", "4H"]
    log.info(f"加载周期 {bars} | v15 MOM+EMA+VOL | max_batches={max_batches}")
    store = prepare_store(
        [symbol],
        bars=bars,
        data_dir=data_dir,
        use_synthetic=use_synthetic,
        fetch=fetch,
        max_batches=max_batches,
        days=days,
        force_refetch=force_refetch,
    )
    df = store.get(symbol, step_bar)
    if df is None or len(df) < 200:
        raise RuntimeError("主数据不足，请先 --fetch 或使用 --synthetic")

    t0, t1 = df["ts"].iloc[0], df["ts"].iloc[-1]
    split_i = int(len(df) * is_ratio)
    split_i = max(split_i, 120)
    split_i = min(split_i, len(df) - 80)
    split_ts = df["ts"].iloc[split_i]
    log.info(f"时间范围 {t0} → {t1} | 分割点 {split_ts} (IS={is_ratio:.0%} OOS={1-is_ratio:.0%})")

    eng_is = BacktestEngine(config, store, symbol)
    res_is = eng_is.run(start=t0, end=split_ts, phase="in_sample_70pct", step_bar=step_bar)

    eng_oos = BacktestEngine(config, store, symbol)
    res_oos = eng_oos.run(
        start=df["ts"].iloc[split_i + 1],
        end=t1,
        phase="out_of_sample_30pct",
        step_bar=step_bar,
    )

    cmp = metrics_to_frame(res_is.metrics, res_oos.metrics)
    # 落盘
    res_is.equity_curve.to_csv(os.path.join(out_dir, f"{symbol}_IS_equity.csv"), index=False)
    res_oos.equity_curve.to_csv(os.path.join(out_dir, f"{symbol}_OOS_equity.csv"), index=False)
    pd.DataFrame(res_is.trades).to_csv(os.path.join(out_dir, f"{symbol}_IS_trades.csv"), index=False)
    pd.DataFrame(res_oos.trades).to_csv(os.path.join(out_dir, f"{symbol}_OOS_trades.csv"), index=False)
    cmp.to_csv(os.path.join(out_dir, f"{symbol}_metrics_compare.csv"), index=False)
    with open(os.path.join(out_dir, f"{symbol}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"in_sample": res_is.metrics, "out_of_sample": res_oos.metrics}, f, ensure_ascii=False, indent=2)
    log.info(f"结果已写入 {out_dir}")
    return res_is, res_oos, cmp
