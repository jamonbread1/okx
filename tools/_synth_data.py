# -*- coding: utf-8 -*-
"""
tools._synth_data — 在 sandbox 内合成 OKX 1D K 线（仅供 param_sweep_ml 流程验证用）。

⚠️ 仅供 sandbox 使用！数据为几何布朗运动 + drift/vol regime switch 合成，
    与真实行情毫无关系。**仅用于验证 param_sweep_ml.py all 端到端流程**，
    不代表任何策略表现。

覆盖范围（与用户真实 bars.db 一致，方便 backtest 不报 ts range 异常）：
  5 币种 × 1D，2019-11-27 → 2026-08-03，2442 根/币种

写入：
  data/okx_history/bars.db  — SQLite 主表
  data/okx_history/parquet/{SYM}/bars_1D.parquet  — parquet 备份

用法：
  python tools/_synth_data.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.trade_pipeline import _ensure_db, DB_PATH, PQ_ROOT  # noqa: E402

# 与用户本地真实数据一致的时间范围
START = pd.Timestamp("2019-11-27 16:00:00")
END = pd.Timestamp("2026-08-03 16:00:00")
N_BARS = int((END - START).total_seconds() / 86400) + 1  # 约 2442

# 5 币种，初始价 (与 2019-11 真实区间粗对齐)
SYMBOLS = {
    "BTC-USDT-SWAP":  {"p0": 7500.0,  "mu": 0.0006, "sigma": 0.035, "vol_scale": 1.0},
    "ETH-USDT-SWAP":  {"p0": 150.0,   "mu": 0.0005, "sigma": 0.045, "vol_scale": 1.1},
    "SOL-USDT-SWAP":  {"p0": 20.0,    "mu": 0.0007, "sigma": 0.065, "vol_scale": 1.4},
    "DOGE-USDT-SWAP": {"p0": 0.002,   "mu": 0.0004, "sigma": 0.055, "vol_scale": 1.2},
    "AVAX-USDT-SWAP": {"p0": 3.0,     "mu": 0.0005, "sigma": 0.060, "vol_scale": 1.3},
}


def synth_ohlc(p0: float, n: int, mu: float, sigma: float, vol_scale: float,
               rng: np.random.Generator) -> pd.DataFrame:
    """生成 1D OHLC 序列：GBM + regime-switch 制造趋势 + 噪声 high/low/vol。"""
    # 1) regime 标签：每 60~250 根切换一次
    regime = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        seg = int(rng.integers(60, 250))
        regime[i:min(i + seg, n)] = int(rng.choice([-1, 0, 1], p=[0.25, 0.50, 0.25]))
        i += seg
    # drift 随 regime 变
    drift = np.where(regime == 1, mu * 2.5, np.where(regime == -1, -mu * 1.5, mu))
    # 2) 局部波动
    vol = sigma * vol_scale * (1.0 + 0.4 * np.abs(regime))
    eps = rng.normal(0, 1, n) * vol
    log_ret = drift + eps
    log_ret[0] = 0.0
    log_p = np.cumsum(log_ret) + np.log(p0)
    close = np.exp(log_p)
    # 3) OHLC: open=前收, high/low 在 ±0.5*vol 范围内, close=本根
    open_ = np.empty(n)
    open_[0] = p0
    open_[1:] = close[:-1]
    intraday = np.abs(rng.normal(0, 0.5, n)) * vol * close
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday
    low = np.maximum(low, 1e-9)
    # 4) vol/buy_vol/sell_vol/cvd 简单合成
    base_vol = 1_000.0 * vol_scale
    vol = np.abs(rng.normal(base_vol, base_vol * 0.5, n))
    buy_frac = np.clip(0.5 + 0.1 * regime + 0.05 * rng.normal(0, 1, n), 0.1, 0.9)
    buy_vol = vol * buy_frac
    sell_vol = vol * (1.0 - buy_frac)
    cvd = np.cumsum(buy_vol - sell_vol)
    return pd.DataFrame({
        "ts": pd.date_range(START, periods=n, freq="1D"),
        "open": open_, "high": high, "low": low, "close": close,
        "vol": vol, "buy_vol": buy_vol, "sell_vol": sell_vol, "cvd": cvd,
    })


def write_bars(symbol: str, bar: str, df: pd.DataFrame) -> int:
    conn = _ensure_db(DB_PATH)
    try:
        rows = [
            (symbol, bar, int(pd.Timestamp(r.ts).timestamp()),
             float(r.open), float(r.high), float(r.low), float(r.close),
             float(r.vol), float(r.buy_vol), float(r.sell_vol), float(r.cvd))
            for r in df.itertuples(index=False)
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(symbol,bar,ts,open,high,low,close,vol,buy_vol,sell_vol,cvd) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        cur = conn.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM ohlcv WHERE symbol=? AND bar=?",
            (symbol, bar),
        )
        n_bars, ts_min, ts_max = cur.fetchone()
        conn.execute(
            "INSERT OR REPLACE INTO meta (symbol,bar,n_bars,ts_min,ts_max,built_at) "
            "VALUES (?,?,?,?,?,datetime('now'))",
            (symbol, bar, int(n_bars), ts_min, ts_max),
        )
        conn.commit()
        return int(n_bars)
    finally:
        conn.close()


def write_parquet(symbol: str, bar: str, df: pd.DataFrame) -> str:
    out_dir = os.path.join(PQ_ROOT, symbol)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bars_{bar}.parquet")
    df.to_parquet(path, index=False)
    return path


def main():
    seed = int(os.environ.get("SYNTH_SEED", "42"))
    rng = np.random.default_rng(seed)
    print(f"[synth] seed={seed}  n_bars={N_BARS}  range={START.date()} → {END.date()}")
    print(f"[synth] DB={os.path.abspath(DB_PATH)}")
    print(f"[synth] PQ={os.path.abspath(PQ_ROOT)}")
    for sym, cfg in SYMBOLS.items():
        df = synth_ohlc(cfg["p0"], N_BARS, cfg["mu"], cfg["sigma"], cfg["vol_scale"], rng)
        n_db = write_bars(sym, "1D", df)
        path = write_parquet(sym, "1D", df)
        print(f"  [{sym}] rows={len(df)} p0={cfg['p0']:.4g} p_end={df['close'].iloc[-1]:.4g} "
              f"| DB n={n_db} | parquet={os.path.basename(path)}")
    print("\n[synth] 完成。可立即用: python -m v3.run_backtest --bar 1D --only ming")
    return 0


if __name__ == "__main__":
    sys.exit(main())
