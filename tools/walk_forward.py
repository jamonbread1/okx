# -*- coding: utf-8 -*-
"""P3 滚动 Walk-Forward 验证。

在多个滚动「训练→验证」窗口上跑回测，输出 OOS 指标的**分布**（而不只是单点），
用来判断参数稳定性与是否有过拟合。

用法:
  python tools/walk_forward.py --bar 1H --symbols BTC-USDT-SWAP --train-days 90 \
      --test-days 30 --step-days 30 --out-dir data/backtest_results_v2/wf
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.multi_engine import MultiSymbolBacktester
from logger import setup_logger
from v2.run_backtest import load_v2_config


def _resolve(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def main():
    ap = argparse.ArgumentParser(description="P3 滚动 Walk-Forward")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--out-dir", default="data/backtest_results_v2/wf")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    cfg_path = _resolve(args.config)
    out_dir = _resolve(args.out_dir)
    cfg = load_v2_config(cfg_path)
    if args.only:
        cfg["strategy"]["enabled_strategies"] = [args.only.strip().lower()]

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bt = MultiSymbolBacktester(cfg, symbols, main_bar=args.bar)
    store = bt._load_store()
    bt._store = store
    timeline = bt._aligned_index(store)
    timeline = pd.DatetimeIndex(pd.to_datetime(timeline))

    t0 = timeline[0].normalize()
    t1 = timeline[-1].normalize()
    win = pd.Timedelta(days=args.test_days)
    step = pd.Timedelta(days=args.step_days)

    os.makedirs(out_dir, exist_ok=True)
    log = setup_logger("wf")

    rows = []
    cursor = t0 + pd.Timedelta(days=args.train_days)
    if cursor + win > t1:
        log.error(
            f"数据跨度不足以做 walk-forward: "
            f"需要 train={args.train_days}d + test={args.test_days}d，"
            f"实际 {t0.date()} → {t1.date()}"
        )
        sys.exit(1)

    while cursor + win <= t1:
        test_end = cursor + win
        log.info(f"验证窗 {cursor} → {test_end}")
        try:
            res = bt._run_phase(cursor, test_end, "test")
        except Exception as e:
            log.warning(f"窗口异常 {cursor}: {e}")
            cursor += step
            continue
        m = res.metrics
        rows.append({
            "test_start": str(cursor),
            "test_end": str(test_end),
            "total_return": m.get("total_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe": m.get("sharpe"),
            "calmar": m.get("calmar"),
            "win_rate": m.get("win_rate"),
            "trade_count": m.get("trade_count"),
            "total_pnl": m.get("total_pnl"),
        })
        cursor += step

    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, "walk_forward_oos.csv")
    df.to_csv(out, index=False)
    print("\n===== Walk-Forward OOS 指标分布 =====")
    if not df.empty:
        for col in ["total_return", "sharpe", "max_drawdown", "win_rate", "calmar"]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            print(f"  {col}: mean={s.mean():.4f} std={s.std():.4f} "
                  f"min={s.min():.4f} p25={s.quantile(.25):.4f} "
                  f"med={s.median():.4f} p75={s.quantile(.75):.4f} max={s.max():.4f}")
        print(f"  正收益窗口: {int((df['total_return'].astype(float) > 0).sum())}/{len(df)}")
        print(f"  OOS 交易数总和: {int(df['trade_count'].fillna(0).sum())}")
    else:
        print("  (无有效窗口)")
    print(f"\n明细: {out}")
    with open(os.path.join(out_dir, "walk_forward_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "symbols": symbols,
            "bar": args.bar,
            "data_range": [str(t0), str(t1)],
            "n_windows": len(rows),
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
