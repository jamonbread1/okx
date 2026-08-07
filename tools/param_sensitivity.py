# -*- coding: utf-8 -*-
"""P3 参数敏感性扫描。

把 sl_mult / tp_mult / min_conf 各 ±pct 扫一遍，观察收益/回撤是否「断崖式」变化。
若在某个参数点收益剧烈跳变，说明该点是拟合出来的；平稳的平台则更可信。

用法:
  python tools/param_sensitivity.py --bar 1H --symbols BTC-USDT-SWAP \
      --params stop_loss_atr_mult,take_profit_atr_mult,min_open_confidence \
      --delta 0.2 --out-dir data/backtest_results/sens
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.multi_engine import MultiSymbolBacktester
from logger import setup_logger
from v3.run_backtest import load_config

log = setup_logger("sens")


def _resolve(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def _ts_or_none(s: str):
    s = (s or "").strip()
    if not s:
        return None
    t = pd.Timestamp(s)
    if pd.isna(t):
        return None
    return t


def run_once(cfg, symbols, main_bar, start, end):
    bt = MultiSymbolBacktester(
        cfg, symbols, main_bar=main_bar,
        start=start, end=end,
    )
    res_is, res_oos = bt.run(is_ratio=0.7)
    return res_is.metrics, res_oos.metrics


def main():
    ap = argparse.ArgumentParser(description="P3 参数敏感性扫描")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--symbols", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--params", default="stop_loss_atr_mult,take_profit_atr_mult,min_open_confidence")
    ap.add_argument("--delta", type=float, default=0.2, help="扰动比例，例如 0.2 表示 ±20%%")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--out-dir", default="data/backtest_results/sens")
    args = ap.parse_args()

    cfg_path = _resolve(args.config)
    out_dir = _resolve(args.out_dir)
    base = load_config(cfg_path)
    if args.only:
        base["strategy"]["enabled_strategies"] = [args.only.strip().lower()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    params = [p.strip() for p in args.params.split(",") if p.strip()]
    start = _ts_or_none(args.start)
    end = _ts_or_none(args.end)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    # 基线
    is0, oos0 = run_once(base, symbols, args.bar, start, end)
    rows.append({
        "param": "baseline", "value": "-",
        "is_return": is0.get("total_return"), "oos_return": oos0.get("total_return"),
        "is_dd": is0.get("max_drawdown"), "oos_dd": oos0.get("max_drawdown"),
        "is_sharpe": is0.get("sharpe"), "oos_sharpe": oos0.get("sharpe"),
    })

    for p in params:
        raw = base.get("strategy", {}).get(p)
        if raw is None:
            log.warning(f"参数 {p} 不在 strategy 配置中，跳过")
            continue
        try:
            base_val = float(raw)
        except (TypeError, ValueError):
            log.warning(f"参数 {p}={raw!r} 无法转为 float，跳过")
            continue
        for mult in (1 - args.delta, 1 + args.delta):
            cfg = copy.deepcopy(base)
            val = base_val * mult
            cfg["strategy"][p] = val
            log.info(f"扫描 {p}={val:.4f}")
            try:
                is_m, oos_m = run_once(cfg, symbols, args.bar, start, end)
            except Exception as e:
                log.warning(f"{p}={val:.4f} 失败: {e}")
                continue
            rows.append({
                "param": p, "value": round(val, 4),
                "is_return": is_m.get("total_return"), "oos_return": oos_m.get("total_return"),
                "is_dd": is_m.get("max_drawdown"), "oos_dd": oos_m.get("max_drawdown"),
                "is_sharpe": is_m.get("sharpe"), "oos_sharpe": oos_m.get("sharpe"),
            })

    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, "sensitivity.csv")
    df.to_csv(out, index=False)
    print("\n===== 参数敏感性 =====")
    print(df.to_string(index=False))
    print(f"\n明细: {out}")


if __name__ == "__main__":
    main()
