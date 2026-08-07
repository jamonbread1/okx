# -*- coding: utf-8 -*-
"""tools.param_sweep — 通用策略参数扫描器 + 抗过拟合分析。

核心设计: 让你**指定任意参数 + 范围**做 grid scan。
不只是 ming 的 3 个核心参数, 任何 v3.strategies/<name>.py 里的
`cfg.get(...)` 字段都能扫。

用法:
  # 默认 (ming 经典 3 参数)
  python tools/param_sweep.py

  # 自定义参数 (--params NAME:KEY=VAL1,VAL2,VAL3,...)
  python tools/param_sweep.py \\
      --params ming:risk_pct=0.005,0.0075,0.01,0.015 \\
      --params ming:r2_chandelier_k=3.0,3.5,4.0,4.5 \\
      --params ming:timeout_max_bars=20,30,45,60

  # 多参数 + 短周期 + 多币种
  python tools/param_sweep.py \\
      --bar 4H \\
      --symbols BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP \\
      --params don:don_period=15,20,25,30 \\
      --params don:don_min_adx=20,25,30 \\
      --max-dd-limit 0.12

  # ML 驱动 (LGBM surrogate + 虚拟扫描)
  python tools/param_sweep_ml.py all \\
      --strategy ming \\
      --n-samples 100 --n-virtual 3000 --n-validate 30 \\
      --out-dir data/ml_sweep

过拟合防御:
  - IS-OOS sharpe 衰减 > 50% 报警
  - IS-OOS CAGR 衰减 > 50% 报警 (用年化收益, 跨时长可比)
  - OOS CAGR 负值报警
  - OOS 段 max_dd > IS 段 + 2% 报警
  - OOS 段交易数 < 10 报警
  - 参数空间"过陡" (相邻 OOS sharpe 波动 > 30%) 报警

综合评分:
  score = oos_sharpe * (1 - overfit_score) * dd_penalty
  dd_penalty = 1.0 if |oos_dd| <= max_dd_limit else max_dd_limit / |oos_dd|
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---- 过拟合检测 ----
def _cagr(equity_curve_or_metrics: dict, days: float) -> float:
    """从 metrics 字典算年化收益（CAGR）。统一用 backtest.metrics._cagr。"""
    from backtest.metrics import _cagr as _bm_cagr
    return _bm_cagr(float(equity_curve_or_metrics.get("total_return", 0) or 0), days)


def overfit_warnings(is_metrics: dict, oos_metrics: dict,
                     is_days: float = 365 * 4, oos_days: float = 365 * 2) -> List[str]:
    """对比 IS / OOS 关键指标，给出过拟合风险提示。

    v3: 用 CAGR 衰减（不是 raw total_return 衰减）
      - raw total_return 衰减在 IS/OOS 时长不等时会被"短时段 OOS 收益看起来更小"误导
      - CAGR 把收益折算到年化，跨时长可比
      - 同时仍保留 sharpe 衰减（sharpe 本身已年化）
    """
    warnings = []
    is_sharpe = float(is_metrics.get("sharpe", 0) or 0)
    oos_sharpe = float(oos_metrics.get("sharpe", 0) or 0)
    is_dd = float(is_metrics.get("max_drawdown", 0) or 0)
    oos_dd = float(oos_metrics.get("max_drawdown", 0) or 0)

    if is_sharpe > 0:
        diff = (is_sharpe - oos_sharpe) / is_sharpe
        if diff > 0.5:
            warnings.append(
                f"⚠️ IS-OOS sharpe 衰减 {diff:.0%} > 50% (IS={is_sharpe:.2f}, OOS={oos_sharpe:.2f})"
            )
    is_cagr = _cagr(is_metrics, is_days)
    oos_cagr = _cagr(oos_metrics, oos_days)
    if is_cagr > 0:
        diff_cagr = (is_cagr - oos_cagr) / is_cagr
        if diff_cagr > 0.5:
            warnings.append(
                f"⚠️ IS-OOS CAGR 衰减 {diff_cagr:.0%} > 50% "
                f"(IS CAGR={is_cagr*100:.1f}%/y, OOS CAGR={oos_cagr*100:.1f}%/y)"
            )
        if oos_cagr < 0:
            warnings.append(
                f"⚠️ OOS CAGR={oos_cagr*100:.2f}%/y 为负（策略在 OOS 段亏钱）"
            )
    if oos_dd < is_dd and is_dd > 0:
        warnings.append(
            f"⚠️ OOS 段 max_dd {oos_dd:.2%} 比 IS 段 {is_dd:.2%} 更深（OOS 段更脆弱）"
        )
    oos_trades = int(oos_metrics.get("trade_count", 0) or 0)
    if oos_trades < 10:
        warnings.append(
            f"⚠️ OOS 段交易数 {oos_trades} < 10（OOS 段统计意义不足）"
        )
    return warnings


def local_overfit_check(results: List[dict]) -> List[str]:
    """对一组相邻参数结果做局部过拟合检查（"参数过陡"特征）。

    如果 sharpe 在相邻参数上波动 > 30%，说明参数空间过陡 → 过拟合高风险。
    """
    if len(results) < 3:
        return []
    # results 可以是 List[SweepResult] (dataclass) 或 List[dict] (测试用). 同时支持.
    def _oos_sharpe(r):
        if isinstance(r, dict):
            return float(r.get("oos_sharpe", 0) or 0)
        return float(getattr(r, "oos_sharpe", 0) or 0)
    sharpes = [_oos_sharpe(r) for r in results]
    sharpes_clean = [s for s in sharpes if s > 0]
    if not sharpes_clean:
        return []
    median = float(np.median(sharpes_clean))
    max_diff = max(abs(s - median) / median for s in sharpes_clean if median > 0)
    warnings = []
    if max_diff > 0.3:
        warnings.append(
            f"⚠️ 参数空间过陡: OOS sharpe 相对 median 波动 {max_diff:.0%} > 30%"
        )
    return warnings


# ---- 单次回测 sub-process ----
@dataclass
class SweepResult:
    params: Dict[str, float]
    is_metrics: Dict = field(default_factory=dict)
    oos_metrics: Dict = field(default_factory=dict)
    is_return: float = 0.0
    oos_return: float = 0.0
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_max_dd: float = 0.0
    oos_max_dd: float = 0.0
    is_trade_count: int = 0
    oos_trade_count: int = 0
    overfit_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""


def run_single_backtest(
    strategy: str,
    bar: str,
    params_by_strategy: Dict[str, Dict[str, float]],
    bar_arg: str = "1D",
    symbols: str = "",
    out_dir: str = "data/param_sweep_tmp",
    timeout: int = 300,
) -> Tuple[Optional[SweepResult], str]:
    """用 sub-process 跑一次 v3.run_backtest, 回读 metrics。

    params_by_strategy: {strategy_name: {param_name: value, ...}, ...}
                       → 自动转成 --params NAME:KEY=VAL 命令行
    """
    cmd = [
        sys.executable, "-m", "v3.run_backtest",
        "--bar", bar_arg,
        "--only", strategy,
        "--out-dir", out_dir,
    ]
    if symbols:
        cmd.extend(["--symbols", symbols])
    for strat, params in params_by_strategy.items():
        for k, v in params.items():
            cmd.extend(["--params", f"{strat}:{k}={v}"])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=_ROOT,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    elapsed = time.time() - t0

    json_p = Path(out_dir) / "MULTI_metrics.json"
    if not json_p.exists():
        return None, f"no metrics.json: returncode={proc.returncode} stderr={proc.stderr[-300:] if proc.stderr else ''}"
    try:
        with open(json_p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"read metrics error: {e}"
    is_m = data.get("in_sample", {}) or {}
    oos_m = data.get("out_of_sample", {}) or {}
    overfit = data.get("overfit", {}) or {}
    bar_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600,
                   "4H": 14400, "1D": 86400, "1W": 604800}
    bar_sec = bar_seconds.get(bar_arg, 86400)
    is_days = float(is_m.get("bars", 0) or 0) * bar_sec / 86400.0
    oos_days = float(oos_m.get("bars", 0) or 0) * bar_sec / 86400.0
    if is_days <= 0:
        is_days = 365 * 4
    if oos_days <= 0:
        oos_days = 365 * 2
    # 合并所有策略的参数 (用于报告)
    all_params = {}
    for d in params_by_strategy.values():
        all_params.update(d)
    res = SweepResult(
        params=all_params,
        is_metrics=is_m,
        oos_metrics=oos_m,
        is_return=float(is_m.get("total_return", 0) or 0),
        oos_return=float(oos_m.get("total_return", 0) or 0),
        is_sharpe=float(is_m.get("sharpe", 0) or 0),
        oos_sharpe=float(oos_m.get("sharpe", 0) or 0),
        is_max_dd=float(is_m.get("max_drawdown", 0) or 0),
        oos_max_dd=float(oos_m.get("max_drawdown", 0) or 0),
        is_trade_count=int(is_m.get("trade_count", 0) or 0),
        oos_trade_count=int(oos_m.get("trade_count", 0) or 0),
        overfit_score=float(overfit.get("overfit_score", 0) or 0),
        warnings=overfit_warnings(is_m, oos_m, is_days=is_days, oos_days=oos_days),
        elapsed=elapsed,
    )
    return res, ""


# ---- 评分函数 ----
def score_result(r: SweepResult, max_dd_limit: float = 0.15) -> float:
    """综合评分: oos_sharpe × stability × dd_penalty."""
    if r.error or r.oos_sharpe <= 0:
        return -1e9
    dd = abs(r.oos_max_dd)
    dd_pen = 1.0 if dd <= max_dd_limit else max_dd_limit / dd if dd > 0 else 0.0
    stab = max(0.0, 1.0 - r.overfit_score)
    return r.oos_sharpe * stab * dd_pen


# ---- 报告 ----
def _format_params_cell(params: Dict[str, float], max_show: int = 4) -> str:
    """把 params dict 格式化成表格单元格 (最多 max_show 个)."""
    items = list(params.items())
    if len(items) <= max_show:
        return " ".join(f"{k}={v}" for k, v in items)
    return " ".join(f"{k}={v}" for k, v in items[:max_show]) + f" ...(+{len(items)-max_show})"


def format_report(
    results: List[SweepResult],
    strategy: str,
    bar_arg: str,
    local_warnings: List[str],
    param_keys: List[str],
) -> str:
    """输出 markdown 报告."""
    lines = [f"# 参数扫描报告: {strategy} ({bar_arg})", ""]
    valid = [r for r in results if not r.error]
    if not valid:
        lines.append("⚠️ 无有效结果")
        return "\n".join(lines)

    valid.sort(key=lambda r: score_result(r), reverse=True)

    # 概述
    lines.append("## 概述")
    lines.append(f"- 总扫描组数: {len(results)}")
    lines.append(f"- 有效组数: {len(valid)}")
    lines.append(f"- 单次回测平均耗时: {sum(r.elapsed for r in valid)/len(valid):.1f}s")
    lines.append("")

    # 扫了哪些参数
    lines.append(f"## 扫描参数 ({len(param_keys)} 个)")
    lines.append("| 参数 | 扫描值数 |")
    lines.append("|---|---:|")
    for k in param_keys:
        n = len({v for r in valid for rk, rv in r.params.items() if rk == k for v in [rv]})
        lines.append(f"| {k} | {n} |")
    lines.append("")

    # Top 5
    lines.append("## Top 5 (按 score 排序)")
    lines.append("")
    lines.append("| Rank | score | params | IS ret | OOS ret | IS CAGR | OOS CAGR | IS sharpe | OOS sharpe | OOS dd | OOS trades | overfit |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(valid[:5]):
        s = score_result(r)
        is_cagr = _cagr(r.is_metrics, max(1.0, float(r.is_metrics.get('bars', 0) or 0)))
        oos_cagr = _cagr(r.oos_metrics, max(1.0, float(r.oos_metrics.get('bars', 0) or 0)))
        lines.append(
            f"| {i+1} | {s:.3f} | {_format_params_cell(r.params)} | "
            f"{r.is_return*100:+.2f}% | {r.oos_return*100:+.2f}% | "
            f"{is_cagr*100:+.2f}%/y | {oos_cagr*100:+.2f}%/y | "
            f"{r.is_sharpe:.2f} | {r.oos_sharpe:.2f} | "
            f"{r.oos_max_dd*100:.2f}% | {r.oos_trade_count} | {r.overfit_score:.3f} |"
        )
    lines.append("")

    # 全部
    lines.append(f"## 全部结果 ({len(valid)} 组)")
    lines.append("")
    lines.append("| # | params | IS ret | OOS ret | IS CAGR | OOS CAGR | IS sharpe | OOS sharpe | OOS dd | OOS trades | overfit | score | 报警 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(valid):
        s = score_result(r)
        is_cagr = _cagr(r.is_metrics, max(1.0, float(r.is_metrics.get('bars', 0) or 0)))
        oos_cagr = _cagr(r.oos_metrics, max(1.0, float(r.oos_metrics.get('bars', 0) or 0)))
        w = "; ".join(r.warnings) if r.warnings else ""
        lines.append(
            f"| {i+1} | {_format_params_cell(r.params)} | "
            f"{r.is_return*100:+.2f}% | {r.oos_return*100:+.2f}% | "
            f"{is_cagr*100:+.2f}%/y | {oos_cagr*100:+.2f}%/y | "
            f"{r.is_sharpe:.2f} | {r.oos_sharpe:.2f} | "
            f"{r.oos_max_dd*100:.2f}% | {r.oos_trade_count} | "
            f"{r.overfit_score:.3f} | {s:.3f} | {w} |"
        )
    lines.append("")

    err = [r for r in results if r.error]
    if err:
        lines.append(f"## 失败组 ({len(err)} 组)")
        for r in err:
            lines.append(f"- params={r.params} error={r.error[:200]}")
        lines.append("")

    if local_warnings:
        lines.append("## 全局过拟合警告")
        for w in local_warnings:
            lines.append(f"- {w}")
        lines.append("")

    if valid:
        best = valid[0]
        lines.append("## 推荐参数 (综合评分最高)")
        lines.append("")
        lines.append("```yaml")
        for k, v in best.params.items():
            lines.append(f"  {k}: {v}")
        lines.append("```")
        lines.append("")
        lines.append(f"- OOS total_return: {best.oos_return*100:+.2f}%")
        lines.append(f"- OOS sharpe: {best.oos_sharpe:.2f}")
        lines.append(f"- OOS CAGR: {_cagr(best.oos_metrics, max(1.0, float(best.oos_metrics.get('bars', 0) or 0)))*100:+.2f}%/y")
        lines.append(f"- OOS max_dd: {best.oos_max_dd*100:.2f}%")
        lines.append(f"- OOS trades: {best.oos_trade_count}")
        lines.append(f"- overfit_score: {best.overfit_score:.3f}")
        lines.append(f"- 报警: {'; '.join(best.warnings) if best.warnings else '无'}")
    return "\n".join(lines)


# ---- 参数解析 ----
def parse_param_spec(spec: str) -> Tuple[str, str, List[float]]:
    """解析 --params NAME:KEY=VAL1,VAL2,VAL3 形式.

    Returns: (strategy_name, param_key, [values])
    """
    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)=(.+)$", spec)
    if not m:
        raise ValueError(
            f"--params 格式应为 NAME:KEY=VAL1,VAL2,..., 收到: {spec!r}"
        )
    name, key, raw_values = m.group(1), m.group(2), m.group(3)
    values: List[float] = []
    for v in raw_values.split(","):
        v = v.strip()
        if not v:
            continue
        # int 检测
        try:
            iv = int(v)
            values.append(iv)
            continue
        except ValueError:
            pass
        # float
        try:
            values.append(float(v))
        except ValueError:
            raise ValueError(f"--params 值无法解析: {v!r} in {spec!r}")
    if not values:
        raise ValueError(f"--params 至少需要一个值: {spec!r}")
    return name, key, values


# ---- 主函数 ----
def main() -> int:
    ap = argparse.ArgumentParser(
        description="通用策略参数扫描器 + 抗过拟合分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认 (ming 经典 3 参数)
  python tools/param_sweep.py

  # 自定义参数
  python tools/param_sweep.py \\
      --params ming:risk_pct=0.005,0.01,0.015 \\
      --params ming:r2_chandelier_k=3.0,3.5,4.0 \\
      --params ming:timeout_max_bars=20,30,45

  # 任意策略 (e.g. don)
  python tools/param_sweep.py \\
      --strategy don \\
      --params don:don_period=15,20,25 \\
      --params don:don_min_adx=20,25,30
""",
    )
    ap.add_argument("--strategy", default="ming", help="策略名（默认 ming）")
    ap.add_argument("--bar", default="1D", help="主周期（默认 1D）")
    ap.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP",
                    help="品种列表")
    ap.add_argument("--params", action="append", default=[],
                    help="扫描参数 NAME:KEY=VAL1,VAL2,... (可多次)")
    ap.add_argument("--out-dir", default="data/param_sweep",
                    help="报告输出目录")
    ap.add_argument("--max-dd-limit", type=float, default=0.15,
                    help="max_dd 容忍上限（默认 0.15）")
    args = ap.parse_args()

    # 解析所有 --params
    param_specs: List[Tuple[str, str, List[float]]] = []
    for spec in args.params:
        param_specs.append(parse_param_spec(spec))

    # 校验所有 spec 都是同一个 strategy (多策略扫描不在 v1 支持范围内)
    strategies = {name for name, _, _ in param_specs}
    if len(strategies) > 1:
        print(f"❌ 多策略扫描不支持: {strategies}")
        return 1
    if strategies and args.strategy not in strategies:
        print(f"⚠️ --strategy {args.strategy} 不在 --params 里, 但仍使用")
    if strategies and args.strategy != list(strategies)[0]:
        # 自动修正: 用 --params 里的 strategy
        args.strategy = list(strategies)[0]
        print(f"⚠️ 自动设置 --strategy={args.strategy} (与 --params 一致)")

    # 默认 (ming 经典 3 参数)
    if not param_specs:
        print("⚠️ 未指定 --params, 用 ming 经典 3 参数默认")
        param_specs = [
            ("ming", "risk_pct", [0.005, 0.0075, 0.01, 0.015]),
            ("ming", "r2_chandelier_k", [3.0, 3.5, 4.0, 4.5]),
            ("ming", "timeout_max_bars", [20, 30, 45]),
        ]
        strategies = {"ming"}
        args.strategy = "ming"

    # 算总组合数
    n_combos = 1
    for _, _, values in param_specs:
        n_combos *= len(values)
    param_keys = [k for _, k, _ in param_specs]
    print(f"=== Param Sweep: {args.strategy} ({args.bar}) ===")
    for name, key, values in param_specs:
        print(f"  {name}.{key}: {values}")
    print(f"组合数: {n_combos} (预计 {n_combos * 5 // 60} 分钟, 按 5s/组)")
    print(f"输出: {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    results: List[SweepResult] = []
    t_start = time.time()
    combo_idx = 0
    # 笛卡尔积
    from itertools import product
    value_lists = [values for _, _, values in param_specs]
    for combo in product(*value_lists):
        combo_idx += 1
        params_by_strat: Dict[str, Dict[str, float]] = {args.strategy: {}}
        for (name, key, _), value in zip(param_specs, combo):
            params_by_strat[name][key] = value
        params_str = ", ".join(
            f"{k}={v}" for k, v in params_by_strat[args.strategy].items()
        )
        print(f"[{combo_idx}/{n_combos}] {params_str} ... ", end="", flush=True)
        res, err = run_single_backtest(
            strategy=args.strategy,
            bar=args.bar,
            params_by_strategy=params_by_strat,
            bar_arg=args.bar,
            symbols=args.symbols,
            out_dir=tmp_dir,
        )
        if res is None:
            all_params = {k: v for d in params_by_strat.values() for k, v in d.items()}
            res = SweepResult(params=all_params, error=err, elapsed=0.0)
            print(f"FAIL ({err[:80]})")
        else:
            print(
                f"is_ret={res.is_return*100:+.2f}% oos_ret={res.oos_return*100:+.2f}% "
                f"oos_sharpe={res.oos_sharpe:.2f} oos_dd={res.oos_max_dd*100:.2f}% "
                f"overfit={res.overfit_score:.2f} ({res.elapsed:.1f}s)"
            )
        results.append(res)

    total_elapsed = time.time() - t_start
    print(f"\n=== 完成: {len(results)} 组 / {total_elapsed:.0f}s ===\n")

    local_warns = local_overfit_check([r for r in results if not r.error])

    report = format_report(results, args.strategy, args.bar, local_warns, param_keys)
    out_md = os.path.join(args.out_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📊 报告: {out_md}")

    summary = {
        "strategy": args.strategy,
        "bar": args.bar,
        "param_specs": [
            {"strategy": name, "param": key, "values": values}
            for name, key, values in param_specs
        ],
        "n_combos": n_combos,
        "elapsed_sec": total_elapsed,
        "results": [
            {
                **r.params,
                "is_return": r.is_return, "oos_return": r.oos_return,
                "is_sharpe": r.is_sharpe, "oos_sharpe": r.oos_sharpe,
                "is_max_dd": r.is_max_dd, "oos_max_dd": r.oos_max_dd,
                "is_trades": r.is_trade_count, "oos_trades": r.oos_trade_count,
                "overfit": r.overfit_score,
                "score": score_result(r, max_dd_limit=args.max_dd_limit),
                "warnings": r.warnings,
                "error": r.error,
            }
            for r in results
        ],
        "global_warnings": local_warns,
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📋 JSON: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
