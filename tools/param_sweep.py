# -*- coding: utf-8 -*-
"""tools.param_sweep — 策略参数扫描 + 抗过拟合分析。

目标：
  1. 对一组策略（如 ming）的关键参数（risk_pct, chandelier_k, max_bars）扫描
  2. 每组跑 IS / OOS（复用 v3.run_backtest 逻辑）
  3. 输出 sharpe / max_dd / total_return / overfit_score
  4. **过拟合防御**：
     - IS-OOS sharpe 差异 > 50% 报警
     - IS-OOS total_return 差异 > 50% 报警
     - 同一参数 +10% / -10% 扫描后 sharpe 变化 > 30% 报警（"过陡"，过拟合特征）
  5. 排序按"风险调整收益 / 过拟合稳定性"综合评分

⚠️  注意：
  - 扫描 24 组 × 7 年回测 = 168 个 backtest，每次 ~5s → 总 ~14 min
  - 用 sub-process 避免单一 backtest 内存泄漏
  - 输出 markdown 报告 + JSON 摘要

用法：
  # 默认扫描 ming
  python tools/param_sweep.py

  # 指定策略
  python tools/param_sweep.py --strategy ming

  # 自定义参数范围
  python tools/param_sweep.py \\
    --risk-pct-list 0.010,0.015,0.020,0.025,0.030 \\
    --k-list 3.5,4.5,5.5,6.5 \\
    --max-bars-list 20,30,45,60

  # 输出报告
  python tools/param_sweep.py --out-dir data/param_sweep_v8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 让脚本能找到 v3 包
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

    v3 改进（2026-08）：用 CAGR 衰减（不是 raw total_return 衰减）
      - raw total_return 衰减在 IS/OOS 时长不等时会被"短时段 OOS 收益看起来更小"误导
      - CAGR 把收益折算到年化，跨时长可比
      - 同时仍保留 sharpe 衰减（sharpe 本身已年化）
    """
    warnings = []
    is_sharpe = float(is_metrics.get("sharpe", 0) or 0)
    oos_sharpe = float(oos_metrics.get("sharpe", 0) or 0)
    is_dd = float(is_metrics.get("max_drawdown", 0) or 0)
    oos_dd = float(oos_metrics.get("max_drawdown", 0) or 0)

    # 1. IS-OOS sharpe 衰减（sharpe 已年化，跨时长可比）
    if is_sharpe > 0:
        diff = (is_sharpe - oos_sharpe) / is_sharpe
        if diff > 0.5:
            warnings.append(
                f"⚠️ IS-OOS sharpe 衰减 {diff:.0%} > 50%（IS={is_sharpe:.2f}, OOS={oos_sharpe:.2f}）"
            )
    # 2. IS-OOS CAGR 衰减（v3 关键改进：用年化收益对比）
    is_cagr = _cagr(is_metrics, is_days)
    oos_cagr = _cagr(oos_metrics, oos_days)
    if is_cagr > 0:
        diff_cagr = (is_cagr - oos_cagr) / is_cagr
        if diff_cagr > 0.5:
            warnings.append(
                f"⚠️ IS-OOS CAGR 衰减 {diff_cagr:.0%} > 50% "
                f"（IS CAGR={is_cagr*100:.1f}%/y, OOS CAGR={oos_cagr*100:.1f}%/y）"
            )
        # 额外：OOS CAGR 为负也报警（说明策略只在 IS 段有效，OOS 段负收益）
        if oos_cagr < 0:
            warnings.append(
                f"⚠️ OOS CAGR={oos_cagr*100:.2f}%/y 为负（策略在 OOS 段亏钱）"
            )
    # 3. OOS 段回撤 > IS 段回撤（不一致）
    if oos_dd < is_dd and is_dd > 0:
        warnings.append(
            f"⚠️ OOS 段 max_dd {oos_dd:.2%} 比 IS 段 {is_dd:.2%} 更深（OOS 段更脆弱）"
        )
    # 4. OOS 段交易数太少
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
    sharpes = [r.get("oos_sharpe", 0) or 0 for r in results]
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
    params: Dict[str, float],
    bar_arg: str = "1D",
    symbols: str = "",
    out_dir: str = "data/param_sweep_tmp",
    timeout: int = 300,
) -> Tuple[Optional[SweepResult], str]:
    """用 sub-process 跑一次 v3.run_backtest，回读 metrics。"""
    # 构造命令行
    cmd = [
        sys.executable, "-m", "v3.run_backtest",
        "--bar", bar_arg,
        "--only", strategy,
        "--out-dir", out_dir,
    ]
    if symbols:
        cmd.extend(["--symbols", symbols])
    # ming 参数
    for k, v in params.items():
        cmd.extend(["--params", f"ming:{k}={v}"])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=_ROOT,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    elapsed = time.time() - t0

    if proc.returncode != 0:
        # 找 metrics.json
        json_p = Path(out_dir) / "MULTI_metrics.json"
        if not json_p.exists():
            return None, f"returncode={proc.returncode}\n{proc.stderr[-500:] if proc.stderr else ''}"
    json_p = Path(out_dir) / "MULTI_metrics.json"
    if not json_p.exists():
        return None, f"no metrics.json: {proc.stderr[-500:] if proc.stderr else ''}"
    try:
        with open(json_p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"read metrics error: {e}"
    is_m = data.get("in_sample", {}) or {}
    oos_m = data.get("out_of_sample", {}) or {}
    overfit = data.get("overfit", {}) or {}
    # v3: 用 bars 字段和 bar 周期换算 IS/OOS 实际天数
    # 默认按 1D 周期 1 bar = 1 day
    bar_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600,
                   "4H": 14400, "1D": 86400, "1W": 604800}
    bar_sec = bar_seconds.get(bar_arg, 86400)
    is_days = float(is_m.get("bars", 0) or 0) * bar_sec / 86400.0
    oos_days = float(oos_m.get("bars", 0) or 0) * bar_sec / 86400.0
    if is_days <= 0:
        is_days = 365 * 4
    if oos_days <= 0:
        oos_days = 365 * 2
    res = SweepResult(
        params=dict(params),
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
    """综合评分：风险调整收益 / 过拟合稳定性。

    score = oos_sharpe * stability_factor
    stability_factor = (1 - overfit_score) * dd_penalty
    dd_penalty = 1.0 if |oos_max_dd| <= max_dd_limit else max_dd_limit / |oos_max_dd|
    """
    if r.error or r.oos_sharpe <= 0:
        return -1e9
    dd = abs(r.oos_max_dd)
    dd_pen = 1.0 if dd <= max_dd_limit else max_dd_limit / dd if dd > 0 else 0.0
    stab = max(0.0, 1.0 - r.overfit_score)
    return r.oos_sharpe * stab * dd_pen


# ---- 报告 ----
def format_report(
    results: List[SweepResult],
    strategy: str,
    bar_arg: str,
    local_warnings: List[str],
) -> str:
    """输出 markdown 报告。"""
    lines = []
    lines.append(f"# 参数扫描报告: {strategy} ({bar_arg})\n")
    valid = [r for r in results if not r.error]
    if not valid:
        lines.append("\n⚠️ 无有效结果\n")
        return "\n".join(lines)

    # 排序：按 score 降序
    valid.sort(key=lambda r: score_result(r), reverse=True)

    # 概述
    lines.append("## 概述")
    lines.append(f"- 总扫描组数: {len(results)}")
    lines.append(f"- 有效组数: {len(valid)}")
    lines.append(f"- 单次回测平均耗时: {sum(r.elapsed for r in valid)/len(valid):.1f}s")
    lines.append("")

    # Top 5
    lines.append("## Top 5（按 score 排序）\n")
    lines.append("| Rank | score | risk_pct | k | max_bars | IS ret | OOS ret | IS CAGR | OOS CAGR | IS sharpe | OOS sharpe | IS dd | OOS dd | OOS trades | overfit |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(valid[:5]):
        s = score_result(r)
        is_cagr = _cagr(r.is_metrics, max(1.0, float(r.is_metrics.get('bars', 0) or 0)))
        oos_cagr = _cagr(r.oos_metrics, max(1.0, float(r.oos_metrics.get('bars', 0) or 0)))
        lines.append(
            f"| {i+1} | {s:.3f} | {r.params.get('risk_pct', '-'):.4f} | "
            f"{r.params.get('r2_chandelier_k', '-'):.1f} | {r.params.get('timeout_max_bars', '-'):.0f} | "
            f"{r.is_return*100:+.2f}% | {r.oos_return*100:+.2f}% | "
            f"{is_cagr*100:+.2f}%/y | {oos_cagr*100:+.2f}%/y | "
            f"{r.is_sharpe:.2f} | {r.oos_sharpe:.2f} | "
            f"{r.is_max_dd*100:.2f}% | {r.oos_max_dd*100:.2f}% | "
            f"{r.oos_trade_count} | {r.overfit_score:.3f} |"
        )
    lines.append("")

    # 全部
    lines.append("## 全部结果\n")
    lines.append("| # | risk_pct | k | max_bars | IS ret | OOS ret | IS CAGR | OOS CAGR | IS sharpe | OOS sharpe | OOS dd | OOS trades | overfit | score | 报警 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(valid):
        s = score_result(r)
        is_cagr = _cagr(r.is_metrics, max(1.0, float(r.is_metrics.get('bars', 0) or 0)))
        oos_cagr = _cagr(r.oos_metrics, max(1.0, float(r.oos_metrics.get('bars', 0) or 0)))
        w = "; ".join(r.warnings) if r.warnings else ""
        lines.append(
            f"| {i+1} | {r.params.get('risk_pct', '-'):.4f} | "
            f"{r.params.get('r2_chandelier_k', '-'):.1f} | {r.params.get('timeout_max_bars', '-'):.0f} | "
            f"{r.is_return*100:+.2f}% | {r.oos_return*100:+.2f}% | "
            f"{is_cagr*100:+.2f}%/y | {oos_cagr*100:+.2f}%/y | "
            f"{r.is_sharpe:.2f} | {r.oos_sharpe:.2f} | "
            f"{r.oos_max_dd*100:.2f}% | {r.oos_trade_count} | "
            f"{r.overfit_score:.3f} | {s:.3f} | {w} |"
        )
    lines.append("")

    # 错误组
    err = [r for r in results if r.error]
    if err:
        lines.append("## 失败组\n")
        for r in err:
            lines.append(f"- params={r.params} error={r.error[:200]}")
        lines.append("")

    # 全局警告
    if local_warnings:
        lines.append("## 全局过拟合警告\n")
        for w in local_warnings:
            lines.append(f"- {w}")
        lines.append("")

    # 推荐
    if valid:
        best = valid[0]
        lines.append("## 推荐参数（综合评分最高）\n")
        lines.append("```yaml")
        for k, v in best.params.items():
            lines.append(f"ming_{k}: {v}")
        lines.append("```")
        lines.append(f"\n- OOS total_return: {best.oos_return*100:+.2f}%")
        lines.append(f"- OOS sharpe: {best.oos_sharpe:.2f}")
        lines.append(f"- OOS max_dd: {best.oos_max_dd*100:.2f}%")
        lines.append(f"- OOS trades: {best.oos_trade_count}")
        lines.append(f"- overfit_score: {best.overfit_score:.3f}")
        lines.append(f"- 报警: {'; '.join(best.warnings) if best.warnings else '无'}")
    return "\n".join(lines)


# ---- 主函数 ----
def parse_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="策略参数扫描 + 抗过拟合分析")
    ap.add_argument("--strategy", default="ming", help="策略名（默认 ming）")
    ap.add_argument("--bar", default="1D", help="主周期（默认 1D）")
    ap.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP",
                    help="品种列表")
    ap.add_argument("--risk-pct-list", default="0.010,0.015,0.020,0.025,0.030",
                    help="risk_pct 扫描值")
    ap.add_argument("--k-list", default="3.5,4.5,5.5,6.5",
                    help="r2_chandelier_k 扫描值")
    ap.add_argument("--max-bars-list", default="20,30,45,60",
                    help="timeout_max_bars 扫描值")
    ap.add_argument("--out-dir", default="data/param_sweep",
                    help="报告输出目录")
    ap.add_argument("--max-dd-limit", type=float, default=0.15,
                    help="max_dd 容忍上限（默认 0.15）")
    args = ap.parse_args()

    risk_pcts = parse_list(args.risk_pct_list)
    ks = parse_list(args.k_list)
    max_bars_list = parse_list(args.max_bars_list)
    # 总组合数
    n_combos = len(risk_pcts) * len(ks) * len(max_bars_list)
    print(f"=== Param Sweep: {args.strategy} ({args.bar}) ===")
    print(f"risk_pct 范围: {risk_pcts}")
    print(f"k 范围: {ks}")
    print(f"max_bars 范围: {max_bars_list}")
    print(f"组合数: {n_combos} (预计 {n_combos * 5 // 60} 分钟)")
    print(f"输出: {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    # 临时目录放回测结果
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    results: List[SweepResult] = []
    combo_idx = 0
    t_start = time.time()
    for risk in risk_pcts:
        for k in ks:
            for mb in max_bars_list:
                combo_idx += 1
                params = {
                    "risk_pct": risk,
                    "r2_chandelier_k": k,
                    "timeout_max_bars": int(mb),
                }
                # 限制其他参数：固定 r15_partial 0.15 / r1_be_buffer 0.5 / 关闭 natr_block 0.99
                # 让扫描焦点在 3 个核心参数上
                params.update({
                    "r15_partial_pct": 0.15,
                    "r1_be_buffer_atr": 0.5,
                })
                print(f"[{combo_idx}/{n_combos}] risk={risk} k={k} max_bars={mb} ... ", end="", flush=True)
                res, err = run_single_backtest(
                    strategy=args.strategy,
                    bar=args.bar,
                    params=params,
                    bar_arg=args.bar,
                    symbols=args.symbols,
                    out_dir=tmp_dir,
                )
                if res is None:
                    res = SweepResult(params=params, error=err, elapsed=0.0)
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

    # 局部过拟合检查
    local_warns = local_overfit_check([r for r in results if not r.error])

    # 写报告
    report = format_report(results, args.strategy, args.bar, local_warns)
    out_md = os.path.join(args.out_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📊 报告: {out_md}")

    # 写 JSON 摘要
    summary = {
        "strategy": args.strategy,
        "bar": args.bar,
        "n_combos": n_combos,
        "elapsed_sec": total_elapsed,
        "results": [
            {
                **{k: v for k, v in r.params.items()},
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
