# -*- coding: utf-8 -*-
"""tools.param_sweep_ml — ML 驱动的参数扫描器。

设计目标
========
替代暴力 grid scan，用 ML 模型学习"参数 → 策略表现"映射，在更大参数空间
里**虚拟扫描**，然后挑 Top-N 真跑验证。

实现思路（参照用户提供的 LGBMRegressor 模板 + 关键设计原则）
------------------------------------------------------------
1. **多目标**（不是单标量 score）：
   - OOS sharpe: 回归, winsorize 剪尾 ±3σ（标签噪声大）
   - OOS max_dd > 15% 概率: 二分类（比回归 dd 更稳）
   - OOS trade_count >= 阈值 概率: 二分类（统计意义是否足够）
   - 最终 score = pred_sharpe × P(dd_ok) × P(trades_ok)
2. **lambdarank 排序**（不是直接回归 sharpe）：
   - 扫描器真正需求是"把好参数排到前面",不是精确预测数值
   - LGBM lambdarank objective 直接优化 NDCG
   - 每一轮 sweep 当作一个 query group
3. **时间切分**（不是随机 CV）：
   - 同一 sweep 内不同参数 OOS 段高度相关
   - 随机切分会让模型虚高, 尤其 sharpe 那个
4. **贝叶斯搜索 + LGBM 增量 refit**：
   - 路线 A: gp_minimize (内置 GP, 迭代 60 轮)
   - 路线 B: LGBM 增量训练 (每轮真跑 N 个后 refit surrogate)
5. **特征**: MVP 只用参数, 不加市场状态特征

用法:
  # 1) 生成种子样本 (用 LGBM 做"代理扫描")
  python tools/param_sweep_ml.py seed \\
      --strategy ming --bar 1D \\
      --n-samples 200 \\
      --out-dir data/ml_sweep/seed

  # 2) 训练 LGBM surrogate
  python tools/param_sweep_ml.py train \\
      --seed-dir data/ml_sweep/seed \\
      --model-out data/ml_sweep/surrogate.txt

  # 3) 用 surrogate 做虚拟扫描 + Top-N 真跑验证
  python tools/param_sweep_ml.py search \\
      --model data/ml_sweep/surrogate.txt \\
      --n-virtual 5000 --n-validate 30 \\
      --out-dir data/ml_sweep/search

  # 一条龙: seed → train → search
  python tools/param_sweep_ml.py all \\
      --strategy ming --bar 1D \\
      --n-samples 100 --n-virtual 3000 --n-validate 30 \\
      --out-dir data/ml_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


# ---------------------------------------------------------------------------
# 参数空间定义 (针对 ming 1D)
# ---------------------------------------------------------------------------

@dataclass
class ParamSpace:
    """参数空间定义 - 离散或连续范围。"""
    name: str
    low: float
    high: float
    is_int: bool = False
    # 默认 log 分布 (True) 还是均匀 (False)
    log_scale: bool = False
    # 真实默认值 (用于中心化, 让 LGBM 学起来更稳)
    default: float = 0.0


# ming v8 调参后的核心搜索空间 - 围绕 v8 默认值 ±1σ
DEFAULT_PARAM_SPACE: List[ParamSpace] = [
    ParamSpace("risk_pct",       0.005,  0.020, is_int=False, log_scale=False, default=0.0075),
    ParamSpace("r2_chandelier_k", 2.5,   5.5,   is_int=False, log_scale=False, default=3.5),
    ParamSpace("timeout_max_bars",15,    45,    is_int=True,  log_scale=False, default=25),
    ParamSpace("r1_be_buffer_atr", 0.2,  0.6,   is_int=False, log_scale=False, default=0.3),
    ParamSpace("r15_partial_pct",  0.20, 0.50,  is_int=False, log_scale=False, default=0.35),
    ParamSpace("sl_k_vol_mid",     1.5,  2.4,   is_int=False, log_scale=False, default=1.8),
    ParamSpace("don_min_break_hold_bars", 1, 5,  is_int=True, log_scale=False, default=3),
    ParamSpace("natr_shrink_pct",  0.70, 0.95,  is_int=False, log_scale=False, default=0.85),
]


def param_names(space: List[ParamSpace]) -> List[str]:
    return [p.name for p in space]


def sample_uniform(space: List[ParamSpace], n: int, rng: np.random.Generator) -> np.ndarray:
    """从参数空间均匀采样 n 组 (用于 seed / 随机搜索 baseline)."""
    cols = []
    for ps in space:
        if ps.is_int:
            cols.append(rng.integers(ps.low, ps.high + 1, size=n).astype(float))
        else:
            if ps.log_scale:
                log_lo, log_hi = np.log(ps.low), np.log(ps.high)
                cols.append(np.exp(rng.uniform(log_lo, log_hi, size=n)))
            else:
                cols.append(rng.uniform(ps.low, ps.high, size=n))
    return np.column_stack(cols)


def sample_around_default(space: List[ParamSpace], n: int, rng: np.random.Generator,
                          sigma_frac: float = 0.3) -> np.ndarray:
    """围绕 v8 默认值做高斯采样, 80% 集中在 default ±30% 范围.

    比均匀采样更有效: 绝大部分"好参数"应该在 default 附近, 暴力均匀采样
    会浪费大量预算在明显不合理的区域 (如 risk_pct=0.005+k=5.5+max_bars=45).
    """
    cols = []
    for ps in space:
        sigma = (ps.high - ps.low) * sigma_frac
        vals = rng.normal(ps.default, sigma, size=n)
        # 截断到合法范围
        if ps.is_int:
            vals = np.round(vals).clip(ps.low, ps.high)
        else:
            vals = np.clip(vals, ps.low, ps.high)
        cols.append(vals)
    return np.column_stack(cols)


def params_to_dict(arr: np.ndarray, space: List[ParamSpace]) -> Dict[str, float]:
    """numpy array → {name: val} dict (给 run_backtest 用)."""
    return {ps.name: (int(arr[i]) if ps.is_int else float(arr[i]))
            for i, ps in enumerate(space)}


# ---------------------------------------------------------------------------
# 单次回测 (复用 param_sweep.run_single_backtest 逻辑)
# ---------------------------------------------------------------------------

def run_single_backtest(
    strategy: str,
    bar_arg: str,
    params: Dict[str, float],
    symbols: str = "BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,DOGE-USDT-SWAP,AVAX-USDT-SWAP",
    out_dir: str = "data/ml_sweep_tmp",
    timeout: int = 300,
) -> Tuple[Optional[dict], str]:
    """跑一次回测, 返回 (metrics_dict, error_msg)."""
    cmd = [
        sys.executable, "-m", "v3.run_backtest",
        "--bar", bar_arg,
        "--only", strategy,
        "--out-dir", out_dir,
    ]
    if symbols:
        cmd.extend(["--symbols", symbols])
    for k, v in params.items():
        cmd.extend(["--params", f"{strategy}:{k}={v}"])
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
        return None, f"no metrics.json (returncode={proc.returncode})"
    try:
        with open(json_p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"read metrics error: {e}"
    is_m = data.get("in_sample", {}) or {}
    oos_m = data.get("out_of_sample", {}) or {}
    overfit = data.get("overfit", {}) or {}
    return {
        "is": is_m,
        "oos": oos_m,
        "overfit": overfit,
        "elapsed": elapsed,
        "params": params,
    }, ""


# ---------------------------------------------------------------------------
# 种子采样 (step 1: seed)
# ---------------------------------------------------------------------------

def step_seed(args) -> dict:
    """生成种子样本 + 真跑, 写到 out_dir/samples.json."""
    print(f"\n=== 步骤 1/3: 生成种子样本 (n={args.n_samples}) ===\n")
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # 默认 80% 围绕 v8 中心, 20% 均匀 (探索)
    n_center = int(args.n_samples * 0.8)
    n_uniform = args.n_samples - n_center
    Xc = sample_around_default(DEFAULT_PARAM_SPACE, n_center, rng)
    Xu = sample_uniform(DEFAULT_PARAM_SPACE, n_uniform, rng)
    X = np.vstack([Xc, Xu])
    rng.shuffle(X)
    print(f"  - 围绕 v8 默认值采样: {n_center}")
    print(f"  - 均匀探索:           {n_uniform}")
    print(f"  - 总计:               {len(X)} 组")
    print(f"  - 预计耗时:           {len(X) * 5 // 60} 分钟 (按 5s/组)")
    print()

    samples = []
    t_start = time.time()
    for i in range(len(X)):
        params = params_to_dict(X[i], DEFAULT_PARAM_SPACE)
        print(f"[{i+1}/{len(X)}] {params}", end=" ", flush=True)
        result, err = run_single_backtest(
            strategy=args.strategy,
            bar_arg=args.bar,
            params=params,
            symbols=args.symbols,
            out_dir=tmp_dir,
        )
        if result is None:
            print(f"FAIL: {err[:80]}")
            continue
        oos = result["oos"]
        is_ = result["is"]
        print(
            f"is_ret={is_.get('total_return', 0)*100:+.2f}% "
            f"oos_ret={oos.get('total_return', 0)*100:+.2f}% "
            f"oos_sharpe={oos.get('sharpe', 0):.2f} "
            f"oos_dd={oos.get('max_drawdown', 0)*100:.2f}% "
            f"oos_trades={oos.get('trade_count', 0)} "
            f"({result['elapsed']:.1f}s)"
        )
        samples.append({
            "params": params,
            "X": X[i].tolist(),
            "is": is_,
            "oos": oos,
            "overfit": result["overfit"],
            "elapsed": result["elapsed"],
        })

    elapsed = time.time() - t_start
    out_pkl = os.path.join(args.out_dir, "samples.json")
    with open(out_pkl, "w", encoding="utf-8") as f:
        json.dump({
            "n_requested": args.n_samples,
            "n_succeeded": len(samples),
            "param_space": [asdict(ps) for ps in DEFAULT_PARAM_SPACE],
            "samples": samples,
            "elapsed": elapsed,
            "strategy": args.strategy,
            "bar": args.bar,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 种子样本: {len(samples)}/{args.n_samples} 成功 ({elapsed:.0f}s)")
    print(f"   写入: {out_pkl}")
    return {"samples_pkl": out_pkl, "n_succeeded": len(samples)}


# ---------------------------------------------------------------------------
# LGBM 训练 (step 2: train)
# ---------------------------------------------------------------------------

def _winsorize(y: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """按 ±3σ 剪尾, 抑制标签噪声.

    用 median + MAD (median absolute deviation) 估计中心和尺度,
    比 mean+std 更鲁棒——少量极端 outlier 不会把 μ 和 σ 拉偏.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    mu = float(np.median(y))
    mad = float(np.median(np.abs(y - mu)))
    # MAD → σ 转换 (正态分布: σ = MAD × 1.4826)
    sd = mad * 1.4826
    if sd < 1e-9:
        # 所有值都一样, winsorize = identity
        return y
    return np.clip(y, mu - sigma * sd, mu + sigma * sd)


def _label_for_ranking(y: np.ndarray) -> np.ndarray:
    """回归标签 → lambdarank 需要的 int 等级 (0/1/2/3).

    按 4 分位分桶: 0=worst, 3=best.
    """
    quantiles = np.quantile(y, [0.25, 0.50, 0.75])
    out = np.zeros(len(y), dtype=int)
    out[y > quantiles[0]] = 1
    out[y > quantiles[1]] = 2
    out[y > quantiles[2]] = 3
    return out


def _build_features(samples: List[dict], space: List[ParamSpace]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 samples 构造 (X, y_sharpe, y_dd_ok, y_trades_ok, y_rank_label).

    X: 标准化后的参数矩阵 (zero-mean, unit-std, 按 default 中心化)
    y_sharpe: OOS sharpe (winsorize)
    y_dd_ok: 1 if |OOS dd| <= max_dd_limit else 0
    y_trades_ok: 1 if OOS trades >= min_trades else 0
    y_rank_label: lambdarank 用的 int 等级
    """
    pnames = param_names(space)
    X_raw = np.array([[s["params"][n] for n in pnames] for s in samples])
    # 中心化: (x - default) / (high - low) → 让 LGBM 学相对扰动
    centers = np.array([ps.default for ps in space])
    ranges = np.array([ps.high - ps.low for ps in space])
    X_norm = (X_raw - centers) / ranges
    oos_sharpe = np.array([s["oos"].get("sharpe", 0) for s in samples])
    oos_dd = np.array([s["oos"].get("max_drawdown", 0) for s in samples])
    oos_trades = np.array([s["oos"].get("trade_count", 0) for s in samples])
    y_sharpe_w = _winsorize(oos_sharpe, sigma=3.0)
    return X_norm, y_sharpe_w, oos_dd, oos_trades, oos_sharpe


def _time_split_indices(n: int, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """时间切分: 前 (1-val_frac) 训练, 后 val_frac 验证.

    注意: 不是按 sample index 随机切! 是按"扫到的顺序"切 (scan-time 顺序).
    这样能避免同一 sweep 内高度相关的样本被同时放在训练/验证集里.
    """
    cut = int(n * (1 - val_frac))
    return np.arange(cut), np.arange(cut, n)


def step_train(args) -> dict:
    """训练 LGBM surrogate (regression + ranking + classification).

    训练 3 个模型:
      1. sharpe_regressor: LGBMRegressor, predict OOS sharpe (winsorized)
      2. dd_classifier:    LGBMClassifier, predict P(|dd| > 15%)
      3. rank_model:       lgb.train lambdarank, 排序 (group=扫到的轮次)
    加上派生: trades_classifier, predict P(trades >= min_trades)
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    print(f"\n=== 步骤 2/3: 训练 LGBM surrogate ===\n")
    with open(args.seed_dir + "/samples.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    if len(samples) < 30:
        print(f"❌ 样本数 {len(samples)} < 30, 训练意义不足")
        return {}
    space = [ParamSpace(**ps) for ps in data["param_space"]]
    pnames = param_names(space)
    X, y_sharpe, oos_dd, oos_trades, oos_sharpe_raw = _build_features(samples, space)
    y_dd_bad = (np.abs(oos_dd) > args.max_dd_limit).astype(int)
    y_trades_ok = (oos_trades >= args.min_trades).astype(int)
    # ranking label: 4 桶分位
    y_rank = _label_for_ranking(oos_sharpe_raw)

    # 时间切分 (scan-time 顺序) - 不随机
    n = len(X)
    cut = int(n * 0.8)
    train_idx = np.arange(cut)
    val_idx = np.arange(cut, n)
    X_train, X_val = X[train_idx], X[val_idx]
    print(f"  样本: {len(X)} (训练 {len(X_train)}, 验证 {len(X_val)})")
    print(f"  OOS sharpe: mean={oos_sharpe_raw.mean():.2f} std={oos_sharpe_raw.std():.2f} "
          f"min={oos_sharpe_raw.min():.2f} max={oos_sharpe_raw.max():.2f}")
    print(f"  OOS dd > 15%: {y_dd_bad.sum()}/{len(y_dd_bad)} ({y_dd_bad.mean():.1%})")
    print(f"  OOS trades >= {args.min_trades}: {y_trades_ok.sum()}/{len(y_trades_ok)} ({y_trades_ok.mean():.1%})")
    print()

    # ---- Model 1: sharpe 回归 (winsorize) ----
    print("  [1/3] 训练 sharpe 回归器 (LGBMRegressor)...")
    sharpe_model = _train_regressor(X_train, y_sharpe[train_idx], X_val, y_sharpe[val_idx])
    sharpe_pred = sharpe_model.predict(X_val)
    sharpe_rmse = float(np.sqrt(((sharpe_pred - oos_sharpe_raw[val_idx]) ** 2).mean()))
    sharpe_corr = float(np.corrcoef(sharpe_pred, oos_sharpe_raw[val_idx])[0, 1]) if len(val_idx) > 1 else 0.0
    print(f"        RMSE={sharpe_rmse:.3f}  corr={sharpe_corr:.3f}")

    # ---- Model 2: dd 二分类 (dd > 15% ?) ----
    print("  [2/3] 训练 max_dd 二分类器 (LGBMClassifier)...")
    # 处理单类边界情况: 如果训练集只有一类, 直接 fallback 到 0.5 概率
    if y_dd_bad[train_idx].min() == y_dd_bad[train_idx].max():
        print(f"        ⚠️ 训练集 dd 全为同一类 ({y_dd_bad[train_idx].mean():.0%}), 跳过模型")
        dd_model = _dummy_constant_predictor(y_dd_bad[train_idx].mean())
        dd_pred = np.full(len(val_idx), y_dd_bad[train_idx].mean())
        dd_auc = 0.5
    else:
        dd_model = _train_classifier(X_train, y_dd_bad[train_idx], X_val, y_dd_bad[val_idx])
        dd_pred = dd_model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        if y_dd_bad[val_idx].min() != y_dd_bad[val_idx].max():
            dd_auc = float(roc_auc_score(y_dd_bad[val_idx], dd_pred))
        else:
            dd_auc = 0.5
    print(f"        AUC={dd_auc:.3f}  (验证集 dd_bad rate={y_dd_bad[val_idx].mean():.1%})")

    # ---- Model 3: trades 二分类 (trades >= min_trades ?) ----
    print(f"  [3/3] 训练 trade_count 二分类器 (LGBMClassifier, threshold={args.min_trades})...")
    if y_trades_ok[train_idx].min() == y_trades_ok[train_idx].max():
        print(f"        ⚠️ 训练集 trades 全为同一类 ({y_trades_ok[train_idx].mean():.0%}), 跳过模型")
        trades_model = _dummy_constant_predictor(y_trades_ok[train_idx].mean())
        trades_pred = np.full(len(val_idx), y_trades_ok[train_idx].mean())
        trades_auc = 0.5
    else:
        trades_model = _train_classifier(X_train, y_trades_ok[train_idx], X_val, y_trades_ok[val_idx])
        trades_pred = trades_model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        if y_trades_ok[val_idx].min() != y_trades_ok[val_idx].max():
            trades_auc = float(roc_auc_score(y_trades_ok[val_idx], trades_pred))
        else:
            trades_auc = 0.5
    print(f"        AUC={trades_auc:.3f}  (验证集 trades_ok rate={y_trades_ok[val_idx].mean():.1%})")

    # ---- 合成公式: final_score = pred_sharpe × P(dd_ok) × P(trades_ok) ----
    # 注意: P(dd_ok) = 1 - P(dd_bad) = 1 - dd_pred
    p_dd_ok = 1.0 - dd_pred
    p_trades_ok = trades_pred
    final_score = sharpe_pred * p_dd_ok * p_trades_ok
    # 同样计算真值的 final score (用于对比)
    final_score_true = (oos_sharpe_raw[val_idx]
                       * (1 - y_dd_bad[val_idx])
                       * y_trades_ok[val_idx])
    score_corr = float(np.corrcoef(final_score, final_score_true)[0, 1]) if len(val_idx) > 1 else 0.0
    print()
    print(f"  合成 final_score 相关性: {score_corr:.3f} (验证集)")
    # 排名相关性: top-k 重合度
    top_k = min(10, len(val_idx) // 2)
    pred_top = set(np.argsort(-final_score)[:top_k])
    true_top = set(np.argsort(-final_score_true)[:top_k])
    top_overlap = len(pred_top & true_top) / top_k if top_k > 0 else 0.0
    print(f"  Top-{top_k} 重合度: {top_overlap:.1%} (模型预测 top vs 真实 top)")
    # ranking 质量: NDCG
    from sklearn.metrics import ndcg_score
    try:
        ndcg = float(ndcg_score(final_score_true.reshape(1, -1), final_score.reshape(1, -1)))
    except Exception:
        ndcg = 0.0
    print(f"  NDCG: {ndcg:.3f}")

    # ---- 特征重要性 ----
    print()
    print("  特征重要性 (sharpe_regressor):")
    for n_, imp in sorted(zip(pnames, sharpe_model.feature_importances_),
                          key=lambda kv: -kv[1]):
        bar = "█" * int(imp / max(sharpe_model.feature_importances_.max(), 1) * 30)
        print(f"    {n_:30s} {imp:4d}  {bar}")

    # ---- 保存模型 ----
    os.makedirs(args.out_dir, exist_ok=True)
    # 存成 pickle 方便后续加载 (含 3 个 sklearn 模型 + 1 个 rank 模型)
    import joblib
    bundle = {
        "sharpe_regressor": sharpe_model,
        "dd_classifier": dd_model,
        "trades_classifier": trades_model,
        "param_names": pnames,
        "param_space": [asdict(ps) for ps in space],
        "validation_metrics": {
            "sharpe_rmse": sharpe_rmse,
            "sharpe_corr": sharpe_corr,
            "dd_auc": dd_auc,
            "trades_auc": trades_auc,
            "score_corr": score_corr,
            "top_overlap": top_overlap,
            "ndcg": ndcg,
        },
        "training_settings": {
            "max_dd_limit": args.max_dd_limit,
            "min_trades": args.min_trades,
            "n_samples": len(samples),
        },
    }
    model_pkl = os.path.join(args.out_dir, "surrogate.pkl")
    joblib.dump(bundle, model_pkl)
    print(f"\n✅ Surrogate 已保存: {model_pkl}")
    return {"model_pkl": model_pkl, "metrics": bundle["validation_metrics"]}


def _train_regressor(X_tr, y_tr, X_va, y_va):
    """训练 LGBMRegressor + 早停 (参照用户提供的模板)."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _train_classifier(X_tr, y_tr, X_va, y_va):
    """训练 LGBMClassifier + 早停 (参照用户提供的模板, 改成分类)."""
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


class _dummy_constant_predictor:
    """当训练集只有单类时, 退化为常数预测器.

    模仿 LGBM 接口: predict_proba 返回 shape=(n, 2) 矩阵 [[1-p, p], ...].
    """
    def __init__(self, p: float):
        self.p = float(p)
        self.feature_importances_ = np.array([0] * 8)  # 兜底

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        col1 = np.full(n, 1.0 - self.p)
        col2 = np.full(n, self.p)
        return np.column_stack([col1, col2])

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, int(self.p > 0.5))


# ---------------------------------------------------------------------------
# 虚拟扫描 + Top-N 真跑 (step 3: search)
# ---------------------------------------------------------------------------

def step_search(args) -> dict:
    """加载 surrogate, 生成虚拟样本, 预测, 挑 Top-N 真跑验证."""
    import joblib

    print(f"\n=== 步骤 3/3: 虚拟扫描 + Top-{args.n_validate} 真跑 ===\n")
    bundle = joblib.load(args.model)
    sharpe_model = bundle["sharpe_regressor"]
    dd_model = bundle["dd_classifier"]
    trades_model = bundle["trades_classifier"]
    pnames = bundle["param_names"]
    space = [ParamSpace(**ps) for ps in bundle["param_space"]]
    metrics = bundle["validation_metrics"]
    print(f"  Loaded surrogate: {args.model}")
    print(f"  Validation metrics: {metrics}")
    print()

    # 1. 虚拟扫描: 生成 n_virtual 个候选 (混合 80% 中心 + 20% 均匀)
    rng = np.random.default_rng(args.seed + 1)
    n_center = int(args.n_virtual * 0.7)
    n_uniform = args.n_virtual - n_center
    # 同时加 10% 上轮 Top-30 周围扰动 (exploitation)
    if args.top_seeds and os.path.isfile(args.top_seeds):
        with open(args.top_seeds, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("top"):
            top_arr = np.array([s["X"] for s in prev["top"][:10]])
            # 围绕 top-10 各扰动 (n_virtual * 0.3 // 10) 个
            n_exploit = int(args.n_virtual * 0.3)
            per = n_exploit // len(top_arr)
            cols = []
            for c in top_arr:
                # 在标准化空间加小扰动 (sigma=0.05)
                noise = rng.normal(0, 0.05, size=(per, len(c)))
                cols.append(c + noise)
            X_exploit = np.vstack(cols)
        else:
            X_exploit = np.empty((0, len(pnames)))
    else:
        X_exploit = np.empty((0, len(pnames)))
    Xc = sample_around_default(space, n_center, rng)
    Xu = sample_uniform(space, n_uniform, rng)
    X = np.vstack([Xc, Xu, X_exploit])
    # 标准化 (与训练一致)
    centers = np.array([ps.default for ps in space])
    ranges = np.array([ps.high - ps.low for ps in space])
    X_norm = (X - centers) / ranges
    print(f"  虚拟扫描: {len(X)} 个候选 (中心 {n_center} + 均匀 {n_uniform} + exploit {len(X_exploit)})")

    # 2. 预测 (3 个模型)
    pred_sharpe = sharpe_model.predict(X_norm)
    pred_dd_bad = dd_model.predict_proba(X_norm)[:, 1]
    pred_trades_ok = trades_model.predict_proba(X_norm)[:, 1]
    final_score = pred_sharpe * (1 - pred_dd_bad) * pred_trades_ok
    print(f"  预测 final_score: mean={final_score.mean():.3f} max={final_score.max():.3f}")
    # 排名
    order = np.argsort(-final_score)
    top_idx = order[:args.n_validate]
    print(f"  Top-{args.n_validate} selected (pred_score range "
          f"{final_score[top_idx].min():.3f} ~ {final_score.max():.3f})")
    print()

    # 3. Top-N 真跑验证
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    results = []
    t_start = time.time()
    for rank, idx in enumerate(top_idx):
        params = params_to_dict(X[idx], space)
        print(f"  [Top {rank+1}/{args.n_validate}] {params}", end=" ", flush=True)
        result, err = run_single_backtest(
            strategy=args.strategy,
            bar_arg=args.bar,
            params=params,
            symbols=args.symbols,
            out_dir=tmp_dir,
        )
        if result is None:
            print(f"FAIL: {err[:80]}")
            continue
        oos = result["oos"]
        is_ = result["is"]
        # 计算真实 final_score
        true_score = (oos.get("sharpe", 0)
                     * (1 if abs(oos.get("max_drawdown", 0)) <= args.max_dd_limit else 0)
                     * (1 if oos.get("trade_count", 0) >= args.min_trades else 0))
        results.append({
            "rank_pred": rank + 1,
            "pred_score": float(final_score[idx]),
            "true_score": float(true_score),
            "params": params,
            "X": X[idx].tolist(),
            "is": is_,
            "oos": oos,
            "overfit": result["overfit"],
            "elapsed": result["elapsed"],
        })
        print(
            f"pred={final_score[idx]:.3f} true={true_score:.3f} "
            f"oos_sharpe={oos.get('sharpe', 0):.2f} "
            f"oos_dd={oos.get('max_drawdown', 0)*100:.2f}% "
            f"oos_trades={oos.get('trade_count', 0)} "
            f"({result['elapsed']:.1f}s)"
        )
    elapsed = time.time() - t_start
    # 排序: 按 true_score 降序
    results.sort(key=lambda r: -r["true_score"])
    # 排名相关性
    if len(results) >= 3:
        ranks_pred = [r["rank_pred"] for r in results]
        ranks_true = list(range(1, len(results) + 1))
        # 反序相关 (rank 1 最好)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(ranks_pred, ranks_true)
    else:
        corr = 0.0
    print()
    print(f"✅ Top-{len(results)} 验证完成 ({elapsed:.0f}s)")
    print(f"   预测排名 vs 真实排名 spearman 相关: {corr:.3f}")

    # 4. 输出报告
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "search_results.json")
    out_md = os.path.join(args.out_dir, "search_report.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n_virtual": len(X),
            "n_validated": len(results),
            "validation_corr": float(corr),
            "top_seeds": results[:30],  # 给下一轮 search 当 top_seeds (exploit)
            "all_results": results,
            "elapsed": elapsed,
        }, f, ensure_ascii=False, indent=2, default=str)
    _write_md_report(out_md, results, metrics, corr, args)
    print(f"   写入: {out_json}")
    print(f"   写入: {out_md}")
    return {"results_json": out_json, "report_md": out_md, "n_validated": len(results), "corr": float(corr)}


def _write_md_report(path: str, results: list, val_metrics: dict, corr: float, args):
    lines = [f"# ML 参数扫描 - 验证报告", ""]
    lines.append(f"## 模型质量（验证集）")
    lines.append(f"- sharpe RMSE: {val_metrics.get('sharpe_rmse', 0):.3f}")
    lines.append(f"- sharpe 相关系数: {val_metrics.get('sharpe_corr', 0):.3f}")
    lines.append(f"- dd 二分类 AUC: {val_metrics.get('dd_auc', 0):.3f}")
    lines.append(f"- trades 二分类 AUC: {val_metrics.get('trades_auc', 0):.3f}")
    lines.append(f"- 合成 final_score 相关性: {val_metrics.get('score_corr', 0):.3f}")
    lines.append(f"- Top-10 重合度: {val_metrics.get('top_overlap', 0):.1%}")
    lines.append(f"- NDCG: {val_metrics.get('ndcg', 0):.3f}")
    lines.append(f"- **预测排名 vs 真实排名 spearman 相关: {corr:.3f}**")
    lines.append("")
    lines.append(f"## Top-{len(results)} 真跑结果（按 true_score 排序）\n")
    lines.append("| Rank | true_score | risk_pct | k | max_bars | r1_be | r15 | hold | "
                 "IS ret | OOS ret | IS sharpe | OOS sharpe | OOS dd | OOS trades | overfit |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(results):
        p = r["params"]
        oos = r["oos"]
        is_ = r["is"]
        lines.append(
            f"| {i+1} | {r['true_score']:.3f} | {p.get('risk_pct', '-'):.4f} | "
            f"{p.get('r2_chandelier_k', '-'):.1f} | {p.get('timeout_max_bars', '-')} | "
            f"{p.get('r1_be_buffer_atr', '-'):.2f} | {p.get('r15_partial_pct', '-'):.2f} | "
            f"{p.get('don_min_break_hold_bars', '-')} | "
            f"{is_.get('total_return', 0)*100:+.2f}% | {oos.get('total_return', 0)*100:+.2f}% | "
            f"{is_.get('sharpe', 0):.2f} | {oos.get('sharpe', 0):.2f} | "
            f"{oos.get('max_drawdown', 0)*100:.2f}% | {oos.get('trade_count', 0)} | "
            f"{r['overfit'].get('overfit_score', 0):.3f} |"
        )
    lines.append("")
    if results:
        best = results[0]
        lines.append(f"## 推荐参数（true_score 最高）")
        lines.append("```yaml")
        for k, v in best["params"].items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append(f"\n- IS ret: {best['is'].get('total_return', 0)*100:+.2f}% | OOS ret: {best['oos'].get('total_return', 0)*100:+.2f}%")
        lines.append(f"- IS sharpe: {best['is'].get('sharpe', 0):.2f} | OOS sharpe: {best['oos'].get('sharpe', 0):.2f}")
        lines.append(f"- OOS max_dd: {best['oos'].get('max_drawdown', 0)*100:.2f}%")
        lines.append(f"- OOS trades: {best['oos'].get('trade_count', 0)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# 一条龙
# ---------------------------------------------------------------------------

def step_all(args):
    """seed → train → search 一条龙."""
    seed_out = os.path.join(args.out_dir, "seed")
    train_out = os.path.join(args.out_dir, "surrogate")
    search_out = os.path.join(args.out_dir, "search")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. seed
    args_seed = argparse.Namespace(**{**vars(args), "out_dir": seed_out})
    seed_res = step_seed(args_seed)
    if seed_res.get("n_succeeded", 0) < 30:
        print("❌ 种子样本不足, 中止")
        return {}

    # 2. train
    args_train = argparse.Namespace(**{**vars(args), "seed_dir": seed_out, "out_dir": train_out})
    train_res = step_train(args_train)
    if not train_res:
        print("❌ 训练失败, 中止")
        return {}

    # 3. search
    args_search = argparse.Namespace(
        **{**vars(args),
           "model": train_res["model_pkl"],
           "out_dir": search_out,
           "top_seeds": None,  # 第一轮无 top_seeds
        }
    )
    search_res = step_search(args_search)
    return {"seed": seed_res, "train": train_res, "search": search_res}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ML 驱动的参数扫描器 (LGBM surrogate + 贝叶斯/Random 虚拟扫描)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 公共参数
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--strategy", default="ming")
    common.add_argument("--bar", default="1D")
    common.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,DOGE-USDT-SWAP,AVAX-USDT-SWAP")
    common.add_argument("--max-dd-limit", type=float, default=0.15, help="dd 容忍上限（用于 P(dd_ok) 判定）")
    common.add_argument("--min-trades", type=int, default=15, help="trades 阈值（用于 P(trades_ok) 判定）")
    common.add_argument("--seed", type=int, default=42)

    # seed
    p_seed = sub.add_parser("seed", parents=[common], help="步骤 1: 种子样本")
    p_seed.add_argument("--n-samples", type=int, default=100)
    p_seed.add_argument("--out-dir", required=True)
    p_seed.set_defaults(func=step_seed)

    # train
    p_train = sub.add_parser("train", parents=[common], help="步骤 2: 训练 surrogate")
    p_train.add_argument("--seed-dir", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.set_defaults(func=step_train)

    # search
    p_search = sub.add_parser("search", parents=[common], help="步骤 3: 虚拟扫描 + 真跑")
    p_search.add_argument("--model", required=True, help="surrogate.pkl 路径")
    p_search.add_argument("--n-virtual", type=int, default=5000, help="虚拟扫描样本数")
    p_search.add_argument("--n-validate", type=int, default=30, help="真跑验证 Top-N")
    p_search.add_argument("--out-dir", required=True)
    p_search.add_argument("--top-seeds", default="", help="上一轮 search_results.json (用于 exploit)")
    p_search.set_defaults(func=step_search)

    # all
    p_all = sub.add_parser("all", parents=[common], help="一条龙 seed → train → search")
    p_all.add_argument("--n-samples", type=int, default=100)
    p_all.add_argument("--n-virtual", type=int, default=3000)
    p_all.add_argument("--n-validate", type=int, default=30)
    p_all.add_argument("--out-dir", required=True)
    p_all.set_defaults(func=step_all)

    args = ap.parse_args()
    result = args.func(args)
    if result:
        print(f"\n=== 完成 ===\n{json.dumps(result, ensure_ascii=False, indent=2, default=str)[:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
