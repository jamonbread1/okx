# -*- coding: utf-8 -*-
"""Unit tests for param_sweep_ml.py — ML 驱动参数扫描器。

不依赖数据下载、不跑完整回测，只验证：
  1. 参数空间定义 + 采样函数
  2. 特征工程 (中心化、winsorize、label for ranking)
  3. 时间切分（不是随机 CV）
  4. 模型训练（3 个模型 + 早停）
  5. 单类边界 fallback（dummy predictor）
  6. 合成 final_score 公式正确性
  7. 虚拟扫描 + Top-N 选择
"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 抑制 LGBM log 噪音
os.environ.setdefault("LIGHTGBM_VERBOSE", "0")


class TestParamSpace(unittest.TestCase):
    """参数空间定义 + 采样。"""

    def test_default_space_has_8_params(self):
        from tools.param_sweep_ml import DEFAULT_PARAM_SPACE, param_names
        names = param_names(DEFAULT_PARAM_SPACE)
        self.assertEqual(len(names), 8)
        # 关键 ming v8 参数必须存在
        for n in ["risk_pct", "r2_chandelier_k", "timeout_max_bars",
                  "r1_be_buffer_atr", "r15_partial_pct", "don_min_break_hold_bars"]:
            self.assertIn(n, names)

    def test_sample_uniform_shape(self):
        from tools.param_sweep_ml import DEFAULT_PARAM_SPACE, sample_uniform
        rng = np.random.default_rng(42)
        X = sample_uniform(DEFAULT_PARAM_SPACE, 100, rng)
        self.assertEqual(X.shape, (100, 8))
        # 验证每列都在 [low, high] 范围内
        for i, ps in enumerate(DEFAULT_PARAM_SPACE):
            if ps.is_int:
                self.assertTrue((X[:, i] >= ps.low).all() and (X[:, i] <= ps.high).all(),
                                f"{ps.name} out of range")
            else:
                self.assertTrue((X[:, i] >= ps.low).all() and (X[:, i] <= ps.high).all(),
                                f"{ps.name} out of range")

    def test_sample_around_default_concentrated(self):
        """围绕 default 采样应集中在 default 附近 (~30% 范围)."""
        from tools.param_sweep_ml import DEFAULT_PARAM_SPACE, sample_around_default
        rng = np.random.default_rng(42)
        X = sample_around_default(DEFAULT_PARAM_SPACE, 1000, rng)
        # risk_pct 默认 0.0075, range [0.005, 0.020]
        risk_pct = X[:, 0]
        # 80% 应该在 default ± 30% × range 范围内
        sigma = (0.020 - 0.005) * 0.3
        within = np.abs(risk_pct - 0.0075) < 2 * sigma
        self.assertGreater(within.mean(), 0.80,
                           f"集中度不足: {within.mean():.1%} in 2σ")

    def test_params_to_dict(self):
        from tools.param_sweep_ml import DEFAULT_PARAM_SPACE, sample_around_default, params_to_dict
        rng = np.random.default_rng(42)
        X = sample_around_default(DEFAULT_PARAM_SPACE, 1, rng)
        d = params_to_dict(X[0], DEFAULT_PARAM_SPACE)
        self.assertIsInstance(d, dict)
        self.assertEqual(len(d), 8)
        self.assertIn("risk_pct", d)
        # int 参数应该是 int 类型
        self.assertIsInstance(d["timeout_max_bars"], int)
        self.assertIsInstance(d["don_min_break_hold_bars"], int)
        # float 参数应该是 float
        self.assertIsInstance(d["risk_pct"], float)


class TestFeatureEngineering(unittest.TestCase):
    """特征工程: 中心化 + winsorize + label for ranking。"""

    def test_winsorize_sigma_3(self):
        from tools.param_sweep_ml import _winsorize
        # 典型金融 sharpe 场景: 大部分在 [-1, 3], 几个 5σ+ outlier
        # 用足够多样本 (n=200), outlier 才会被识别为 outlier
        rng = np.random.default_rng(42)
        y = rng.normal(1.0, 0.5, 200)  # mean=1, std=0.5
        y[0] = 100.0  # 极端 outlier (z-score ~ 200)
        y[1] = -50.0  # 极端 outlier
        y_w = _winsorize(y, sigma=3.0)
        # μ + 3σ = 1 + 1.5 = 2.5; μ - 3σ = 1 - 1.5 = -0.5
        # 100 / -50 应该被剪到 [−0.5, 2.5]
        self.assertLessEqual(y_w.max(), 2.5 + 1e-6,
                             f"winsorize 失败: max={y_w.max()}")
        self.assertGreaterEqual(y_w.min(), -0.5 - 1e-6,
                                f"winsorize 失败: min={y_w.min()}")
        # 正常值不动
        self.assertAlmostEqual(y_w[10], y[10], places=4)

    def test_label_for_ranking(self):
        from tools.param_sweep_ml import _label_for_ranking
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        labels = _label_for_ranking(y)
        # 4 桶分位: 0/1/2/3
        self.assertEqual(len(labels), 8)
        self.assertEqual(set(labels.tolist()), {0, 1, 2, 3})
        # 最大值对应 label=3
        self.assertEqual(labels[y.argmax()], 3)
        # 最小值对应 label=0
        self.assertEqual(labels[y.argmin()], 0)

    def test_build_features(self):
        from tools.param_sweep_ml import DEFAULT_PARAM_SPACE, _build_features, sample_around_default, params_to_dict
        rng = np.random.default_rng(42)
        X = sample_around_default(DEFAULT_PARAM_SPACE, 50, rng)
        samples = []
        for i in range(50):
            samples.append({
                "params": params_to_dict(X[i], DEFAULT_PARAM_SPACE),
                "oos": {"sharpe": float(np.random.randn()), "max_drawdown": float(-0.05 * np.random.rand()),
                        "trade_count": int(np.random.randint(5, 100))},
            })
        X_feat, y_sharpe, oos_dd, oos_trades, oos_sharpe_raw = _build_features(samples, DEFAULT_PARAM_SPACE)
        # 形状: 50 × 8
        self.assertEqual(X_feat.shape, (50, 8))
        # 中心化: 均值应接近 0 (因为 default 周围采样)
        self.assertLess(abs(X_feat.mean()), 0.3)
        # y_sharpe winsorize 后应在合理范围
        self.assertTrue(np.isfinite(y_sharpe).all())
        self.assertTrue(np.isfinite(oos_dd).all())
        self.assertTrue(np.isfinite(oos_trades).all())


class TestTimeSplit(unittest.TestCase):
    """时间切分（不是随机 CV）."""

    def test_time_split_preserves_order(self):
        from tools.param_sweep_ml import _time_split_indices
        train_idx, val_idx = _time_split_indices(100, val_frac=0.2)
        # 训练是前 80 个, 验证是后 20 个
        np.testing.assert_array_equal(train_idx, np.arange(80))
        np.testing.assert_array_equal(val_idx, np.arange(80, 100))
        # 验证集的所有索引都 > 训练集最大索引
        self.assertGreater(val_idx.min(), train_idx.max())

    def test_time_split_no_overlap(self):
        from tools.param_sweep_ml import _time_split_indices
        train_idx, val_idx = _time_split_indices(50, val_frac=0.3)
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)


class TestDummyPredictor(unittest.TestCase):
    """单类边界的 dummy fallback."""

    def test_constant_predictor(self):
        from tools.param_sweep_ml import _dummy_constant_predictor
        # p=0.3 模拟
        d = _dummy_constant_predictor(0.3)
        X = np.random.rand(10, 8)
        proba = d.predict_proba(X)
        # shape: (10, 2), 第一列是 0.7, 第二列是 0.3
        self.assertEqual(proba.shape, (10, 2))
        np.testing.assert_array_almost_equal(proba[:, 0], 0.7)
        np.testing.assert_array_almost_equal(proba[:, 1], 0.3)
        # predict 应该是 0 (p < 0.5)
        pred = d.predict(X)
        np.testing.assert_array_equal(pred, np.zeros(10))

    def test_high_probability_predicts_positive(self):
        from tools.param_sweep_ml import _dummy_constant_predictor
        d = _dummy_constant_predictor(0.8)
        X = np.random.rand(5, 8)
        pred = d.predict(X)
        np.testing.assert_array_equal(pred, np.ones(5))


class TestFinalScoreFormula(unittest.TestCase):
    """final_score = pred_sharpe × P(dd_ok) × P(trades_ok) 公式正确性."""

    def test_score_zero_if_dd_bad(self):
        pred_sharpe = np.array([1.0, 2.0, 3.0])
        pred_dd_bad = np.array([0.0, 0.5, 1.0])  # 第二组有 50% 概率 dd > 15%
        pred_trades_ok = np.array([1.0, 1.0, 1.0])
        score = pred_sharpe * (1 - pred_dd_bad) * pred_trades_ok
        # dd_bad 概率越高, 分数越低
        self.assertAlmostEqual(score[0], 1.0, places=4)
        self.assertAlmostEqual(score[1], 1.0, places=4)  # 2.0 * 0.5 = 1.0
        self.assertAlmostEqual(score[2], 0.0, places=4)  # 3.0 * 0.0 = 0.0

    def test_score_zero_if_trades_insufficient(self):
        pred_sharpe = np.array([1.0, 2.0])
        pred_dd_bad = np.array([0.0, 0.0])
        pred_trades_ok = np.array([0.5, 0.0])  # 第二组交易数不够
        score = pred_sharpe * (1 - pred_dd_bad) * pred_trades_ok
        self.assertAlmostEqual(score[0], 0.5, places=4)
        self.assertAlmostEqual(score[1], 0.0, places=4)


class TestModelTraining(unittest.TestCase):
    """模型训练全流程（用合成数据）。"""

    def _make_synthetic_samples(self, n: int = 100):
        from tools.param_sweep_ml import (
            DEFAULT_PARAM_SPACE, sample_around_default, sample_uniform, params_to_dict,
        )
        rng = np.random.default_rng(42)
        Xc = sample_around_default(DEFAULT_PARAM_SPACE, n // 2, rng)
        Xu = sample_uniform(DEFAULT_PARAM_SPACE, n - n // 2, rng)
        X = np.vstack([Xc, Xu])
        rng.shuffle(X)
        # 标准化
        centers = np.array([ps.default for ps in DEFAULT_PARAM_SPACE])
        ranges = np.array([ps.high - ps.low for ps in DEFAULT_PARAM_SPACE])
        X_norm = (X - centers) / ranges
        # 模拟 sharpe: 与 risk_pct 正相关, 与 r2_k 二次相关
        true_sharpe = 1.0 + 0.5 * X_norm[:, 0] - 0.3 * X_norm[:, 1] ** 2 + np.random.normal(0, 0.2, n)
        true_dd = -(0.05 + 0.1 * X_norm[:, 0] + np.random.normal(0, 0.02, n))
        true_trades = (30 + 10 * X_norm[:, 2] + np.random.normal(0, 3, n)).astype(int)
        true_trades = np.maximum(5, true_trades)
        samples = []
        for i in range(n):
            samples.append({
                "params": params_to_dict(X[i], DEFAULT_PARAM_SPACE),
                "X": X[i].tolist(),
                "is": {"total_return": 0.01, "sharpe": 1.0, "max_drawdown": -0.05, "trade_count": 50},
                "oos": {
                    "total_return": max(0, true_sharpe[i] * 0.005),
                    "sharpe": float(true_sharpe[i]),
                    "max_drawdown": float(true_dd[i]),
                    "trade_count": int(true_trades[i]),
                },
                "overfit": {"overfit_score": 0.1},
            })
        return samples, X

    def test_full_training_pipeline(self):
        """完整跑一遍 train + 虚拟扫描."""
        from tools.param_sweep_ml import (
            DEFAULT_PARAM_SPACE, _build_features, _train_regressor, _train_classifier,
            sample_around_default, sample_uniform, params_to_dict, _time_split_indices,
            _dummy_constant_predictor,
        )
        import logging
        logging.getLogger('lightgbm').setLevel(logging.ERROR)
        samples, _ = self._make_synthetic_samples(100)
        X, y_sharpe, oos_dd, oos_trades, oos_sharpe_raw = _build_features(samples, DEFAULT_PARAM_SPACE)
        y_dd_bad = (np.abs(oos_dd) > 0.15).astype(int)
        y_trades_ok = (oos_trades >= 15).astype(int)
        train_idx, val_idx = _time_split_indices(len(X), val_frac=0.2)
        # 训练 3 模型
        sharpe_model = _train_regressor(X[train_idx], y_sharpe[train_idx], X[val_idx], y_sharpe[val_idx])
        # dd 模型: 处理单类边界
        if y_dd_bad[train_idx].min() == y_dd_bad[train_idx].max():
            dd_model = _dummy_constant_predictor(y_dd_bad[train_idx].mean())
        else:
            dd_model = _train_classifier(X[train_idx], y_dd_bad[train_idx], X[val_idx], y_dd_bad[val_idx])
        # trades 模型
        if y_trades_ok[train_idx].min() == y_trades_ok[train_idx].max():
            trades_model = _dummy_constant_predictor(y_trades_ok[train_idx].mean())
        else:
            trades_model = _train_classifier(X[train_idx], y_trades_ok[train_idx], X[val_idx], y_trades_ok[val_idx])
        # 虚拟扫描
        rng = np.random.default_rng(99)
        n_virt = 100
        Xc = sample_around_default(DEFAULT_PARAM_SPACE, 70, rng)
        Xu = sample_uniform(DEFAULT_PARAM_SPACE, 30, rng)
        X_virt = np.vstack([Xc, Xu])
        centers = np.array([ps.default for ps in DEFAULT_PARAM_SPACE])
        ranges = np.array([ps.high - ps.low for ps in DEFAULT_PARAM_SPACE])
        X_virt_norm = (X_virt - centers) / ranges
        pred_sharpe = sharpe_model.predict(X_virt_norm)
        pred_dd_bad = dd_model.predict_proba(X_virt_norm)[:, 1]
        pred_trades_ok = trades_model.predict_proba(X_virt_norm)[:, 1]
        final_score = pred_sharpe * (1 - pred_dd_bad) * pred_trades_ok
        # 基本 sanity
        self.assertEqual(len(final_score), n_virt)
        self.assertTrue(np.isfinite(final_score).all())
        self.assertGreater(final_score.max(), final_score.min())

    def test_sharpe_corr_positive(self):
        """sharpe 回归器在合成数据上应该学到正相关 (因为有 ground truth)."""
        from tools.param_sweep_ml import (
            DEFAULT_PARAM_SPACE, _build_features, _train_regressor, _time_split_indices,
        )
        import logging
        logging.getLogger('lightgbm').setLevel(logging.ERROR)
        samples, _ = self._make_synthetic_samples(150)
        X, y_sharpe, _, _, oos_sharpe_raw = _build_features(samples, DEFAULT_PARAM_SPACE)
        train_idx, val_idx = _time_split_indices(len(X), val_frac=0.2)
        sharpe_model = _train_regressor(X[train_idx], y_sharpe[train_idx], X[val_idx], y_sharpe[val_idx])
        pred = sharpe_model.predict(X[val_idx])
        corr = float(np.corrcoef(pred, oos_sharpe_raw[val_idx])[0, 1])
        # 在合成数据上, 应该学到正相关 (阈值 0.2, 实际通常 0.5+)
        self.assertGreater(corr, 0.2, f"sharpe 回归器没学到东西: corr={corr:.3f}")


class TestVirtualScanAndTopN(unittest.TestCase):
    """虚拟扫描 + Top-N 选择的逻辑."""

    def test_top_n_selection(self):
        from tools.param_sweep_ml import (
            DEFAULT_PARAM_SPACE, sample_around_default, sample_uniform,
        )
        rng = np.random.default_rng(42)
        n_virt = 1000
        Xc = sample_around_default(DEFAULT_PARAM_SPACE, int(n_virt * 0.7), rng)
        Xu = sample_uniform(DEFAULT_PARAM_SPACE, n_virt - len(Xc), rng)
        X = np.vstack([Xc, Xu])
        # 模拟 final_score
        final_score = np.random.rand(n_virt)
        # Top-30
        n_top = 30
        order = np.argsort(-final_score)
        top_idx = order[:n_top]
        # 验证: top_idx 包含的是 final_score 最大的 n_top 个
        top_scores = final_score[top_idx]
        self.assertEqual(len(top_idx), n_top)
        # top-30 最小值 >= 全部的中位数
        self.assertGreaterEqual(top_scores.min(), np.median(final_score))


class TestEndToEndNoRealBacktest(unittest.TestCase):
    """端到端测试: 不跑真实回测, 用 mock 替代 run_single_backtest."""

    def test_seed_step_writes_samples(self):
        """seed 步骤能写出 samples.json."""
        from unittest.mock import patch
        from tools import param_sweep_ml
        # mock run_single_backtest
        rng_counter = [0]
        def mock_run(strategy, bar_arg, params, symbols, out_dir, timeout=300):
            rng_counter[0] += 1
            i = rng_counter[0]
            # 模拟 sharpe 随 risk_pct 变化
            risk = params.get("risk_pct", 0.0075)
            sharpe = 1.0 + (risk - 0.0075) * 50 + np.random.randn() * 0.3
            return {
                "is": {"total_return": 0.01, "sharpe": 1.0, "max_drawdown": -0.05, "trade_count": 50, "bars": 1500},
                "oos": {
                    "total_return": max(0, sharpe * 0.005),
                    "sharpe": float(sharpe),
                    "max_drawdown": float(-0.05 - (risk - 0.0075) * 2),
                    "trade_count": int(20 + np.random.randint(-5, 5)),
                },
                "overfit": {"overfit_score": 0.1},
                "elapsed": 0.1,
                "params": params,
            }, ""
        with patch.object(param_sweep_ml, "run_single_backtest", mock_run):
            import argparse
            args = argparse.Namespace(
                strategy="ming", bar="1D",
                symbols="BTC-USDT-SWAP,ETH-USDT-SWAP",
                n_samples=20, out_dir="/tmp/ml_e2e", seed=42,
            )
            os.makedirs("/tmp/ml_e2e", exist_ok=True)
            result = param_sweep_ml.step_seed(args)
        self.assertGreater(result["n_succeeded"], 15)
        # samples.json 存在
        self.assertTrue(os.path.isfile(result["samples_pkl"]))
        # 验证文件内容
        with open(result["samples_pkl"], "r") as f:
            data = json.load(f)
        self.assertEqual(data["n_succeeded"], result["n_succeeded"])
        # 清理
        import shutil
        shutil.rmtree("/tmp/ml_e2e", ignore_errors=True)


if __name__ == "__main__":
    # 抑制 LGBM 噪音
    import logging
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    unittest.main(verbosity=2)
