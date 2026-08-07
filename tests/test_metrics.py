# -*- coding: utf-8 -*-
"""Unit tests for backtest/metrics.py — CAGR 衰减 + overfit 检查。"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestCAGR(unittest.TestCase):
    """_cagr 计算正确性."""

    def test_one_year(self):
        from backtest.metrics import _cagr
        # 1 年 100% → CAGR = 100%
        self.assertAlmostEqual(_cagr(1.0, 365.0), 1.0, places=4)

    def test_half_year_doubles(self):
        from backtest.metrics import _cagr
        # 半年 50% → CAGR = (1.5)^2 - 1 = 125%
        self.assertAlmostEqual(_cagr(0.5, 182.5), 1.25, places=4)

    def test_zero_return(self):
        from backtest.metrics import _cagr
        self.assertAlmostEqual(_cagr(0.0, 365.0), 0.0, places=4)

    def test_negative_caps_at_minus_one(self):
        from backtest.metrics import _cagr
        # 全部亏光 = -100% → CAGR 不应崩
        self.assertEqual(_cagr(-1.0, 365.0), -1.0)

    def test_days_zero_fallback(self):
        from backtest.metrics import _cagr
        # 0 天时回退到 0
        self.assertEqual(_cagr(0.5, 0.0), 0.0)


class TestCheckOverfitting(unittest.TestCase):
    """check_overfitting 用 CAGR 衰减（不是 raw total_return 衰减）."""

    def test_stable_passes(self):
        from backtest.metrics import check_overfitting
        is_m = {"total_return": 0.10, "sharpe": 1.5, "max_drawdown": -0.05,
                "trade_count": 100, "win_rate": 0.6}
        oos_m = {"total_return": 0.04, "sharpe": 1.4, "max_drawdown": -0.04,
                 "trade_count": 30, "win_rate": 0.58}
        r = check_overfitting(is_m, oos_m, is_days=365 * 4, oos_days=365 * 2)
        self.assertTrue(r["ok"], f"应该 pass, 但报错: {r['warnings']}")
        self.assertLess(r["overfit_score"], 0.45)

    def test_cagr_decay_warning(self):
        """IS CAGR 10% / OOS CAGR 1% (差距大) 应报警."""
        from backtest.metrics import check_overfitting
        is_m = {"total_return": 0.467, "sharpe": 1.8, "max_drawdown": -0.05,
                "trade_count": 200, "win_rate": 0.6}
        # 4 年 46.7% → CAGR ≈ 10%
        # 2 年 1.0% → CAGR ≈ 0.5%
        # CAGR decay = 1 - 0.005/0.10 = 95% > 55% 阈值
        oos_m = {"total_return": 0.01, "sharpe": 1.6, "max_drawdown": -0.04,
                 "trade_count": 30, "win_rate": 0.58}
        r = check_overfitting(is_m, oos_m, is_days=365 * 4, oos_days=365 * 2)
        cagr_warnings = [w for w in r["warnings"] if "CAGR" in w]
        self.assertGreater(len(cagr_warnings), 0, f"应有 CAGR 报警: {r['warnings']}")
        self.assertFalse(r["ok"])

    def test_oos_negative_cagr_warning(self):
        """OOS CAGR 为负应报警."""
        from backtest.metrics import check_overfitting
        is_m = {"total_return": 0.10, "sharpe": 1.5, "max_drawdown": -0.05,
                "trade_count": 100, "win_rate": 0.6}
        oos_m = {"total_return": -0.05, "sharpe": 0.8, "max_drawdown": -0.08,
                 "trade_count": 30, "win_rate": 0.45}
        r = check_overfitting(is_m, oos_m, is_days=365 * 4, oos_days=365 * 2)
        neg_warnings = [w for w in r["warnings"] if "negative" in w]
        self.assertGreater(len(neg_warnings), 0, f"应有 OOS 负 CAGR 报警: {r['warnings']}")

    def test_low_trade_count_warning(self):
        """OOS 交易数 < 15 应报警."""
        from backtest.metrics import check_overfitting
        is_m = {"total_return": 0.10, "sharpe": 1.5, "max_drawdown": -0.05,
                "trade_count": 100, "win_rate": 0.6}
        oos_m = {"total_return": 0.04, "sharpe": 1.4, "max_drawdown": -0.04,
                 "trade_count": 5, "win_rate": 0.6}
        r = check_overfitting(is_m, oos_m, is_days=365 * 4, oos_days=365 * 2)
        low_trade_warnings = [w for w in r["warnings"] if "trade count" in w]
        self.assertGreater(len(low_trade_warnings), 0)

    def test_sharpe_decay_warning(self):
        """Sharpe 衰减 > 45% 应报警."""
        from backtest.metrics import check_overfitting
        is_m = {"total_return": 0.10, "sharpe": 2.0, "max_drawdown": -0.05,
                "trade_count": 100, "win_rate": 0.6}
        # OOS sharpe 0.5 → decay = 1 - 0.5/2.0 = 75%
        oos_m = {"total_return": 0.04, "sharpe": 0.5, "max_drawdown": -0.04,
                 "trade_count": 30, "win_rate": 0.55}
        r = check_overfitting(is_m, oos_m, is_days=365 * 4, oos_days=365 * 2)
        sharpe_warnings = [w for w in r["warnings"] if "Sharpe decay" in w]
        self.assertGreater(len(sharpe_warnings), 0, f"应有 Sharpe 报警: {r['warnings']}")


class TestComputeMetrics(unittest.TestCase):
    """compute_metrics 基础计算."""

    def test_short_equity_returns_error(self):
        from backtest.metrics import compute_metrics
        import pandas as pd
        # < 2 根应返回 error
        eq = pd.Series([100.0])
        r = compute_metrics(eq, trades=None, bars_per_year=365)
        self.assertIn("error", r)

    def test_basic_metrics(self):
        from backtest.metrics import compute_metrics
        import pandas as pd
        # 100 → 110, 5 根 bar
        eq = pd.Series([100, 102, 105, 108, 110], dtype=float)
        r = compute_metrics(eq, trades=None, bars_per_year=365)
        self.assertAlmostEqual(r["total_return"], 0.10, places=4)
        self.assertAlmostEqual(r["start_equity"], 100.0)
        self.assertAlmostEqual(r["end_equity"], 110.0)
        self.assertEqual(r["bars"], 5)

    def test_metrics_to_frame(self):
        from backtest.metrics import metrics_to_frame
        is_m = {"a": 1, "b": 2}
        oos_m = {"a": 3, "c": 4}
        df = metrics_to_frame(is_m, oos_m)
        self.assertEqual(len(df), 3)  # a, b, c
        self.assertIn("metric", df.columns)
        self.assertIn("in_sample_70pct", df.columns)
        self.assertIn("out_of_sample_30pct", df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
