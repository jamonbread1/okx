# -*- coding: utf-8 -*-
"""Unit tests for param_sweep.py — 通用参数扫描器."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestParseParamSpec(unittest.TestCase):
    """--params NAME:KEY=VAL1,VAL2,VAL3 解析."""

    def test_single_value(self):
        from tools.param_sweep import parse_param_spec
        name, key, values = parse_param_spec("ming:risk_pct=0.0075")
        self.assertEqual(name, "ming")
        self.assertEqual(key, "risk_pct")
        self.assertEqual(values, [0.0075])

    def test_multiple_values_mixed(self):
        from tools.param_sweep import parse_param_spec
        name, key, values = parse_param_spec("ming:max_bars=20,30,45,60")
        self.assertEqual(name, "ming")
        self.assertEqual(key, "max_bars")
        # 全部转 int (因为是纯整数)
        self.assertEqual(values, [20, 30, 45, 60])

    def test_float_values(self):
        from tools.param_sweep import parse_param_spec
        name, key, values = parse_param_spec("don:don_period=15,20,25.5")
        self.assertEqual(name, "don")
        self.assertEqual(key, "don_period")
        # 15, 20 是 int; 25.5 是 float
        self.assertEqual(values, [15, 20, 25.5])

    def test_invalid_format_raises(self):
        from tools.param_sweep import parse_param_spec
        with self.assertRaises(ValueError):
            parse_param_spec("invalid_format")
        with self.assertRaises(ValueError):
            parse_param_spec(":key=1")
        with self.assertRaises(ValueError):
            parse_param_spec("name:=1")

    def test_strat_with_underscore(self):
        from tools.param_sweep import parse_param_spec
        name, key, values = parse_param_spec("risk_mgmt:max_dd=0.10,0.15")
        self.assertEqual(name, "risk_mgmt")
        self.assertEqual(key, "max_dd")
        self.assertEqual(values, [0.10, 0.15])


class TestScoreResult(unittest.TestCase):
    """综合评分 = oos_sharpe × (1 - overfit) × dd_penalty."""

    def _make(self, **kwargs):
        from tools.param_sweep import SweepResult
        defaults = dict(
            params={}, is_metrics={}, oos_metrics={},
            is_return=0.0, oos_return=0.0,
            is_sharpe=1.0, oos_sharpe=1.0,
            is_max_dd=-0.05, oos_max_dd=-0.05,
            is_trade_count=10, oos_trade_count=10,
            overfit_score=0.0, warnings=[], elapsed=1.0, error="",
        )
        defaults.update(kwargs)
        return SweepResult(**defaults)

    def test_basic_score(self):
        from tools.param_sweep import score_result
        r = self._make(oos_sharpe=2.0, oos_max_dd=-0.10, overfit_score=0.0)
        # 2.0 * (1 - 0) * 1.0 = 2.0
        self.assertAlmostEqual(score_result(r, 0.15), 2.0, places=4)

    def test_dd_penalty(self):
        from tools.param_sweep import score_result
        r = self._make(oos_sharpe=2.0, oos_max_dd=-0.30, overfit_score=0.0)
        # 2.0 * 1.0 * (0.15 / 0.30) = 1.0
        self.assertAlmostEqual(score_result(r, 0.15), 1.0, places=4)

    def test_overfit_penalty(self):
        from tools.param_sweep import score_result
        r = self._make(oos_sharpe=2.0, oos_max_dd=-0.05, overfit_score=0.5)
        # 2.0 * 0.5 * 1.0 = 1.0
        self.assertAlmostEqual(score_result(r, 0.15), 1.0, places=4)

    def test_error_returns_neg_inf(self):
        from tools.param_sweep import score_result
        r = self._make(error="some error")
        self.assertEqual(score_result(r, 0.15), -1e9)

    def test_zero_sharpe_returns_neg_inf(self):
        from tools.param_sweep import score_result
        r = self._make(oos_sharpe=0.0)
        self.assertEqual(score_result(r, 0.15), -1e9)


class TestFormatParamsCell(unittest.TestCase):
    """_format_params_cell 把 params dict 格式化成表格单元格."""

    def test_short_dict(self):
        from tools.param_sweep import _format_params_cell
        result = _format_params_cell({"a": 1, "b": 2})
        self.assertIn("a=1", result)
        self.assertIn("b=2", result)

    def test_long_dict_truncates(self):
        from tools.param_sweep import _format_params_cell
        d = {f"param_{i}": i for i in range(10)}
        result = _format_params_cell(d, max_show=3)
        self.assertIn("(+7)", result)
        self.assertIn("param_0=0", result)
        self.assertIn("param_2=2", result)
        self.assertNotIn("param_3=3", result)


class TestOverfitWarnings(unittest.TestCase):
    """overfit_warnings 边界条件."""

    def test_zero_sharpe_no_warning(self):
        from tools.param_sweep import overfit_warnings
        is_m = {"sharpe": 0, "total_return": 0.01, "max_drawdown": -0.05, "trade_count": 50}
        oos_m = {"sharpe": 0, "total_return": 0.001, "max_drawdown": -0.04, "trade_count": 10}
        # is_sharpe > 0 不满足 → 不报警 (虽然其它条件可能报警)
        w = overfit_warnings(is_m, oos_m, is_days=365*4, oos_days=365*2)
        # 应该没有 sharpe 衰减报警
        sharpe_w = [x for x in w if "sharpe" in x.lower()]
        self.assertEqual(len(sharpe_w), 0)

    def test_negative_cagr_warning(self):
        from tools.param_sweep import overfit_warnings
        is_m = {"sharpe": 1.5, "total_return": 0.10, "max_drawdown": -0.05, "trade_count": 50}
        # 4 年 5% → CAGR 1.2%; 2 年 -10% → CAGR -5%
        oos_m = {"sharpe": 0.8, "total_return": -0.10, "max_drawdown": -0.15, "trade_count": 10}
        w = overfit_warnings(is_m, oos_m, is_days=365*4, oos_days=365*2)
        # 应该报警: OOS CAGR 负值
        neg_w = [x for x in w if "为负" in x]
        self.assertGreater(len(neg_w), 0)

    def test_low_trade_count_warning(self):
        from tools.param_sweep import overfit_warnings
        is_m = {"sharpe": 1.5, "total_return": 0.10, "max_drawdown": -0.05, "trade_count": 50}
        oos_m = {"sharpe": 1.2, "total_return": 0.04, "max_drawdown": -0.04, "trade_count": 5}
        w = overfit_warnings(is_m, oos_m, is_days=365*4, oos_days=365*2)
        low_w = [x for x in w if "交易数" in x]
        self.assertGreater(len(low_w), 0)


class TestLocalOverfitCheck(unittest.TestCase):
    """local_overfit_check — 相邻参数 sharpe 波动 > 30% 报警."""

    def test_smooth_curve_no_warning(self):
        from tools.param_sweep import local_overfit_check
        results = [{"oos_sharpe": x} for x in [1.0, 1.1, 1.05, 1.08, 1.02]]
        # median ≈ 1.05, max diff ≈ 0.1/1.05 ≈ 9.5% < 30%
        w = local_overfit_check(results)
        self.assertEqual(len(w), 0)

    def test_sharp_peak_warning(self):
        from tools.param_sweep import local_overfit_check
        # median = 1.0, max = 1.8, 差异 80% > 30%
        results = [{"oos_sharpe": x} for x in [1.0, 1.8, 0.9, 1.0, 1.1]]
        w = local_overfit_check(results)
        self.assertGreater(len(w), 0)
        # 验证: 报警里包含 "过陡" 关键词
        self.assertTrue(any("过陡" in x for x in w))

    def test_too_few_results(self):
        from tools.param_sweep import local_overfit_check
        # < 3 个 → 不报警
        results = [{"oos_sharpe": 1.0}, {"oos_sharpe": 100.0}]
        w = local_overfit_check(results)
        self.assertEqual(len(w), 0)


class TestFormatReport(unittest.TestCase):
    """format_report — 报告输出."""

    def _make(self, **kwargs):
        from tools.param_sweep import SweepResult
        defaults = dict(
            params={}, is_metrics={}, oos_metrics={},
            is_return=0.01, oos_return=0.005,
            is_sharpe=1.0, oos_sharpe=1.2,
            is_max_dd=-0.05, oos_max_dd=-0.04,
            is_trade_count=50, oos_trade_count=20,
            overfit_score=0.1, warnings=[], elapsed=2.0, error="",
        )
        defaults.update(kwargs)
        return SweepResult(**defaults)

    def test_report_includes_top_5(self):
        from tools.param_sweep import format_report
        results = [
            self._make(params={"risk_pct": 0.005}, oos_sharpe=2.0),
            self._make(params={"risk_pct": 0.01}, oos_sharpe=1.5),
            self._make(params={"risk_pct": 0.02}, oos_sharpe=1.0),
        ]
        report = format_report(results, "ming", "1D", [], ["risk_pct"])
        self.assertIn("Top 5", report)
        self.assertIn("推荐参数", report)
        self.assertIn("risk_pct=0.005", report)

    def test_report_handles_no_results(self):
        from tools.param_sweep import format_report
        report = format_report([], "ming", "1D", [], ["risk_pct"])
        self.assertIn("无有效结果", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
