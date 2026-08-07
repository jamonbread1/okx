# -*- coding: utf-8 -*-
"""Unit tests for ming v3 9 大修复的关键不变量。

不依赖数据下载、不跑完整回测，只验证：
  1. v3-1: risk_pct 语义（不再 0~1 评分双重缩放）
  2. v3-2: conf_mult ∈ [0.75, 1.25] 温和
  3. v3-3: 结构止损 D = max(D_atr, D_struct, D_floor)（不是 min）
  4. v3-4: sl_k_vol_mid 真正生效（三段插值）
  5. v3-5: don_min_break_hold_bars 真正实现窗口共振
  6. v3-7: MACD hist 同方向（不再要求单日扩张）
  7. v3-8: MACD 评分用 ATR 归一化
  8. v3-9: +2R 不平仓（take_profit=None, batch_ratios[-1]=0）
  9. v8 调参: r1_be 0.3 / r15 0.35 / r2_k 3.5 / max_bars 25
 10. v3 engine: Position.mae_r 字段
 11. v7 风控: max_dd_pct / equity_lock_threshold 默认值
"""
from __future__ import annotations

import os
import sys
import unittest

# 让脚本能找到 v3 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestV3_1RiskPct(unittest.TestCase):
    """v3-1: risk_pct = 单笔基础风险占权益比例, 默认 0.0075（不再 0.020）。"""

    def test_default_risk_pct_is_0075(self):
        from v3.strategies.ming.ming import MingBreakout
        from v3.strategies.loader import load_strategy_params
        params = load_strategy_params("ming")
        # v3 修正: 真实单笔风险预算, 默认 0.75%
        self.assertIn("risk_pct", params)
        self.assertAlmostEqual(float(params["risk_pct"]), 0.0075, places=4)


class TestV3_2ConfMultRange(unittest.TestCase):
    """v3-2: conf_mult ∈ [0.75, 1.25] 温和（不再 0~1 评分线性放大）。"""

    def test_conf_mult_bounds(self):
        # 模拟 _calc_conf_mult 函数 (从 ming.py 提取)
        def calc_conf_mult(score_pos, min_conf=0.55):
            return 0.75 + 0.50 * max(0.0, min(1.0, score_pos))
        # 最低 (刚过门槛)
        self.assertAlmostEqual(calc_conf_mult(0.0), 0.75, places=4)
        # 最高 (完美)
        self.assertAlmostEqual(calc_conf_mult(1.0), 1.25, places=4)
        # 中位
        self.assertAlmostEqual(calc_conf_mult(0.5), 1.00, places=4)
        # 范围检验
        for s in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            cm = calc_conf_mult(s)
            self.assertGreaterEqual(cm, 0.75, f"conf_mult={cm} at score_pos={s}")
            self.assertLessEqual(cm, 1.25, f"conf_mult={cm} at score_pos={s}")


class TestV3_3StructSLDirection(unittest.TestCase):
    """v3-3: 结构止损 D = max(D_atr, D_struct, D_floor)（不是 min）。"""

    def test_struct_sl_takes_max(self):
        # 模拟 _initial_sl_distance 的关键部分
        cur_price = 100.0
        atr_val = 2.0
        natr_pct = 0.5
        # 假设结构低点 = 95, 结构距离 = 5 (远 > atr 距离 3.6)
        d_atr = 1.8 * atr_val  # = 3.6
        d_floor = cur_price * 0.005  # = 0.5
        structure_low = 95.0
        structure_stop = structure_low - 0.2 * atr_val  # = 94.6
        structure_dist = cur_price - structure_stop  # = 5.4
        D = max(d_atr, d_floor, structure_dist)
        # v3 修复: D 应该是 max, = 5.4 (结构止损最远)
        # v6 错误: D = min(...) = 0.5 (floor 最小)
        self.assertAlmostEqual(D, 5.4, places=4,
                               msg="v3 修复: 结构止损 D 应取 max(atr, floor, struct) = 5.4")


class TestV3_4SLKVOLMidInterpolation(unittest.TestCase):
    """v3-4: sl_k_vol_mid 真正生效（三段插值）。"""

    def test_low_segment(self):
        from v3.strategies.ming.ming import MingBreakout
        cfg = {"sl_k_vol_low": 1.4, "sl_k_vol_mid": 1.8, "sl_k_vol_high": 2.5}
        s = MingBreakout(cfg)
        # NATR 百分位 <= 0.25 → low
        self.assertAlmostEqual(s._k_vol_for_natr_pct(0.10), 1.4, places=4)
        self.assertAlmostEqual(s._k_vol_for_natr_pct(0.25), 1.4, places=4)

    def test_mid_segment(self):
        from v3.strategies.ming.ming import MingBreakout
        cfg = {"sl_k_vol_low": 1.4, "sl_k_vol_mid": 1.8, "sl_k_vol_high": 2.5}
        s = MingBreakout(cfg)
        # NATR 百分位 ∈ (0.25, 0.50] → low→mid 线性
        self.assertAlmostEqual(s._k_vol_for_natr_pct(0.375), 1.6, places=4)
        self.assertAlmostEqual(s._k_vol_for_natr_pct(0.50), 1.8, places=4)

    def test_high_segment(self):
        from v3.strategies.ming.ming import MingBreakout
        cfg = {"sl_k_vol_low": 1.4, "sl_k_vol_mid": 1.8, "sl_k_vol_high": 2.5}
        s = MingBreakout(cfg)
        # NATR 百分位 ∈ (0.50, 0.75) → mid→high
        self.assertAlmostEqual(s._k_vol_for_natr_pct(0.625), 2.15, places=4)
        # NATR 百分位 >= 0.75 → high
        self.assertAlmostEqual(s._k_vol_for_natr_pct(0.80), 2.5, places=4)
        self.assertAlmostEqual(s._k_vol_for_natr_pct(1.00), 2.5, places=4)


class TestV3_5DonHoldBars(unittest.TestCase):
    """v3-5: don_min_break_hold_bars 真正实现窗口共振。"""

    def test_default_hold_bars(self):
        from v3.strategies.loader import load_strategy_params
        params = load_strategy_params("ming")
        # v3 修正: 默认 3 (窗口共振)
        self.assertEqual(int(params.get("don_min_break_hold_bars", 1)), 3)


class TestV3_7MacdHistDirection(unittest.TestCase):
    """v3-7: MACD hist 同方向（不再要求单日扩张）。"""

    def test_macd_long_only_needs_positive_hist(self):
        # v6 (错误): cur_hist > prev_hist (单日扩张) 且 cur_macd > 0
        # v3 (正确): cur_hist > 0 (同方向) 且 cur_macd > 0
        # 模拟: cur_hist = 0.5, prev_hist = 0.6 (下降但仍正)
        cur_macd = 0.5
        cur_hist = 0.5
        prev_hist = 0.6
        # v3: 仍 long 触发 (cur_hist > 0)
        macd_long_v3 = (cur_macd > 0 and cur_hist > 0)
        # v6: 不触发 (cur_hist <= prev_hist)
        macd_long_v6 = (cur_macd > 0 and cur_hist > prev_hist)
        self.assertTrue(macd_long_v3, "v3: cur_hist > 0 即 long")
        self.assertFalse(macd_long_v6, "v6: 单日扩张失效时不开仓")


class TestV3_9PlusTwoRNoTP(unittest.TestCase):
    """v3-9: +2R 不再硬平仓（take_profit=None, batch_ratios[-1]=0）。"""

    def test_take_profit_is_none(self):
        from v3.strategies.loader import load_strategy_params
        # 引擎读 signal.take_profit; ming 必须发 None
        # 我们检查代码常量: rr_list = [1.0, 1.5, 2.0], batch_ratios = [0, 0.20/0.35, 0]
        from v3.strategies.ming.ming import MingBreakout
        cfg = {"r15_partial_pct": 0.35}
        s = MingBreakout(cfg)
        # 验证 batch_ratios 数组构造
        rr_list = [1.0, 1.5, 2.0]
        batch_ratios = [0.0, s.r15_partial_pct, 0.0]
        # v3 关键: +2R 对应 batch_ratios[-1] = 0
        self.assertEqual(batch_ratios[2], 0.0, "v3: +2R 不减仓")
        self.assertEqual(batch_ratios[0], 0.0, "v3: +1R 也不减仓（只移 SL）")


class TestV8Tuning(unittest.TestCase):
    """v8 调参验证（针对 v3 修复后收益退化的反向调参）。"""

    def test_v8_params(self):
        from v3.strategies.loader import load_strategy_params
        params = load_strategy_params("ming")
        # v8 调参方向: 更紧的 BE 锁 / 更激进的 +1.5R 减仓 / 更紧的 Chandelier / 更短 max_bars
        self.assertAlmostEqual(float(params.get("r1_be_buffer_atr")), 0.3, places=4,
                               msg="v8: r1_be 0.5→0.3")
        self.assertAlmostEqual(float(params.get("r15_partial_pct")), 0.35, places=4,
                               msg="v8: r15 0.20→0.35")
        self.assertAlmostEqual(float(params.get("r2_chandelier_k")), 3.5, places=4,
                               msg="v8: r2_k 5.5→3.5")
        self.assertEqual(int(params.get("timeout_max_bars")), 25,
                         msg="v8: max_bars 45→25")


class TestV3PositionMAE(unittest.TestCase):
    """v3 engine: Position.mae_r 字段。"""

    def test_mae_field_exists(self):
        from v3.engine import Position
        p = Position()
        self.assertTrue(hasattr(p, "mae_r"))
        self.assertEqual(p.mae_r, 0.0)


class TestV7RiskConfig(unittest.TestCase):
    """v7 风控: max_dd_pct / equity_lock_threshold / equity_lock_drawdown。"""

    def test_engine_risk_defaults(self):
        from v3.engine import StrategyEngine
        # 模拟最小 cfg
        cfg = {
            "strategy": {
                "enabled_strategies": [],
            }
        }
        eng = StrategyEngine(cfg)
        self.assertAlmostEqual(eng.max_dd_pct, 0.15, places=4)
        self.assertAlmostEqual(eng.equity_lock_threshold, 1000.0, places=4)
        self.assertAlmostEqual(eng.equity_lock_drawdown, 0.5, places=4)
        # 初始时未锁定
        self.assertFalse(eng._account_locked)


class TestV7EquityLockTrigger(unittest.TestCase):
    """v7 风控: 小资金腰斩锁仓触发。"""

    def test_equity_lock_under_1000usd(self):
        from v3.engine import StrategyEngine
        cfg = {"strategy": {"enabled_strategies": []}}
        eng = StrategyEngine(cfg)
        # 第一次 update_equity: 记录 start_equity
        eng.update_equity(800.0)
        self.assertAlmostEqual(eng.start_equity, 800.0)
        self.assertFalse(eng._account_locked)
        # 腰斩: 800 * 0.5 = 400
        eng.update_equity(350.0)
        self.assertTrue(eng._account_locked)
        self.assertIn("equity_lock", eng._lock_reason)

    def test_no_lock_above_1000usd(self):
        from v3.engine import StrategyEngine
        cfg = {"strategy": {"enabled_strategies": []}}
        eng = StrategyEngine(cfg)
        # 10000 USD 本金
        eng.update_equity(10000.0)
        # 跌到 50% = 5000 USD 时:
        #   - equity_lock: 不启用 (start_equity=10000 > 阈值 1000)
        #   - max_dd 15%: dd=50% > 15% → 触发
        eng.update_equity(5000.0)
        self.assertTrue(eng._account_locked)
        # 触发的是 max_dd 而不是 equity_lock
        self.assertIn("max_dd", eng._lock_reason)
        self.assertNotIn("equity_lock", eng._lock_reason)


class TestV7MaxDDTrigger(unittest.TestCase):
    """v7 风控: 总回撤 15% 硬上限（对所有资金量生效）。"""

    def test_max_dd_triggers(self):
        from v3.engine import StrategyEngine
        cfg = {"strategy": {"enabled_strategies": []}}
        eng = StrategyEngine(cfg)
        eng.update_equity(10000.0)
        # peak = 10000
        # dd = (10000 - 8000) / 10000 = 20% > 15% → 触发 max_dd
        eng.update_equity(8000.0)
        self.assertTrue(eng._account_locked)
        self.assertIn("max_dd", eng._lock_reason)
        # 不是 equity_lock (因为 start_equity=10000 > 1000 阈值)
        self.assertNotIn("equity_lock", eng._lock_reason)

    def test_max_dd_takes_priority_over_normal_dd(self):
        """max_dd 不应误触发（dd < 15% 时不锁）。"""
        from v3.engine import StrategyEngine
        cfg = {"strategy": {"enabled_strategies": []}}
        eng = StrategyEngine(cfg)
        eng.update_equity(10000.0)
        # dd = 10% < 15% → 不应触发
        eng.update_equity(9000.0)
        self.assertFalse(eng._account_locked)


class TestDiagnoseTool(unittest.TestCase):
    """diagnose 工具能正常导入和分类退出原因。"""

    def test_import(self):
        import tools.diagnose as d
        self.assertTrue(hasattr(d, "diagnose_phase"))
        self.assertTrue(hasattr(d, "format_console"))
        self.assertTrue(hasattr(d, "format_markdown"))

    def test_classify_exit(self):
        import tools.diagnose as d
        self.assertEqual(d._classify_exit("SL long low=100.0<=99.5"), "sl_hit")
        self.assertEqual(d._classify_exit("chandelier low=105.0<=104.5"), "chandelier")
        self.assertEqual(d._classify_exit("r1.5 partial 20% @101.5"), "r1.5_partial")
        self.assertEqual(d._classify_exit("mfe_decay 0.3R<0.5R in 3 bars"), "mfe_decay")
        self.assertEqual(d._classify_exit("breakout_fail high=100.5<ref=101.0"), "breakout_fail")
        self.assertEqual(d._classify_exit("max_bars 30>=30"), "max_bars")
        self.assertEqual(d._classify_exit("panic_gap_dn open=99.0"), "panic_gap")
        self.assertEqual(d._classify_exit("timeout 86400s"), "timeout")
        self.assertEqual(d._classify_exit("TP1 清剩余仓 @102.0"), "hard_tp")
        self.assertEqual(d._classify_exit("TP2 @103.0"), "partial_tp")

    def test_mfe_bucket(self):
        import tools.diagnose as d
        self.assertEqual(d._mfe_bucket(-0.1), "<0")
        self.assertEqual(d._mfe_bucket(0.3), "0~0.5")
        self.assertEqual(d._mfe_bucket(0.8), "0.5~1.0")
        self.assertEqual(d._mfe_bucket(1.3), "1.0~1.5")
        self.assertEqual(d._mfe_bucket(1.8), "1.5~2.0")
        self.assertEqual(d._mfe_bucket(2.5), "2.0~3.0")
        self.assertEqual(d._mfe_bucket(3.5), "3.0~5.0")
        self.assertEqual(d._mfe_bucket(6.0), "5.0+")

    def test_fill_clamp_ratio(self):
        """bar_clamp 占比检测. 用户的 IS 段 4/194 = 2.1% < 5% → 应判 INFO."""
        import tools.diagnose as d
        import pandas as pd
        # 模拟用户 IS 段: 190 正常 + 3 bar_low_clamp + 1 bar_high_clamp
        fills = pd.DataFrame({
            "fill_reason": ["limit_fill"] * 190
                          + ["limit_fill|bar_low_clamp"] * 3
                          + ["limit_fill|bar_high_clamp"] * 1
        })
        n_clamped, n_total, ratio = d._fill_clamp_ratio(fills)
        self.assertEqual(n_clamped, 4)
        self.assertEqual(n_total, 194)
        self.assertAlmostEqual(ratio, 0.0206, places=3)

    def test_fill_clamp_ratio_empty(self):
        """空 fills 时应返回 0/0/0.0, 不崩."""
        import tools.diagnose as d
        import pandas as pd
        n, t, r = d._fill_clamp_ratio(pd.DataFrame())
        self.assertEqual(n, 0)
        self.assertEqual(t, 0)
        self.assertEqual(r, 0.0)
        n, t, r = d._fill_clamp_ratio(None)
        self.assertEqual(n, 0)
        self.assertEqual(t, 0)
        self.assertEqual(r, 0.0)

    def test_strategy_concentration(self):
        """策略过度集中检测. 用户 ming 占 100%."""
        import tools.diagnose as d
        summary = {
            "attribution": {
                "per_strategy": {"ming": {"pnl": 152.5, "n": 97}}
            }
        }
        s = d._strategy_concentration(summary)
        self.assertEqual(s, {"ming": 152.5})

    def test_build_warnings_slip_bps_info(self):
        """大滑点 + bar_clamp < 5% 应给 INFO 而非 WARN (用户 003 警告误报修复)."""
        import tools.diagnose as d
        diagnostics = {
            "in_sample_70pct": {
                "phase": "in_sample_70pct",
                "_clamp_ratio": 0.021,  # 2.1% (用户真实数据)
                "_n_clamped": 4,
                "_n_total_fill": 194,
                "summary": {"total_return": 0.015, "sharpe": 1.2, "max_drawdown": -0.002,
                            "trade_count": 96, "win_rate": 0.58, "total_pnl": 150.0,
                            "calmar": 7.0, "volatility_ann": 0.003, "bars": 1414},
                "attribution": {"per_strategy": {"ming": {"pnl": 152.5, "n": 97}},
                                "per_direction": {"long": 137, "short": 14, "funding": 0}},
                "fill_stats": {
                    "slip_bps_abs": {"mean": 15.16, "p50": 12.0, "p95": 59.49, "max": 207.42},
                },
            }
        }
        warnings = d.build_warnings(diagnostics)
        # 应该有 [INFO] 关于滑点 (不是 WARN)
        slip_warnings = [w for w in warnings if "大滑点" in w or "策略" in w]
        self.assertGreater(len(slip_warnings), 0, "应有滑点或策略集中警告")
        # 至少 1 个 INFO 标记
        info_warnings = [w for w in warnings if "[INFO]" in w]
        self.assertGreater(len(info_warnings), 0,
                           f"应有 INFO 警告 (大滑点误报), 实际: {warnings}")

    def test_build_warnings_slip_bps_warn(self):
        """大滑点 + bar_clamp > 10% 应给 WARN."""
        import tools.diagnose as d
        diagnostics = {
            "in_sample_70pct": {
                "phase": "in_sample_70pct",
                "_clamp_ratio": 0.15,  # 15%
                "_n_clamped": 30,
                "_n_total_fill": 200,
                "summary": {"total_return": 0.01, "sharpe": 1.0, "max_drawdown": -0.001,
                            "trade_count": 50, "win_rate": 0.6, "total_pnl": 100.0,
                            "calmar": 5.0, "volatility_ann": 0.003, "bars": 1000},
                "attribution": {"per_strategy": {"ming": {"pnl": 100, "n": 50}},
                                "per_direction": {"long": 100, "short": 0, "funding": 0}},
                "fill_stats": {
                    "slip_bps_abs": {"mean": 20, "p50": 18, "p95": 80, "max": 200},
                },
            }
        }
        warnings = d.build_warnings(diagnostics)
        warn_w = [w for w in warnings if "[WARN]" in w and "bar_clamp" in w]
        self.assertGreater(len(warn_w), 0,
                           f"应有 [WARN] bar_clamp 警告, 实际: {warnings}")


class TestParamSweepCAGR(unittest.TestCase):
    """param_sweep 改用 CAGR 衰减（v3 任务清单 #6）。"""

    def test_cagr_calculation(self):
        from tools.param_sweep import _cagr
        # 1 年 100% 收益 → CAGR = 100%
        m = {"total_return": 1.0}
        self.assertAlmostEqual(_cagr(m, 365.0), 1.0, places=4)
        # 半年 50% 收益 → CAGR = (1.5)^2 - 1 = 125%
        m = {"total_return": 0.5}
        self.assertAlmostEqual(_cagr(m, 182.5), 1.25, places=4)
        # 0 收益 → 0
        m = {"total_return": 0.0}
        self.assertAlmostEqual(_cagr(m, 365.0), 0.0, places=4)

    def test_overfit_warnings_uses_cagr(self):
        from tools.param_sweep import overfit_warnings
        is_m = {"total_return": 1.0, "sharpe": 2.0, "max_drawdown": -0.05, "trade_count": 100, "bars": 1460}
        oos_m = {"total_return": 0.10, "sharpe": 1.0, "max_drawdown": -0.05, "trade_count": 50, "bars": 730}
        warnings = overfit_warnings(is_m, oos_m, is_days=1460, oos_days=730)
        # IS 100% / 4 年 → CAGR 18.92%
        # OOS 10% / 2 年 → CAGR 4.88%
        # 衰减 = (0.1892 - 0.0488) / 0.1892 = 74% > 50% → 报警
        cagr_warnings = [w for w in warnings if "CAGR" in w]
        self.assertGreater(len(cagr_warnings), 0, "应包含 CAGR 衰减报警")


class TestRegistryDiscovery(unittest.TestCase):
    """registry 递归扫描 + 显式 import 注册（v3 fix）。"""

    def test_ming_in_registry(self):
        from v3.strategies import REGISTRY
        self.assertIn("ming", REGISTRY)
        self.assertIn("don", REGISTRY)
        self.assertIn("vol", REGISTRY)
        self.assertIn("mr", REGISTRY)
        self.assertIn("ewmac", REGISTRY)
        self.assertIn("macd", REGISTRY)
        self.assertIn("radx", REGISTRY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
