# -*- coding: utf-8 -*-
"""策略基类与 Signal 协议。

插件约定（详见 registry.py，完整手册见项目根 README.md）:
  1. 继承 StrategyBase，设置唯一 name
  2. 实现 generate(...) -> Optional[Signal]
  3. 文件放入 v3/strategies/（非 _ 前缀）即可被自动发现
  4. 在 config.yaml 的 enabled_strategies 中启用

设计原则（参考 pysystemtrade / freqtrade）:
  - 策略只负责生成信号，不负责执行
  - 信号包含方向、置信度、止损/止盈、策略标识
  - 参数从 cfg 读取，建议 {name}_xxx 前缀
  - 仓位用 _calc_size；资金费过滤用 _funding_blocked
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class Signal:
    """交易信号 — 策略输出，引擎输入。

    action:   open_long / open_short / close / partial_close / hold
    direction: long / short / flat
    confidence: 0.0~1.0
    strategy:  策略名 (vol / mr / ewmac / macd / don / ming / radx)
    regime:    信号生成时的 regime (trend / chop / mixed)
    stop_loss: 止损价
    take_profit: 止盈价（None = 不设，避免 +2R 误触发全平）
    tp_batches: 分批止盈价格列表
    batch_ratios: 每批平仓比例
    size:      建议张数
    reason:    人类可读的信号描述
    """
    action: str = "hold"
    direction: str = "flat"
    confidence: float = 0.0
    strategy: str = ""
    regime: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    tp_batches: List[float] = field(default_factory=list)
    batch_ratios: List[float] = field(default_factory=list)
    size: float = 0.0
    atr: float = 0.0
    reason: str = ""
    close_ratio: float = 1.0  # partial_close 平仓比例（1.0=全平）
    fill_price: float = 0.0   # 引擎/回测层建议的最差成交价；0 = 由回测用默认价
    rr_list: List[float] = field(default_factory=list)  # 分批止盈的 RR 档位
    # 入场参考价（time-stop 用：持仓 N 日未破入场 K 线高点 → 退出）
    entry_reference_high: float = 0.0
    entry_reference_low: float = 0.0
    # time-stop 配置
    mfe_window_bars: int = 0
    mfe_min_r: float = 0.0
    breakout_window_bars: int = 0
    max_bars: int = 0
    # panic exit 配置
    panic_gap_atr: float = 0.0
    # +1R / +1.5R / +2R 引擎管理配置（策略显式传时优先于 Position 默认值）
    r1_be_buffer_atr: float = 0.0
    r15_partial_pct: float = 0.0
    r2_chandelier_N: int = 0
    r2_chandelier_k: float = 0.0
    # 引擎填入
    leverage: float = 5.0
    tick_sz: float = 0.1
    lot_sz: float = 0.01
    min_sz: float = 0.01
    ct_val: float = 0.01

    @property
    def is_open(self) -> bool:
        return self.action in ("open_long", "open_short")

    @property
    def is_close(self) -> bool:
        return self.action in ("close", "partial_close")


class StrategyBase:
    """策略基类。

    子类必须实现:
      - name: str — 策略标识
      - required_regime: str — 允许开仓的 regime (trend / chop / mixed / any)
      - generate(df, last, capital, ...) -> Optional[Signal]
    """
    name: str = "base"
    required_regime: str = "any"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(
        self,
        df: pd.DataFrame,
        regime: str,
        last: float,
        capital: float,
        leverage: float,
        specs: dict,
        kelly_factor: float = 1.0,
        funding_rate: float = 0.0,
    ) -> Optional[Signal]:
        """生成交易信号。

        df: 当前品种的 K 线数据 (含 OHLCV)
        regime: 当前市场状态
        last: 最新价格
        capital: 可用资金
        leverage: 杠杆
        specs: 合约规格
        kelly_factor: Kelly 仓位缩放
        funding_rate: 当前资金费率
        """
        raise NotImplementedError

    def regime_ok(self, regime: str) -> bool:
        """检查当前 regime 是否允许开仓。"""
        if self.required_regime == "any":
            return True
        if self.required_regime == "trend":
            return regime == "trend"
        if self.required_regime == "chop":
            return regime in ("chop", "mixed")
        if self.required_regime == "mixed":
            return regime in ("trend", "mixed")
        return True

    def _calc_size(
        self,
        capital: float,
        leverage: float,
        pct: float,
        last: float,
        specs: dict,
        kelly_factor: float = 1.0,
        max_loss_pct: float = 0.01,
        sl_distance_pct: float = 0.01,
        atr_pct: float = 0.0,
    ) -> float:
        """计算建仓张数（kelly 或 voltarget，由配置决定）。"""
        from v3.utils.sizing import calc_size
        sizing_mode = str(self.cfg.get("sizing_mode", "kelly"))
        target_daily_vol = float(self.cfg.get("target_daily_vol", 0.015))
        return calc_size(capital, leverage, pct, last, specs, kelly_factor,
                         max_loss_pct, sl_distance_pct,
                         sizing_mode=sizing_mode, target_daily_vol=target_daily_vol,
                         atr_pct=atr_pct)

    def _funding_blocked(self, direction: str, funding_rate: float) -> bool:
        """资金费率方向过滤：向付费方向拒绝新仓。阈值配置化（funding_max_abs）。"""
        threshold = float(self.cfg.get("funding_max_abs", 0.0005))
        if direction == "long" and funding_rate > threshold:
            return True
        if direction == "short" and funding_rate < -threshold:
            return True
        return False

    def _expected_r(
        self,
        sl_pct: float,
        tp_pct: float,
        win_rate: float,
        fee_rt: float,
        slippage: float,
    ) -> float:
        """净期望 R 数。

        exp_R = tp_R * wr - (1 - wr) - cost_R
          tp_R = tp_pct / sl_pct（止盈以 R 计）
          cost_R = (fee_rt + 2*slippage) / sl_pct（成本以 R 计）
        """
        if sl_pct <= 0:
            return -1.0
        tp_r = tp_pct / sl_pct
        cost_r = (fee_rt + 2.0 * slippage) / sl_pct
        return tp_r * win_rate - (1.0 - win_rate) - cost_r
