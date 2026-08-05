"""
RNG v4 — BTC/ETH 永续合约区间假突破回归策略

核心逻辑
--------
本策略不在“接近区间边缘”时直接进行均值回归，而是要求：

1. 先形成一个具有多次边界测试、低趋势效率的有效区间；
2. setup bar（前一根 K）刺破区间边界；
3. setup bar 收盘重新回到区间内，并表现出拒绝；
4. confirmation bar（当前 K）继续向区间内部确认；
5. 以 confirmation bar 收盘附近作为预期入场价；
6. 止损放在假突破极值外侧；
7. 止盈放在区间中部附近；
8. 由引擎执行 SL、TP 和 timeout。

重要：
- 区间计算严格排除 setup bar 和 confirmation bar；
- 信号使用两根已完成 K 线，避免同一根 K 线触边后立即假定确认；
- 本策略只实现 failed-breakout mean reversion，不再混合 breakout；
- 建议 BTC/ETH 分别配置、分别验证，不要共享一组“最优参数”。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from v2.strategies.base import Signal, StrategyBase


class RangePosition(StrategyBase):
    """
    RNG v4：区间假突破后的确认型均值回归。

    时间定义：
        base range:
            df[-window-2 : -2]

        setup bar:
            df[-2]

        confirmation bar:
            df[-1]

    因此区间边界不会使用 setup/confirmation 两根 K 线的数据。
    """

    name = "rng"
    required_regime = "any"
    skip_htf = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        # ============================================================
        # 1. 基础开关
        # ============================================================
        self.allow_long = bool(cfg.get("range_allow_long", True))
        self.allow_short = bool(cfg.get("range_allow_short", True))

        if bool(cfg.get("range_long_only", False)):
            self.allow_short = False

        if bool(cfg.get("range_short_only", False)):
            self.allow_long = False

        if "range_skip_htf" in cfg:
            self.skip_htf = bool(cfg.get("range_skip_htf"))

        # 默认不依赖外部 regime 标签。
        # 原因：外部 regime 可能存在滞后或不同时间框架对齐问题。
        self.use_external_regime = bool(
            cfg.get("range_use_external_regime", False)
        )

        allowed_regimes = cfg.get(
            "range_allowed_regimes",
            ["chop", "mixed"],
        )

        if isinstance(allowed_regimes, str):
            allowed_regimes = [
                x.strip()
                for x in allowed_regimes.split(",")
                if x.strip()
            ]

        self.allowed_regimes = {
            str(x).strip().lower()
            for x in allowed_regimes
        }

        # ============================================================
        # 2. 区间参数
        # ============================================================
        self.window = max(
            10,
            int(cfg.get("range_window", 30)),
        )

        # 区间宽度占价格比例。
        self.min_range_pct = float(
            cfg.get("range_min_range_pct", 0.025)
        )
        self.max_range_pct = float(
            cfg.get("range_max_range_pct", 0.25)
        )

        # 区间宽度相对于 ATR 的约束。
        self.min_range_atr = float(
            cfg.get("range_min_range_atr", 2.5)
        )
        self.max_range_atr = float(
            cfg.get("range_max_range_atr", 10.0)
        )

        # 边界触碰统计。
        # touch_tolerance_atr 表示距离边界多少 ATR 内算一次触碰。
        self.touch_tolerance_atr = float(
            cfg.get("range_touch_tolerance_atr", 0.20)
        )
        self.min_upper_touches = max(
            1,
            int(cfg.get("range_min_upper_touches", 1)),
        )
        self.min_lower_touches = max(
            1,
            int(cfg.get("range_min_lower_touches", 1)),
        )
        self.min_total_touches = max(
            2,
            int(cfg.get("range_min_total_touches", 3)),
        )
        self.touch_cooldown_bars = max(
            0,
            int(cfg.get("range_touch_cooldown_bars", 2)),
        )

        # ============================================================
        # 3. ATR / 趋势过滤
        # ============================================================
        self.atr_period = max(
            5,
            int(cfg.get("range_atr_period", 14)),
        )

        self.adx_period = max(
            5,
            int(cfg.get("range_adx_period", 14)),
        )
        self.max_adx = float(
            cfg.get("range_max_adx", 24.0)
        )

        self.er_period = max(
            5,
            int(cfg.get("range_er_period", self.window)),
        )
        self.max_efficiency_ratio = float(
            cfg.get("range_max_efficiency_ratio", 0.42)
        )

        self.ema_period = max(
            10,
            int(cfg.get("range_ema_period", 50)),
        )
        self.ema_slope_bars = max(
            1,
            int(cfg.get("range_ema_slope_bars", 5)),
        )

        # EMA 在 ema_slope_bars 内最多移动多少 ATR。
        self.max_ema_slope_atr = float(
            cfg.get("range_max_ema_slope_atr", 1.15)
        )

        # 对危险方向设置更严格的趋势约束。
        # 例如 EMA 明显下行时做多，或者明显上行时做空。
        self.max_countertrend_slope_atr = float(
            cfg.get("range_max_countertrend_slope_atr", 0.75)
        )

        # ============================================================
        # 4. 波动率状态过滤
        # ============================================================
        self.use_volatility_filter = bool(
            cfg.get("range_use_volatility_filter", True)
        )
        self.volatility_lookback = max(
            30,
            int(cfg.get("range_volatility_lookback", 120)),
        )
        self.min_volatility_rank = float(
            cfg.get("range_min_volatility_rank", 0.05)
        )
        self.max_volatility_rank = float(
            cfg.get("range_max_volatility_rank", 0.92)
        )

        # ============================================================
        # 5. 假突破 setup bar
        # ============================================================
        # 最小越界距离，单位 ATR。
        self.min_sweep_atr = float(
            cfg.get("range_min_sweep_atr", 0.05)
        )

        # 最大越界距离。越界过大可能是真突破或清算行情。
        self.max_sweep_atr = float(
            cfg.get("range_max_sweep_atr", 1.50)
        )

        # setup 收盘必须重新进入区间多少 ATR。
        self.min_reclaim_atr = float(
            cfg.get("range_min_reclaim_atr", 0.03)
        )

        # setup K 线最小影线比例。
        self.min_rejection_wick_ratio = float(
            cfg.get("range_min_rejection_wick_ratio", 0.20)
        )

        # setup 收盘位置。
        # 多头 setup 收盘必须位于自身 K 线的上方一定比例；
        # 空头 setup 收盘必须位于自身 K 线的下方一定比例。
        self.min_setup_close_location = float(
            cfg.get("range_min_setup_close_location", 0.52)
        )

        # setup bar 最大真实波幅，防止在清算巨震中接刀。
        self.max_setup_range_atr = float(
            cfg.get("range_max_setup_range_atr", 3.25)
        )

        # ============================================================
        # 6. confirmation bar
        # ============================================================
        # confirmation 收盘相对 setup 收盘至少继续推进多少 ATR。
        self.min_confirmation_atr = float(
            cfg.get("range_min_confirmation_atr", 0.03)
        )

        # 是否要求 confirmation K 线方向与交易方向一致。
        self.require_confirmation_body = bool(
            cfg.get("range_require_confirmation_body", True)
        )

        # confirmation 不得再次越过 setup 极值太多。
        self.max_confirmation_retest_atr = float(
            cfg.get("range_max_confirmation_retest_atr", 0.10)
        )

        # 确认后不允许追价过深。
        # 多头入场位置不得高于该区间位置；
        # 空头入场位置不得低于 1 - max_entry_position。
        self.max_entry_position = float(
            cfg.get("range_max_entry_position", 0.30)
        )

        # last 与 confirmation close 相差过大时不交易。
        self.max_entry_gap_atr = float(
            cfg.get("range_max_entry_gap_atr", 0.50)
        )

        # ============================================================
        # 7. 止损和止盈
        # ============================================================
        # 止损置于 setup/confirmation 极值外侧。
        self.stop_buffer_atr = float(
            cfg.get("range_stop_buffer_atr", 0.15)
        )

        self.min_sl_pct = float(
            cfg.get("range_min_sl_pct", 0.004)
        )
        self.max_sl_pct = float(
            cfg.get("range_max_sl_pct", 0.06)
        )

        # 从对应边界向中部推进的比例。
        # 0.48 表示略早于精确中轴，提升成交概率。
        self.target_fraction = float(
            cfg.get("range_target_fraction", 0.48)
        )
        # 严格保持“回归中部附近”，不允许目标越过中轴。
        self.target_fraction = float(
            np.clip(self.target_fraction, 0.30, 0.50)
        )

        # 扣除成本后的最小净 RR。
        self.min_net_rr = float(
            cfg.get("range_min_net_rr", 1.05)
        )

        # 防止名义 RR 极端值抬高 confidence。
        self.max_rr_for_confidence = float(
            cfg.get("range_max_rr_for_confidence", 2.50)
        )
        self.optimal_sweep_atr = float(
            cfg.get("range_optimal_sweep_atr", 0.45)
        )

        # ============================================================
        # 8. 成交成本
        # ============================================================
        # round-trip 手续费总和。
        self.fee_round_trip = float(
            cfg.get("range_fee_round_trip", 0.0010)
        )

        # 单边滑点，两次成交时乘 2。
        self.slippage_one_way = float(
            cfg.get("range_slippage_one_way", 0.0002)
        )

        # 额外安全成本，覆盖资金费、跳价和模型误差。
        self.execution_cost_buffer = float(
            cfg.get("range_execution_cost_buffer", 0.0001)
        )
        self.funding_interval_hours = float(
            cfg.get("range_funding_interval_hours", 8.0)
        )
        timeout_bars = cfg.get("rng_timeout_bars", cfg.get("range_timeout_bars", 0))
        try:
            self.timeout_bars = float(timeout_bars or 0.0)
        except (TypeError, ValueError):
            self.timeout_bars = 0.0
        try:
            self.position_timeout_sec = float(cfg.get("position_timeout_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            self.position_timeout_sec = 0.0
        self.main_bar = str(
            cfg.get("_main_bar") or
            cfg.get("main_bar") or
            cfg.get("signal_bar") or
            "1H"
        )

        # ============================================================
        # 9. 仓位和风险
        # ============================================================
        self.risk_pct = float(
            cfg.get("range_risk_pct", 0.0015)
        )

        # 最大名义价值占权益比例。
        # 例如 0.08 表示最多使用权益的 8% 名义价值。
        self.max_notional_pct = float(
            cfg.get("range_max_notional_pct", 0.08)
        )

        # 继续保留 _calc_size 的基础仓位入口，
        # 但最终一定会被风险预算和名义限额截断。
        self.position_pct = float(
            cfg.get("range_position_pct", 0.08)
        )

        self.short_size_multiplier = float(
            cfg.get("range_short_size_multiplier", 1.0)
        )
        self.long_size_multiplier = float(
            cfg.get("range_long_size_multiplier", 1.0)
        )

        # ============================================================
        # 10. 置信度
        # ============================================================
        self.base_confidence = float(
            cfg.get(
                "range_base_confidence",
                cfg.get("min_open_confidence", 0.55),
            )
        )

    # ================================================================
    # 指标函数
    # ================================================================

    @staticmethod
    def _true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _atr(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        tr = cls._true_range(high, low, close)

        return tr.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()

    @classmethod
    def _adx(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where(
                (up_move > down_move) & (up_move > 0),
                up_move,
                0.0,
            ),
            index=high.index,
            dtype=float,
        )

        minus_dm = pd.Series(
            np.where(
                (down_move > up_move) & (down_move > 0),
                down_move,
                0.0,
            ),
            index=high.index,
            dtype=float,
        )

        atr = cls._atr(high, low, close, period)

        plus_dm_smoothed = plus_dm.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        minus_dm_smoothed = minus_dm.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        safe_atr = atr.replace(0.0, np.nan)

        plus_di = 100.0 * plus_dm_smoothed / safe_atr
        minus_di = 100.0 * minus_dm_smoothed / safe_atr

        denominator = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / denominator

        return dx.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()

    @staticmethod
    def _efficiency_ratio(
        close: pd.Series,
        period: int,
    ) -> float:
        if close is None or len(close) < period + 1:
            return float("nan")

        segment = close.iloc[-(period + 1):].astype(float)

        net_change = abs(
            float(segment.iloc[-1]) -
            float(segment.iloc[0])
        )
        path = float(segment.diff().abs().sum())

        if not np.isfinite(path):
            return float("nan")

        if path <= 1e-12:
            return 0.0

        return net_change / path

    @staticmethod
    def _last_finite(series: pd.Series) -> float:
        if series is None or len(series) == 0:
            return float("nan")

        value = float(series.iloc[-1])
        return value if np.isfinite(value) else float("nan")


    @staticmethod
    def _count_touch_events(mask: pd.Series, cooldown: int = 2) -> int:
        """统计有间隔的独立触边事件，而不是连续触边 K 线数量。"""
        if mask is None or len(mask) == 0:
            return 0
        count = 0
        last_index = -cooldown - 1
        for i, touched in enumerate(mask.astype(bool).to_numpy()):
            if bool(touched) and i - last_index > cooldown:
                count += 1
                last_index = i
        return count

    @staticmethod
    def _bar_hours(bar: str) -> float:
        text = str(bar or "").strip().upper()
        if not text:
            return 1.0
        try:
            if text.endswith("M"):
                return max(1.0, float(text[:-1] or 1.0)) / 60.0
            if text.endswith("H"):
                return max(1.0, float(text[:-1] or 1.0))
            if text.endswith("D") or text in ("1DAY", "DAY"):
                n = text[:-1] if text.endswith("D") else "1"
                return max(1.0, float(n or 1.0)) * 24.0
        except ValueError:
            return 1.0
        return 1.0

    def _estimate_funding_cost(self, direction: str, funding_rate: float) -> float:
        """保守估计预计持仓内的不利资金费成本；有利资金费不计入收益。"""
        try:
            fr = float(funding_rate)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(fr) or fr == 0:
            return 0.0

        if self.timeout_bars > 0:
            holding_hours = self.timeout_bars * self._bar_hours(self.main_bar)
        elif self.position_timeout_sec > 0:
            holding_hours = self.position_timeout_sec / 3600.0
        else:
            holding_hours = self.funding_interval_hours

        interval = max(self.funding_interval_hours, 1e-12)
        settlements = max(0.0, holding_hours / interval)

        if direction == "long" and fr > 0:
            return fr * settlements
        if direction == "short" and fr < 0:
            return abs(fr) * settlements
        return 0.0

    @staticmethod
    def _bar_features(
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> Tuple[float, float, float, float]:
        """
        返回：
            bar_range
            upper_wick_ratio
            lower_wick_ratio
            close_location

        close_location:
            0 = 收在最低点
            1 = 收在最高点
        """
        bar_range = max(high_price - low_price, 1e-12)

        upper_wick = max(
            0.0,
            high_price - max(open_price, close_price),
        )
        lower_wick = max(
            0.0,
            min(open_price, close_price) - low_price,
        )

        upper_wick_ratio = upper_wick / bar_range
        lower_wick_ratio = lower_wick / bar_range
        close_location = (close_price - low_price) / bar_range

        return (
            bar_range,
            upper_wick_ratio,
            lower_wick_ratio,
            close_location,
        )

    @staticmethod
    def _spec_float(
        specs: dict,
        names: Tuple[str, ...],
        default: float,
    ) -> float:
        for name in names:
            value = specs.get(name)
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(value) and value > 0:
                return value

        return float(default)

    def _quantize_size(
        self,
        size: float,
        specs: dict,
    ) -> float:
        """
        按交易所 lot size 向下取整。

        如果 specs 中没有 lot size，则保持原值，
        让订单层继续执行自身的数量标准化。
        """
        if not np.isfinite(size) or size <= 0:
            return 0.0

        lot_size = self._spec_float(
            specs,
            (
                "lot_sz",
                "lotSz",
                "lot_size",
                "qty_step",
                "step_size",
            ),
            0.0,
        )

        min_size = self._spec_float(
            specs,
            (
                "min_sz",
                "minSz",
                "min_size",
                "min_qty",
            ),
            0.0,
        )

        result = float(size)

        if lot_size > 0:
            result = math.floor(
                (result + 1e-12) / lot_size
            ) * lot_size

        if min_size > 0 and result + 1e-12 < min_size:
            return 0.0

        return max(0.0, result)

    # ================================================================
    # 主信号函数
    # ================================================================

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

        # ------------------------------------------------------------
        # 数据检查
        # ------------------------------------------------------------
        if df is None or len(df) == 0:
            return None

        required_columns = {
            "open",
            "high",
            "low",
            "close",
        }
        if not required_columns.issubset(df.columns):
            return None

        need = max(
            self.window + 2,
            self.atr_period * 3 + 5,
            self.adx_period * 3 + 5,
            self.er_period + 3,
            self.ema_period + self.ema_slope_bars + 5,
            (
                self.volatility_lookback +
                self.atr_period +
                5
            )
            if self.use_volatility_filter
            else 0,
        )

        if len(df) < need:
            return None

        if not np.isfinite(capital) or capital <= 0:
            return None

        if not np.isfinite(leverage) or leverage <= 0:
            leverage = 1.0

        # ------------------------------------------------------------
        # 外部 regime：仅作为可选额外约束
        # ------------------------------------------------------------
        regime_text = str(regime or "unknown").strip().lower()

        if self.use_external_regime:
            if regime_text not in self.allowed_regimes:
                return None

        # ------------------------------------------------------------
        # OHLC
        # ------------------------------------------------------------
        open_ = pd.to_numeric(
            df["open"],
            errors="coerce",
        ).astype(float)

        high = pd.to_numeric(
            df["high"],
            errors="coerce",
        ).astype(float)

        low = pd.to_numeric(
            df["low"],
            errors="coerce",
        ).astype(float)

        close = pd.to_numeric(
            df["close"],
            errors="coerce",
        ).astype(float)

        if (
            open_.iloc[-need:].isna().any()
            or high.iloc[-need:].isna().any()
            or low.iloc[-need:].isna().any()
            or close.iloc[-need:].isna().any()
        ):
            return None

        # ------------------------------------------------------------
        # 指标：上下文 / setup / confirmation 分离，避免信号 K 线抬高过滤阈值
        # ------------------------------------------------------------
        atr_series = self._atr(
            high,
            low,
            close,
            self.atr_period,
        )
        atr_context = self._last_finite(atr_series.iloc[:-2])
        atr_setup = self._last_finite(atr_series.iloc[:-1])
        atr_confirm = self._last_finite(atr_series)

        if (
            not np.isfinite(atr_context) or atr_context <= 0
            or not np.isfinite(atr_setup) or atr_setup <= 0
            or not np.isfinite(atr_confirm) or atr_confirm <= 0
        ):
            return None

        adx_series = self._adx(
            high,
            low,
            close,
            self.adx_period,
        )
        adx_context = self._last_finite(adx_series.iloc[:-2])

        if not np.isfinite(adx_context):
            return None

        if adx_context > self.max_adx:
            return None

        er = self._efficiency_ratio(
            close.iloc[:-2],
            self.er_period,
        )

        if not np.isfinite(er):
            return None

        if er > self.max_efficiency_ratio:
            return None

        ema = close.ewm(
            span=self.ema_period,
            adjust=False,
            min_periods=self.ema_period,
        ).mean()

        ema_context = ema.iloc[:-2]
        if (
            len(ema_context) <= self.ema_slope_bars
            or not np.isfinite(float(ema_context.iloc[-1]))
            or not np.isfinite(
                float(ema_context.iloc[-1 - self.ema_slope_bars])
            )
        ):
            return None

        ema_delta = (
            float(ema_context.iloc[-1]) -
            float(ema_context.iloc[-1 - self.ema_slope_bars])
        )
        ema_slope_atr = ema_delta / atr_context

        if abs(ema_slope_atr) > self.max_ema_slope_atr:
            return None

        # ------------------------------------------------------------
        # ATR 波动率分位过滤：使用 setup 之前的上下文状态
        # ------------------------------------------------------------
        volatility_rank = 0.50

        if self.use_volatility_filter:
            atr_pct_series = atr_series / close.replace(0.0, np.nan)

            history = (
                atr_pct_series
                .iloc[-self.volatility_lookback - 3:-3]
                .dropna()
            )

            current_atr_pct = float(atr_pct_series.iloc[-3])

            if (
                len(history) < max(
                    30,
                    self.volatility_lookback // 2,
                )
                or not np.isfinite(current_atr_pct)
            ):
                return None

            volatility_rank = float(
                (history <= current_atr_pct).mean()
            )

            if volatility_rank < self.min_volatility_rank:
                return None

            if volatility_rank > self.max_volatility_rank:
                return None

        # ------------------------------------------------------------
        # 区间：严格排除 setup 和 confirmation
        # ------------------------------------------------------------
        base_high = high.iloc[-self.window - 2:-2]
        base_low = low.iloc[-self.window - 2:-2]
        base_close = close.iloc[-self.window - 2:-2]

        if (
            len(base_high) != self.window
            or len(base_low) != self.window
            or len(base_close) != self.window
        ):
            return None

        range_high = float(base_high.max())
        range_low = float(base_low.min())
        range_width = range_high - range_low
        confirmation_close = float(close.iloc[-1])

        if (
            not np.isfinite(range_high)
            or not np.isfinite(range_low)
            or not np.isfinite(range_width)
            or range_width <= 0
            or confirmation_close <= 0
        ):
            return None

        range_pct = range_width / confirmation_close
        range_atr = range_width / atr_context

        if range_pct < self.min_range_pct:
            return None

        if range_pct > self.max_range_pct:
            return None

        if range_atr < self.min_range_atr:
            return None

        if range_atr > self.max_range_atr:
            return None

        # ------------------------------------------------------------
        # 边界测试次数
        # ------------------------------------------------------------
        touch_tolerance = max(
            self.touch_tolerance_atr * atr_context,
            confirmation_close * 1e-5,
        )

        upper_touch_mask = base_high >= range_high - touch_tolerance
        lower_touch_mask = base_low <= range_low + touch_tolerance
        upper_touches = self._count_touch_events(
            upper_touch_mask,
            cooldown=self.touch_cooldown_bars,
        )
        lower_touches = self._count_touch_events(
            lower_touch_mask,
            cooldown=self.touch_cooldown_bars,
        )

        total_touches = upper_touches + lower_touches

        if upper_touches < self.min_upper_touches:
            return None

        if lower_touches < self.min_lower_touches:
            return None

        if total_touches < self.min_total_touches:
            return None

        # ------------------------------------------------------------
        # setup bar = -2
        # confirmation bar = -1
        # ------------------------------------------------------------
        setup_open = float(open_.iloc[-2])
        setup_high = float(high.iloc[-2])
        setup_low = float(low.iloc[-2])
        setup_close = float(close.iloc[-2])

        confirm_open = float(open_.iloc[-1])
        confirm_high = float(high.iloc[-1])
        confirm_low = float(low.iloc[-1])
        confirm_close = float(close.iloc[-1])

        (
            setup_bar_range,
            setup_upper_wick_ratio,
            setup_lower_wick_ratio,
            setup_close_location,
        ) = self._bar_features(
            setup_open,
            setup_high,
            setup_low,
            setup_close,
        )

        if setup_bar_range / atr_setup > self.max_setup_range_atr:
            return None

        # ------------------------------------------------------------
        # 多头与空头假突破条件
        # ------------------------------------------------------------

        # 多头：前一根向下刺破区间后收回。
        long_sweep_distance = range_low - setup_low

        long_sweep = (
            long_sweep_distance >=
            self.min_sweep_atr * atr_setup
            and long_sweep_distance <=
            self.max_sweep_atr * atr_setup
        )

        long_reclaim = (
            range_low + self.min_reclaim_atr * atr_setup <= setup_close <= range_high
        )

        long_rejection = (
            setup_lower_wick_ratio >=
            self.min_rejection_wick_ratio
            and setup_close_location >=
            self.min_setup_close_location
        )

        long_confirmation = (
            range_low <= confirm_close <= range_high
            and confirm_close >= setup_close + self.min_confirmation_atr * atr_setup
            and confirm_low >= setup_low - self.max_confirmation_retest_atr * atr_setup
        )

        if self.require_confirmation_body:
            long_confirmation = (
                long_confirmation
                and confirm_close > confirm_open
            )

        # 空头：前一根向上刺破区间后收回。
        short_sweep_distance = setup_high - range_high

        short_sweep = (
            short_sweep_distance >=
            self.min_sweep_atr * atr_setup
            and short_sweep_distance <=
            self.max_sweep_atr * atr_setup
        )

        short_reclaim = (
            range_low <= setup_close <= range_high - self.min_reclaim_atr * atr_setup
        )

        short_rejection = (
            setup_upper_wick_ratio >=
            self.min_rejection_wick_ratio
            and setup_close_location <=
            1.0 - self.min_setup_close_location
        )

        short_confirmation = (
            range_low <= confirm_close <= range_high
            and confirm_close <= setup_close - self.min_confirmation_atr * atr_setup
            and confirm_high <= setup_high + self.max_confirmation_retest_atr * atr_setup
        )

        if self.require_confirmation_body:
            short_confirmation = (
                short_confirmation
                and confirm_close < confirm_open
            )

        long_signal = (
            long_sweep
            and long_reclaim
            and long_rejection
            and long_confirmation
        )

        short_signal = (
            short_sweep
            and short_reclaim
            and short_rejection
            and short_confirmation
        )

        # 极端情况下同一组 K 线可能同时穿越两侧，
        # 这通常意味着区间太窄或发生异常波动，直接跳过。
        if long_signal and short_signal:
            return None

        if not long_signal and not short_signal:
            return None

        direction = "long" if long_signal else "short"

        if direction == "long" and not self.allow_long:
            return None

        if direction == "short" and not self.allow_short:
            return None

        # ------------------------------------------------------------
        # 方向性 EMA 斜率过滤
        # ------------------------------------------------------------
        # EMA 明显下行时，不允许逆势做多。
        if (
            direction == "long"
            and ema_slope_atr <
            -self.max_countertrend_slope_atr
        ):
            return None

        # EMA 明显上行时，不允许逆势做空。
        if (
            direction == "short"
            and ema_slope_atr >
            self.max_countertrend_slope_atr
        ):
            return None

        if self._funding_blocked(direction, funding_rate):
            return None

        # ------------------------------------------------------------
        # 入场价
        # ------------------------------------------------------------
        # 信号价格固定使用 confirmation close，保证回测与实盘信号可复现。
        # last 只作为实时偏离检查，真正成交价由执行层决定。
        entry = confirm_close
        market_ref = (
            float(last)
            if np.isfinite(last) and last > 0
            else confirm_close
        )

        if not np.isfinite(entry) or entry <= 0:
            return None

        # 防止实盘 last 与已完成 K 线收盘偏差太大。
        entry_gap_atr = abs(market_ref - confirm_close) / atr_confirm

        if entry_gap_atr > self.max_entry_gap_atr:
            return None

        entry_position = (
            entry - range_low
        ) / range_width

        if direction == "long":
            if entry_position < 0:
                return None

            if entry_position > self.max_entry_position:
                return None
        else:
            if entry_position > 1:
                return None

            if entry_position < 1.0 - self.max_entry_position:
                return None

        # ------------------------------------------------------------
        # 结构止损与中部止盈
        # ------------------------------------------------------------
        stop_buffer = max(
            self.stop_buffer_atr * atr_confirm,
            entry * self.min_sl_pct * 0.25,
        )

        if direction == "long":
            structural_low = min(
                range_low,
                setup_low,
                confirm_low,
            )

            stop = structural_low - stop_buffer

            target = (
                range_low +
                self.target_fraction * range_width
            )

            # 如果目标已经被确认 K 线走过，则不追价。
            if target <= entry:
                return None

            raw_stop_distance = entry - stop

        else:
            structural_high = max(
                range_high,
                setup_high,
                confirm_high,
            )

            stop = structural_high + stop_buffer

            target = (
                range_high -
                self.target_fraction * range_width
            )

            if target >= entry:
                return None

            raw_stop_distance = stop - entry

        if (
            not np.isfinite(stop)
            or not np.isfinite(target)
            or raw_stop_distance <= 0
        ):
            return None

        # 强制最小止损距离。
        min_stop_distance = entry * self.min_sl_pct

        if raw_stop_distance < min_stop_distance:
            if direction == "long":
                stop = entry - min_stop_distance
            else:
                stop = entry + min_stop_distance

        stop_distance = abs(entry - stop)
        reward_distance = abs(target - entry)

        if stop_distance <= 0 or reward_distance <= 0:
            return None

        sl_pct = stop_distance / entry

        if sl_pct < self.min_sl_pct * 0.999:
            return None

        if sl_pct > self.max_sl_pct:
            return None

        # ------------------------------------------------------------
        # 成本调整后的 RR
        # ------------------------------------------------------------
        expected_funding_pct = self._estimate_funding_cost(direction, funding_rate)
        round_trip_cost_pct = (
            self.fee_round_trip
            + 2.0 * self.slippage_one_way
            + self.execution_cost_buffer
            + expected_funding_pct
        )

        cost_distance = entry * round_trip_cost_pct

        net_reward_distance = (
            reward_distance - cost_distance
        )
        net_risk_distance = (
            stop_distance + cost_distance
        )

        if net_reward_distance <= 0:
            return None

        if net_risk_distance <= 0:
            return None

        gross_rr = reward_distance / stop_distance
        net_rr = net_reward_distance / net_risk_distance

        if net_rr < self.min_net_rr:
            return None

        # ------------------------------------------------------------
        # 仓位计算
        # ------------------------------------------------------------
        try:
            kelly = float(kelly_factor)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(kelly) or kelly <= 0:
            return None
        # 当前引擎 Kelly 本身是降仓系数；策略侧再次显式裁剪，避免风险预算放大。
        kelly = float(np.clip(kelly, 0.0, 1.0))

        direction_multiplier = (
            self.long_size_multiplier
            if direction == "long"
            else self.short_size_multiplier
        )

        position_pct = (
            self.position_pct *
            max(0.0, direction_multiplier)
        )

        if position_pct <= 0:
            return None

        base_size = self._calc_size(
            capital, leverage, position_pct, entry, specs, kelly,
            sl_distance_pct=sl_pct,
            atr_pct=atr_confirm / entry,
        )

        if not np.isfinite(base_size) or base_size <= 0:
            return None

        contract_value = self._spec_float(
            specs,
            (
                "ct_val",
                "ctVal",
                "contract_value",
            ),
            float("nan"),
        )

        if not np.isfinite(contract_value) or contract_value <= 0:
            return None

        # 每张合约止损对应的 USDT 风险。
        risk_per_contract = (
            stop_distance * contract_value
        )

        if risk_per_contract <= 0:
            return None

        risk_budget = (
            capital *
            self.risk_pct *
            kelly
        )

        if risk_budget <= 0:
            return None

        max_size_by_risk = (
            risk_budget /
            risk_per_contract
        )

        if self.max_notional_pct > 0:
            max_notional = (
                capital *
                self.max_notional_pct *
                max(0.0, direction_multiplier)
            )
        else:
            max_notional = (
                capital *
                leverage *
                position_pct
            )

        notional_per_contract = (
            entry * contract_value
        )

        if notional_per_contract <= 0:
            return None

        max_size_by_notional = (
            max_notional /
            notional_per_contract
        )

        size_limits = {
            "base": float(base_size),
            "risk": float(max_size_by_risk),
            "notional": float(max_size_by_notional),
        }
        size_limiter = min(size_limits, key=size_limits.get)
        raw_size = size_limits[size_limiter]

        size = self._quantize_size(
            raw_size,
            specs,
        )
        if size + 1e-12 < raw_size:
            size_limiter = "lot_size"

        if size <= 0:
            return None

        planned_price_risk = (
            size *
            stop_distance *
            contract_value
        )

        # 加入预估退出成本后的风险检查。
        estimated_cost = (
            size *
            entry *
            contract_value *
            round_trip_cost_pct
        )

        planned_total_risk = (
            planned_price_risk +
            estimated_cost
        )

        max_total_risk = risk_budget * 1.10

        if planned_total_risk > max_total_risk:
            # 第二次按含成本风险缩放。
            risk_per_contract_with_cost = (
                (
                    stop_distance +
                    cost_distance
                ) *
                contract_value
            )

            if risk_per_contract_with_cost <= 0:
                return None

            resized = risk_budget / risk_per_contract_with_cost

            if resized < size:
                size_limiter = "risk_with_cost"
            size = min(size, resized)
            size = self._quantize_size(size, specs)

            if size <= 0:
                return None

        # ------------------------------------------------------------
        # 置信度
        # ------------------------------------------------------------
        if direction == "long":
            sweep_atr = long_sweep_distance / atr_setup
            wick_ratio = setup_lower_wick_ratio
        else:
            sweep_atr = short_sweep_distance / atr_setup
            wick_ratio = setup_upper_wick_ratio

        # 适度 sweep 最佳，不奖励越接近最大允许值的极端越界。
        optimal_sweep = max(self.optimal_sweep_atr, 1e-12)
        sweep_quality = float(
            np.clip(
                1.0 - abs(sweep_atr - optimal_sweep) / optimal_sweep,
                0.0,
                1.0,
            )
        )

        wick_quality = float(
            np.clip(
                (
                    wick_ratio -
                    self.min_rejection_wick_ratio
                ) / max(
                    1.0 -
                    self.min_rejection_wick_ratio,
                    1e-12,
                ),
                0.0,
                1.0,
            )
        )

        rr_quality = float(
            np.clip(
                (
                    net_rr -
                    self.min_net_rr
                ) / max(
                    self.max_rr_for_confidence -
                    self.min_net_rr,
                    1e-12,
                ),
                0.0,
                1.0,
            )
        )

        trend_quality = float(
            np.clip(
                1.0 - adx_context / max(self.max_adx, 1e-12),
                0.0,
                1.0,
            )
        )

        touch_quality = float(
            np.clip(
                (
                    total_touches -
                    self.min_total_touches
                ) / 4.0,
                0.0,
                1.0,
            )
        )

        confidence = self.base_confidence
        confidence += 0.05 * sweep_quality
        confidence += 0.05 * wick_quality
        confidence += 0.07 * rr_quality
        confidence += 0.04 * trend_quality
        confidence += 0.03 * touch_quality

        confidence = float(
            np.clip(
                confidence,
                0.0,
                0.92,
            )
        )

        # ------------------------------------------------------------
        # 信号说明
        # ------------------------------------------------------------
        side = "L" if direction == "long" else "S"

        reason = (
            f"[RNG-FB]{side} "
            f"pos={entry_position:.2f} "
            f"gross_rr={gross_rr:.2f} "
            f"net_rr={net_rr:.2f} "
            f"adx={adx_context:.1f} "
            f"er={er:.2f} "
            f"ema_slope_atr={ema_slope_atr:.2f} "
            f"vol_rank={volatility_rank:.2f} "
            f"touch={lower_touches}/{upper_touches} "
            f"sweep_atr={sweep_atr:.2f} "
            f"fund_cost={expected_funding_pct:.5f} "
            f"limiter={size_limiter} "
            f"reg={regime_text}"
        )

        return Signal(
            action=(
                "open_long"
                if direction == "long"
                else "open_short"
            ),
            direction=direction,
            confidence=confidence,
            strategy=self.name,
            regime=regime_text,
            stop_loss=float(stop),
            take_profit=float(target),

            # 当前引擎按 rr_list 冻结止盈价；使用 gross_rr 可让实际 TP 等于 target。
            # net_rr 已用于扣成本过滤和置信度计算。
            rr_list=[float(gross_rr)],
            batch_ratios=[1.0],

            size=float(size),

            # 这里返回真实 ATR，而不是整个区间宽度。
            atr=float(atr_confirm),

            reason=reason,
        )