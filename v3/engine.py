# -*- coding: utf-8 -*-
"""策略引擎 — 信号生成 + 持仓管理 + 风险控制。

架构:
  StrategyEngine
    ├── RegimeDetector    检测当前市场状态
    ├── HtfFilter         高周期趋势过滤（硬挡）
    ├── ConflictFilter    跨策略冲突过滤
    ├── Strategy plugins  自动发现 v3/strategies/*（见 registry）
    └── Position          持仓状态机（止损/分批/Chandelier/time-stop/panic）

职责:
  1. 策略是独立模块，通过注册表加载
  2. Regime 决定哪些策略可开仓（trend 策略 vs chop 策略互斥）
  3. HTF 过滤是硬挡：过滤后无同向候选直接返回 hold
  4. 指标计算在 v3.indicators 中，策略不重复实现
  5. 信号周期默认 1H
  6. 成交确认：generate_signal 只产出信号，不登记持仓；
     由外层回测在真实成交后调用 confirm_* 才登记，避免幻影仓位
  7. 止损/止盈档位在开仓时冻结：sl_dist 与 RR 档位存入 Position，
     不再每根 bar 用当前 ATR 重算
  8. 仓位生命周期状态机：策略只发 TradePlan，引擎是仓位状态唯一真源；
     状态机按 panic > time-stop > SL > +1R > +1.5R partial > +2R Chandelier 顺序执行
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from v3.indicators import adx, atr, rsi
from v3.risk import RegimeDetector, HtfFilter, ConflictFilter
from v3.strategies import REGISTRY, Signal, StrategyBase

from logger import setup_logger
log = setup_logger("engine")

_DEFAULT_RR = [1.0, 1.5, 2.5]
_DEFAULT_BATCH = [0.30, 0.30, 0.40]


def _bar_to_seconds(bar: str) -> float:
    text = str(bar or "").strip().upper()
    if not text:
        return 60.0
    try:
        if text.endswith("M"):
            return max(1.0, float(text[:-1] or 1)) * 60.0
        if text.endswith("H"):
            return max(1.0, float(text[:-1] or 1)) * 3600.0
        if text.endswith("D") or text in ("1DAY", "DAY"):
            n = text[:-1] if text.endswith("D") else "1"
            return max(1.0, float(n or 1)) * 86400.0
    except ValueError:
        return 60.0
    return 60.0


@dataclass
class Position:
    long_sz: float = 0.0
    short_sz: float = 0.0
    entry_long: float = 0.0
    entry_short: float = 0.0
    mode: str = ""  # 策略名
    direction: str = "flat"
    tp_batch: int = 0
    trail_stop: Optional[float] = None
    open_ts: float = 0.0
    partial_done: bool = False
    # 开仓时冻结的风险参数
    sl_dist: float = 0.0
    rr_list: List[float] = field(default_factory=lambda: list(_DEFAULT_RR))
    batch_ratios: List[float] = field(default_factory=lambda: list(_DEFAULT_BATCH))
    atr_entry: float = 0.0
    entry_bar_ts: float = 0.0
    last_decay_ts: float = 0.0
    # 仓位生命周期状态机（ming 等"策略自管退出"策略用）
    partials_filled: List[int] = field(default_factory=list)
    mfe_r: float = 0.0  # 最高有利偏移，单位 R
    mae_r: float = 0.0  # 最大不利偏移，单位 R
    chandelier_on: bool = False
    chandelier_high: float = 0.0
    chandelier_low: float = 0.0
    bars_held: int = 0
    entry_ref_high: float = 0.0  # 入场参考价（来自 Signal），用于 time-stop
    entry_ref_low: float = 0.0
    # time-stop / panic 配置
    mfe_window_bars: int = 0
    mfe_min_r: float = 0.0
    breakout_window_bars: int = 0
    max_bars: int = 0
    panic_gap_atr: float = 0.0
    # +1R / +1.5R / +2R 配置
    r1_be_buffer_atr: float = 0.2
    r15_partial_pct: float = 0.30
    r2_chandelier_N: int = 10
    r2_chandelier_k: float = 3.0


class StrategyEngine:
    """策略引擎：regime 检测 → 策略候选 → HTF 过滤 → 冲突过滤 → 持仓管理。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.strat_cfg = cfg.get("strategy", {})

        self.regime = RegimeDetector(self.strat_cfg)
        self.htf = HtfFilter(self.strat_cfg)
        self.conflict = ConflictFilter(self.strat_cfg)

        from v3.strategies.loader import build_strategy_cfg
        from v3.strategies.registry import all_strategies, list_names
        available = all_strategies()
        enabled = list(
            self.strat_cfg.get("enabled_strategies")
            or ["vol", "mr", "ewmac", "macd", "don", "ming"]
        )
        cli_overrides = self.strat_cfg.get("_cli_overrides") or []
        extra_config = self.strat_cfg.get("_extra_config") or ""
        self.strategies: Dict[str, StrategyBase] = {}
        missing = []
        for name in enabled:
            name = str(name).strip()
            if not name:
                continue
            cls = available.get(name) or REGISTRY.get(name)
            if cls is None:
                missing.append(name)
                continue
            try:
                per_strategy_cfg = build_strategy_cfg(
                    name, self.strat_cfg,
                    extra_config_path=extra_config,
                    cli_overrides=cli_overrides,
                )
                self.strategies[name] = cls(per_strategy_cfg)
            except Exception as e:
                log.error(f"策略 {name} 实例化失败: {e}")
        if missing:
            log.warning(f"未找到策略 {missing}；已发现: {list_names()}")
        if not self.strategies:
            log.warning(f"无任何已启用策略被加载。enabled={enabled} discovered={list_names()}")

        self.positions: Dict[str, Position] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._consec_loss: Dict[str, int] = {}
        self._cb_until: Dict[str, float] = {}
        self._recent_signals: Dict[str, List[Dict]] = {}
        self.signal_bar = str(self.strat_cfg.get("signal_bar", "1H"))
        self._pnl_history: Dict[str, List[float]] = {}
        self._sim_ts: Optional[float] = None

        self.min_conf = float(self.strat_cfg.get("min_open_confidence", 0.55))
        self.stop_loss_mult = float(self.strat_cfg.get("stop_loss_atr_mult", 1.8))
        self.trail_mult = float(self.strat_cfg.get("trail_atr_mult", 2.0))
        self.timeout_sec = float(self.strat_cfg.get("position_timeout_sec", 86400))
        self.exit_mode = str(self.strat_cfg.get("exit_mode", "timeout"))
        self.decay_step_hours = float(self.strat_cfg.get("time_decay_step_hours", 6.0))
        self.decay_step_frac = float(self.strat_cfg.get("time_decay_step_frac", 0.25))
        self.funding_tilt = bool(self.strat_cfg.get("funding_tilt_enabled", True))
        self.funding_tilt_conf = float(self.strat_cfg.get("funding_tilt_conf", 0.05))

        # 风险控制：总回撤硬上限 + 小资金腰斩锁仓
        self.max_dd_pct: float = float(self.strat_cfg.get("max_dd_pct", 0.15))
        self.equity_lock_threshold: float = float(
            self.strat_cfg.get("equity_lock_threshold", 1000.0)
        )
        self.equity_lock_drawdown: float = float(
            self.strat_cfg.get("equity_lock_drawdown", 0.5)
        )
        self.start_equity: float = 0.0
        self.peak_equity: float = 0.0
        self._account_locked: bool = False
        self._lock_reason: str = ""

        enabled_names = list(self.strategies.keys())
        log.info(f"StrategyEngine | strategies={enabled_names} | bar={self.signal_bar}")

    def set_sim_time(self, ts: float) -> None:
        self._sim_ts = ts

    def _now(self) -> float:
        return self._sim_ts if self._sim_ts is not None else time.time()

    def update_equity(self, cur_equity: float) -> None:
        """回测层每根 K 线结束后调用，更新 peak_equity 并检查风控触发。

        风控规则（按优先级）：
          1. 小资金腰斩：仅当 start_equity < equity_lock_threshold 时启用（**优先**）
          2. 总回撤 max_dd_pct：对所有资金量生效

        设计意图：小资金用户的核心需求是"本金腰斩报警"——这一信号要优先于
        max_dd 15% 通用上限判断。
        """
        if cur_equity <= 0:
            return
        if self.start_equity <= 0:
            self.start_equity = cur_equity
        if cur_equity > self.peak_equity:
            self.peak_equity = cur_equity
        if self._account_locked:
            return

        # 优先：小资金腰斩锁仓
        if (
            self.start_equity < self.equity_lock_threshold
            and self.start_equity > 0
        ):
            half = self.start_equity * (1.0 - self.equity_lock_drawdown)
            if cur_equity <= half:
                self._account_locked = True
                self._lock_reason = (
                    f"equity_lock_triggered start={self.start_equity:.2f} "
                    f"cur={cur_equity:.2f} half={half:.2f} "
                    f"({self.equity_lock_drawdown:.0%} 腰斩)"
                )
                log.error(f"[风控] {self._lock_reason}")
                self._force_close_all(f"equity_lock {cur_equity:.2f}<={half:.2f}")
                return

        # 总回撤硬上限
        if self.peak_equity > 0 and self.max_dd_pct > 0:
            dd = (self.peak_equity - cur_equity) / self.peak_equity
            if dd >= self.max_dd_pct:
                self._account_locked = True
                self._lock_reason = (
                    f"max_dd_triggered dd={dd:.2%} >= max_dd_pct={self.max_dd_pct:.2%} "
                    f"peak={self.peak_equity:.2f} cur={cur_equity:.2f}"
                )
                log.error(f"[风控] {self._lock_reason}")
                self._force_close_all(f"max_dd {dd:.2%}")

    def _force_close_all(self, reason: str) -> None:
        """风控触发：把所有仓位 close。"""
        for inst_id, pos in list(self.positions.items()):
            if pos.long_sz > 0 or pos.short_sz > 0:
                self.positions.pop(inst_id, None)
                log.warning(f"[风控] 强制清仓 {inst_id} reason={reason}")
        now = self._now()
        for inst_id in list(self.positions.keys()):
            self._cooldown_until[inst_id] = now + 3600
            self._cb_until[inst_id] = now + 3600

    def _timeout_sec_for_position(self, mode: str) -> float:
        """按策略取超时（bars 优先于 sec，sec 优先于全局默认）。"""
        timeout_sec = float(self.timeout_sec)
        mode = str(mode or "").strip()
        if not mode:
            return timeout_sec

        bars = self.strat_cfg.get(f"{mode}_timeout_bars")
        if bars is not None and str(bars).strip() != "":
            try:
                n_bars = float(bars)
            except (TypeError, ValueError):
                n_bars = 0.0
            if n_bars > 0:
                bar = self.strat_cfg.get("_main_bar") or self.strat_cfg.get("main_bar") or self.signal_bar
                return n_bars * _bar_to_seconds(str(bar))

        sec = self.strat_cfg.get(f"{mode}_timeout_sec")
        if sec is not None and str(sec).strip() != "":
            try:
                value = float(sec)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value

        return timeout_sec

    def generate_signal(
        self,
        inst_id: str,
        df: pd.DataFrame,
        df_htf: Optional[pd.DataFrame],
        capital: float,
        leverage: float,
        specs: dict,
        funding_rate: float = 0.0,
    ) -> Signal:
        """主信号入口。

        不登记持仓：返回 open 信号后由回测层真实成交并调用
        confirm_fill(inst_id, fill_px, filled_sz, signal) 才登记。
        """
        if df is None or len(df) < 60:
            return Signal(action="hold", direction="flat", reason="no data")

        last = float(df["close"].iloc[-1])
        if last <= 0:
            return Signal(action="hold", direction="flat", reason="no price")

        pos = self.positions.get(inst_id, Position())

        # 有仓 → 持仓管理
        if pos.long_sz > 0 or pos.short_sz > 0:
            return self._manage_position(inst_id, pos, df, last, specs)

        if self._account_locked:
            return Signal(action="hold", direction="flat",
                          reason=f"account_locked:{self._lock_reason[:60]}")

        if self._in_cooldown(inst_id):
            return Signal(action="hold", direction="flat", reason="cooldown")

        regime = self.regime.detect(df, last)
        htf_bias = self.htf.bias(df_htf)
        kelly = self._kelly_factor(inst_id, regime.atr_pct)

        candidates: List[Signal] = []
        for name, strategy in self.strategies.items():
            if not strategy.regime_ok(regime.regime):
                continue
            sig = strategy.generate(
                df, regime.regime, last, capital, leverage, specs,
                kelly_factor=kelly, funding_rate=funding_rate,
            )
            if sig and sig.is_open:
                candidates.append(sig)

        if not candidates:
            return Signal(action="hold", direction="flat",
                          reason=f"no signal regime={regime.regime} adx={regime.adx}")

        # HTF 硬挡：过滤后无同向候选直接返回 hold
        if htf_bias in ("long", "short"):
            kept = []
            for c in candidates:
                strat = self.strategies.get(c.strategy)
                if strat is not None and getattr(strat, "skip_htf", False):
                    kept.append(c)
                elif c.direction == htf_bias:
                    kept.append(c)
            candidates = kept
            if not candidates:
                return Signal(action="hold", direction="flat",
                              reason=f"htf_block {htf_bias}")

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        best = candidates[0]

        # 入场门槛（含资金费率 tilt：向付费方向收紧）
        min_conf = self.min_conf
        if self.funding_tilt:
            if best.direction == "long" and funding_rate > 0:
                min_conf += self.funding_tilt_conf
            elif best.direction == "short" and funding_rate < 0:
                min_conf += self.funding_tilt_conf
        if best.confidence < min_conf:
            return Signal(action="hold", direction="flat",
                          reason=f"low_conf {best.confidence:.2f}<{min_conf}")

        now = self._now()
        conflict = self.conflict.check(
            best.direction, best.strategy, best.confidence,
            self._recent_signals.get(inst_id, []), now,
        )
        if conflict:
            return Signal(action="hold", direction="flat", reason=conflict)

        self._record_signal(inst_id, best, now)
        return best

    def _manage_position(
        self, inst_id: str, pos: Position, df: pd.DataFrame,
        last: float, specs: dict,
    ) -> Signal:
        """持仓管理：止损 / 分批止盈 / trailing / 超时 / 时间衰减。

        止损距离与止盈档位在开仓时冻结（pos.sl_dist / pos.rr_list），
        不随当前 ATR 漂移。同根 bar 同时触及 SL 与 TP 时假设先触 SL。
        """
        direction = pos.direction
        entry = pos.entry_long if direction == "long" else pos.entry_short

        if entry <= 0:
            return Signal(action="hold", direction=direction, reason="invalid entry")

        sl_dist = pos.sl_dist if pos.sl_dist > 0 else last * 0.01
        rr_list = pos.rr_list or _DEFAULT_RR
        batch_ratios = pos.batch_ratios or _DEFAULT_BATCH

        now = self._now()
        elapsed = max(0.0, now - float(pos.open_ts or 0.0))

        row = df.iloc[-1]
        bar_open = float(row.get("open", last) or last)
        bar_high = float(row.get("high", last) or last)
        bar_low = float(row.get("low", last) or last)

        # 退出模式：硬超时 或 时间衰减止损
        timeout_sec = self._timeout_sec_for_position(pos.mode)
        if elapsed >= timeout_sec - 1e-9:
            if self.exit_mode == "time_decay":
                step_sec = max(60.0, float(self.decay_step_hours) * 3600.0)
                last_d = float(pos.last_decay_ts or 0.0)
                if last_d <= 0 or (now - last_d >= step_sec - 1e-9):
                    self._tighten_trail(pos, direction, entry)
                    pos.last_decay_ts = now
            else:
                return self._close_signal(inst_id, pos, f"timeout {elapsed:.0f}s")

        # 仓位生命周期状态机（ming 等自管退出策略）
        # 优先级：panic > time-stop > SL > +1R/+1.5R/+2R > 常规 SL
        is_ming_style = (
            pos.mode == "ming"
            or pos.mfe_window_bars > 0
            or pos.max_bars > 0
            or pos.panic_gap_atr > 0
        )

        # 更新 bars_held 与 MFE/MAE
        if pos.atr_entry <= 0:
            try:
                _cur_atr = float(atr(df, 14).iloc[-1])
                if _cur_atr > 0:
                    pos.atr_entry = _cur_atr
            except Exception:
                pos.atr_entry = max(last * 0.01, 1e-6)
        pos.bars_held = int(pos.bars_held) + 1
        if direction == "long" and pos.atr_entry > 0:
            cur_r = (bar_high - entry) / pos.sl_dist if pos.sl_dist > 0 else 0.0
            cur_mae = (entry - bar_low) / pos.sl_dist if pos.sl_dist > 0 else 0.0
        elif direction == "short" and pos.atr_entry > 0:
            cur_r = (entry - bar_low) / pos.sl_dist if pos.sl_dist > 0 else 0.0
            cur_mae = (bar_high - entry) / pos.sl_dist if pos.sl_dist > 0 else 0.0
        else:
            cur_r = 0.0
            cur_mae = 0.0
        if cur_r > pos.mfe_r:
            pos.mfe_r = cur_r
        if cur_mae > pos.mae_r:
            pos.mae_r = cur_mae

        # panic exit：跳空开 > panic_gap_atr*ATR 且未收复
        if is_ming_style and pos.panic_gap_atr > 0 and pos.atr_entry > 0:
            gap_threshold = pos.panic_gap_atr * pos.atr_entry
            if direction == "long":
                if len(df) >= 2:
                    prev_close = float(df["close"].iloc[-2])
                    if (prev_close - bar_open) > gap_threshold:
                        if last < (prev_close - gap_threshold):
                            return self._close_signal(
                                inst_id, pos,
                                f"panic_gap_dn open={bar_open:.4f} prev_close={prev_close:.4f} gap>{gap_threshold:.4f}",
                                fill_price=min(bar_open, last),
                            )
            else:
                if len(df) >= 2:
                    prev_close = float(df["close"].iloc[-2])
                    if (bar_open - prev_close) > gap_threshold:
                        if last > (prev_close + gap_threshold):
                            return self._close_signal(
                                inst_id, pos,
                                f"panic_gap_up open={bar_open:.4f} prev_close={prev_close:.4f} gap>{gap_threshold:.4f}",
                                fill_price=max(bar_open, last),
                            )

        # time-stop：max_bars 硬上限（已 +2R 且 Chandelier 未触发可豁免）
        if is_ming_style and pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
            if not (pos.chandelier_on and pos.mfe_r >= 2.0):
                return self._close_signal(inst_id, pos, f"max_bars {pos.bars_held}>={pos.max_bars}")

        # time-stop：mfe_window_bars 内 MFE < mfe_min_r（动量衰减）
        if (is_ming_style and pos.mfe_window_bars > 0
                and pos.bars_held == pos.mfe_window_bars
                and pos.mfe_r < pos.mfe_min_r):
            return self._close_signal(
                inst_id, pos,
                f"mfe_decay {pos.mfe_r:.2f}R<{pos.mfe_min_r:.2f}R in {pos.mfe_window_bars} bars",
            )

        # time-stop：breakout_window_bars 内未破入场参考 high（突破失败）
        if (is_ming_style and pos.breakout_window_bars > 0
                and pos.entry_ref_high > 0
                and pos.bars_held == pos.breakout_window_bars
                and direction == "long"
                and bar_high < pos.entry_ref_high):
            return self._close_signal(
                inst_id, pos,
                f"breakout_fail high={bar_high:.4f}<ref={pos.entry_ref_high:.4f}",
            )

        # +1R：SL 上移到 entry ± r1_be_buffer_atr*ATR
        if is_ming_style and pos.atr_entry > 0 and not pos.partial_done:
            be_sl = (entry - pos.r1_be_buffer_atr * pos.atr_entry) if direction == "long" \
                else (entry + pos.r1_be_buffer_atr * pos.atr_entry)
            r1_threshold = pos.sl_dist
            r1_price_long = entry + r1_threshold
            r1_price_short = entry - r1_threshold
            if direction == "long" and bar_high >= r1_price_long:
                pos.trail_stop = be_sl
                pos.partial_done = True
            elif direction == "short" and bar_low <= r1_price_short:
                pos.trail_stop = be_sl
                pos.partial_done = True

        # +1.5R：减仓 r15_partial_pct
        if is_ming_style and pos.atr_entry > 0 and 1 not in pos.partials_filled:
            r15_price_long = entry + 1.5 * pos.sl_dist
            r15_price_short = entry - 1.5 * pos.sl_dist
            hit_15 = (
                (direction == "long" and bar_high >= r15_price_long) or
                (direction == "short" and bar_low <= r15_price_short)
            )
            if hit_15:
                pos.partials_filled.append(1)
                fill = max(r15_price_long, bar_open) if direction == "long" else min(r15_price_short, bar_open)
                return Signal(
                    action="partial_close", direction=direction,
                    confidence=0.6, strategy=pos.mode,
                    reason=f"r1.5 partial {pos.r15_partial_pct:.0%} @{fill:.4f}",
                    close_ratio=float(pos.r15_partial_pct),
                    fill_price=fill,
                )

        # +2R：启动 Chandelier（HH_N - k*ATR），SL 只能向有利方向移动
        if is_ming_style and pos.atr_entry > 0 and pos.mfe_r >= 2.0 and not pos.chandelier_on:
            pos.chandelier_on = True
            if direction == "long":
                pos.chandelier_high = bar_high
                ce = pos.chandelier_high - pos.r2_chandelier_k * pos.atr_entry
                pos.trail_stop = max(pos.trail_stop or 0.0, ce) if pos.trail_stop else ce
            else:
                pos.chandelier_low = bar_low
                ce = pos.chandelier_low + pos.r2_chandelier_k * pos.atr_entry
                pos.trail_stop = min(pos.trail_stop or float("inf"), ce) if pos.trail_stop else ce

        # Chandelier 跟踪（每根 K 线更新）
        if is_ming_style and pos.chandelier_on and pos.atr_entry > 0:
            if direction == "long":
                pos.chandelier_high = max(pos.chandelier_high, bar_high)
                ce = pos.chandelier_high - pos.r2_chandelier_k * pos.atr_entry
                pos.trail_stop = ce if pos.trail_stop is None else max(pos.trail_stop, ce)
                if pos.trail_stop is not None and bar_low <= pos.trail_stop:
                    fill = min(pos.trail_stop, bar_open) if bar_open > 0 else pos.trail_stop
                    return self._close_signal(
                        inst_id, pos,
                        f"chandelier low={bar_low:.4f}<={pos.trail_stop:.4f}",
                        fill_price=fill,
                    )
            else:
                pos.chandelier_low = min(pos.chandelier_low, bar_low)
                ce = pos.chandelier_low + pos.r2_chandelier_k * pos.atr_entry
                pos.trail_stop = ce if pos.trail_stop is None else min(pos.trail_stop, ce)
                if pos.trail_stop is not None and bar_high >= pos.trail_stop:
                    fill = max(pos.trail_stop, bar_open) if bar_open > 0 else pos.trail_stop
                    return self._close_signal(
                        inst_id, pos,
                        f"chandelier high={bar_high:.4f}>={pos.trail_stop:.4f}",
                        fill_price=fill,
                    )

        # 止损（intrabar 保守：bar_low/bar_high 触及即触发）
        if direction == "long":
            sl_price = entry - sl_dist
            if pos.trail_stop is not None:
                sl_price = max(sl_price, pos.trail_stop)
            if bar_low <= sl_price:
                fill = min(sl_price, bar_open) if bar_open > 0 else sl_price
                return self._close_signal(inst_id, pos,
                                          f"SL long low={bar_low:.4f}<={sl_price:.4f}",
                                          fill_price=fill)
        else:
            sl_price = entry + sl_dist
            if pos.trail_stop is not None:
                sl_price = min(sl_price, pos.trail_stop)
            if bar_high >= sl_price:
                fill = max(sl_price, bar_open) if bar_open > 0 else sl_price
                return self._close_signal(inst_id, pos,
                                          f"SL short high={bar_high:.4f}>={sl_price:.4f}",
                                          fill_price=fill)

        # 分批止盈（档位冻结）
        # ming 等"自管退出"策略完全走上面的状态机，不走这里的 rr_list 减仓路径
        batch = pos.tp_batch
        if batch < len(rr_list) and not is_ming_style:
            tp_price = (entry + sl_dist * rr_list[batch]) if direction == "long" \
                else (entry - sl_dist * rr_list[batch])
            hit = (bar_high >= tp_price) if direction == "long" else (bar_low <= tp_price)
            if hit:
                if batch == len(rr_list) - 1:
                    fill = max(tp_price, bar_open) if direction == "long" and bar_open > 0 \
                        else (min(tp_price, bar_open) if bar_open > 0 else tp_price)
                    return self._close_signal(inst_id, pos,
                                              f"TP{batch+1} 清剩余仓 @{tp_price:.4f}",
                                              fill_price=fill)
                ratio = batch_ratios[batch]
                fill = max(tp_price, bar_open) if direction == "long" and bar_open > 0 \
                    else (min(tp_price, bar_open) if bar_open > 0 else tp_price)
                return Signal(
                    action="partial_close", direction=direction,
                    confidence=0.5, strategy=pos.mode,
                    reason=f"TP{batch+1} @{tp_price:.4f}",
                    close_ratio=ratio,
                    fill_price=fill,
                )

        # MR RSI 归位退出
        if pos.mode == "mr":
            rsi_val = float(rsi(df["close"], 14).iloc[-1]) if len(df) >= 20 else 50
            pnl_pct = (last - entry) / entry if direction == "long" else (entry - last) / entry
            if direction == "long" and rsi_val > 60 and pnl_pct > 0:
                return self._close_signal(inst_id, pos, f"MR RSI normalize rsi={rsi_val:.1f}")
            if direction == "short" and rsi_val < 40 and pnl_pct > 0:
                return self._close_signal(inst_id, pos, f"MR RSI normalize rsi={rsi_val:.1f}")

        # RADX 退出：RSI(2) < ADX(2)
        if pos.mode == "radx" and direction == "long" and len(df) >= 20:
            rsi2 = float(rsi(df["close"], 2).iloc[-1])
            adx2 = float(adx(df, 2).iloc[-1])
            if pd.notna(rsi2) and pd.notna(adx2) and rsi2 < adx2:
                return self._close_signal(inst_id, pos, f"RADX exit RSI2<{adx2:.1f} rsi={rsi2:.1f}")

        # Trailing update（非 ming 模式用旧公式：partial_done 后按 close 中线收紧）
        if pos.partial_done and not is_ming_style:
            trail = entry + (last - entry) * 0.5 if direction == "long" else entry - (entry - last) * 0.5
            if pos.trail_stop is None:
                pos.trail_stop = trail
            elif direction == "long":
                pos.trail_stop = max(pos.trail_stop, trail)
            else:
                pos.trail_stop = min(pos.trail_stop, trail)

        return Signal(action="hold", direction=direction, reason=f"holding {pos.mode}")

    def _tighten_trail(self, pos: Position, direction: str, entry: float) -> None:
        """时间衰减：把止损向盈亏平衡（entry）方向收紧一档（不超过 entry）。"""
        step = max(pos.sl_dist * self.decay_step_frac, entry * 1e-6)
        if direction == "long":
            init_sl = entry - (pos.sl_dist if pos.sl_dist > 0 else entry * 0.01)
            cur = pos.trail_stop if pos.trail_stop is not None else init_sl
            pos.trail_stop = min(entry, cur + step)
        else:
            init_sl = entry + (pos.sl_dist if pos.sl_dist > 0 else entry * 0.01)
            cur = pos.trail_stop if pos.trail_stop is not None else init_sl
            pos.trail_stop = max(entry, cur - step)

    def _close_signal(self, inst_id: str, pos: Position, reason: str,
                      fill_price: float = 0.0) -> Signal:
        """返回平仓信号（不在此处删除持仓；由回测 confirm_close 处理）。"""
        return Signal(
            action="close", direction=pos.direction,
            confidence=0.0, strategy=pos.mode, reason=reason,
            fill_price=fill_price,
        )

    def confirm_fill(self, inst_id: str, fill_px: float, fill_sz: float,
                     signal: Signal) -> None:
        """开仓真实成交后登记/累加持仓，并冻结风险参数。

        部分成交按实际 fill_sz 记，避免登记幻影仓位。
        """
        if fill_sz <= 0 or signal is None:
            return
        direction = signal.direction
        if direction not in ("long", "short"):
            return

        sl_dist = 0.0
        if signal.stop_loss > 0:
            sl_dist = abs(fill_px - signal.stop_loss)
        if sl_dist <= 0:
            sl_dist = max(signal.atr * self.stop_loss_mult, fill_px * 0.005)

        rr_list = list(signal.rr_list) if signal.rr_list else []
        if not rr_list and signal.tp_batches and sl_dist > 0:
            rr_list = [abs(tp - fill_px) / sl_dist for tp in signal.tp_batches]
        if not rr_list:
            rr_list = list(_DEFAULT_RR)
        batch_ratios = list(signal.batch_ratios) if signal.batch_ratios else list(_DEFAULT_BATCH)
        if len(batch_ratios) != len(rr_list):
            batch_ratios = list(_DEFAULT_BATCH)

        pos = self.positions.get(inst_id, Position())
        if direction == "long":
            new_sz = pos.long_sz + fill_sz
            if pos.long_sz > 0:
                pos.entry_long = (pos.entry_long * pos.long_sz + fill_px * fill_sz) / new_sz
            else:
                pos.entry_long = fill_px
            pos.long_sz = new_sz
        else:
            new_sz = pos.short_sz + fill_sz
            if pos.short_sz > 0:
                pos.entry_short = (pos.entry_short * pos.short_sz + fill_px * fill_sz) / new_sz
            else:
                pos.entry_short = fill_px
            pos.short_sz = new_sz

        pos.mode = signal.strategy
        pos.direction = direction
        pos.open_ts = self._now()
        pos.sl_dist = sl_dist
        pos.rr_list = rr_list
        pos.batch_ratios = batch_ratios
        pos.atr_entry = signal.atr
        pos.tp_batch = 0
        pos.trail_stop = None
        pos.partial_done = False
        pos.last_decay_ts = 0.0
        pos.partials_filled = []
        pos.mfe_r = 0.0
        pos.chandelier_on = False
        pos.chandelier_high = fill_px if direction == "long" else 0.0
        pos.chandelier_low = fill_px if direction == "short" else 0.0
        pos.bars_held = 0
        pos.entry_ref_high = float(getattr(signal, "entry_reference_high", 0.0) or 0.0)
        pos.entry_ref_low = float(getattr(signal, "entry_reference_low", 0.0) or 0.0)
        pos.mfe_window_bars = int(getattr(signal, "mfe_window_bars", 0) or 0)
        pos.mfe_min_r = float(getattr(signal, "mfe_min_r", 0.0) or 0.0)
        pos.breakout_window_bars = int(getattr(signal, "breakout_window_bars", 0) or 0)
        pos.max_bars = int(getattr(signal, "max_bars", 0) or 0)
        pos.panic_gap_atr = float(getattr(signal, "panic_gap_atr", 0.0) or 0.0)
        # 策略显式发的 +1R/+1.5R/+2R 参数优先于 dataclass 默认值
        _sig_r1 = float(getattr(signal, "r1_be_buffer_atr", 0.0) or 0.0)
        if _sig_r1 > 0:
            pos.r1_be_buffer_atr = _sig_r1
        _sig_r15 = float(getattr(signal, "r15_partial_pct", 0.0) or 0.0)
        if _sig_r15 > 0:
            pos.r15_partial_pct = _sig_r15
        _sig_r2n = int(getattr(signal, "r2_chandelier_N", 0) or 0)
        if _sig_r2n > 0:
            pos.r2_chandelier_N = _sig_r2n
        _sig_r2k = float(getattr(signal, "r2_chandelier_k", 0.0) or 0.0)
        if _sig_r2k > 0:
            pos.r2_chandelier_k = _sig_r2k
        self.positions[inst_id] = pos

    def confirm_partial_close(self, inst_id: str, filled_sz: float) -> None:
        """分批止盈真实成交后：扣减引擎内部仓位并推进档位。"""
        pos = self.positions.get(inst_id)
        if pos is None or filled_sz <= 0:
            return
        if pos.direction == "long":
            pos.long_sz = max(0.0, pos.long_sz - filled_sz)
        else:
            pos.short_sz = max(0.0, pos.short_sz - filled_sz)

        # 推进档位 + 锁 SL 到盈亏平衡（注意 short 用 entry_short，勿误用 entry_long）
        sl_dist = pos.sl_dist if pos.sl_dist > 0 else 1e-9
        if pos.direction == "long":
            pos.trail_stop = pos.entry_long if pos.tp_batch == 0 else pos.entry_long + sl_dist
        else:
            pos.trail_stop = pos.entry_short if pos.tp_batch == 0 else pos.entry_short - sl_dist
        pos.partial_done = True
        pos.tp_batch += 1

        if pos.long_sz <= 1e-12 and pos.short_sz <= 1e-12:
            self.positions.pop(inst_id, None)

    def confirm_close(self, inst_id: str) -> None:
        """平仓真实成交后清理引擎内部仓位。"""
        self.positions.pop(inst_id, None)

    def _in_cooldown(self, inst_id: str) -> bool:
        now = self._now()
        return now < self._cooldown_until.get(inst_id, 0) or now < self._cb_until.get(inst_id, 0)

    def _kelly_factor(self, inst_id: str, atr_pct: float) -> float:
        hist = self._pnl_history.get(inst_id, [])
        if len(hist) < 30:
            return 1.0
        wins = [v for v in hist if v > 0]
        losses = [v for v in hist if v < 0]
        if not wins or not losses:
            return 1.0
        wr = len(wins) / len(hist)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        b = avg_win / avg_loss
        f_full = (wr * b - (1 - wr)) / b
        if f_full <= 0:
            return 0.0
        return min(0.25 * f_full, 1.0)

    def _record_signal(self, inst_id: str, sig: Signal, ts: float) -> None:
        lst = self._recent_signals.setdefault(inst_id, [])
        lst.append({"ts": ts, "direction": sig.direction, "strategy": sig.strategy, "confidence": sig.confidence})
        cutoff = ts - 3600
        self._recent_signals[inst_id] = [x for x in lst if x["ts"] >= cutoff]

    def note_trade_result(self, inst_id: str, pnl: float) -> None:
        hist = self._pnl_history.setdefault(inst_id, [])
        hist.append(pnl)
        if len(hist) > 200:
            del hist[:-200]
        if pnl < 0:
            self._consec_loss[inst_id] = self._consec_loss.get(inst_id, 0) + 1
            self._cooldown_until[inst_id] = self._now() + 1800
            if self._consec_loss[inst_id] >= 3:
                self._cb_until[inst_id] = self._now() + 1800
                self._consec_loss[inst_id] = 0
        else:
            self._consec_loss[inst_id] = 0
