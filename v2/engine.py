# -*- coding: utf-8 -*-
"""v2 策略引擎 — 组合所有策略，统一信号生成 + 持仓管理。

架构:
  StrategyEngine
    ├── RegimeDetector    → 检测当前 regime
    ├── HtfFilter         → 高周期趋势过滤（硬挡）
    ├── ConflictFilter    → 跨策略冲突过滤
    ├── Strategy plugins  → 自动发现 v2/strategies/*（见 registry）
    └── PositionManager   → 持仓管理 + 分批止盈

与 v1 / 旧 v2 的关键区别:
  1. 策略是独立模块，通过注册表加载
  2. Regime 决定哪些策略可开仓（trend 策略 vs chop 策略互斥）
  3. HTF 过滤是硬挡（v18.1 证明软挡不可行）——过滤后无信号直接 hold
  4. 指标计算在 v2.indicators 中，策略不重复实现
  5. 信号周期默认 1H（v18.2 证明 15m 无真实边）
  6. **成交确认**：generate_signal 只产出信号，不登记持仓；
     由外层回测在真实成交后调用 confirm_fill / confirm_partial_close / confirm_close。
     避免「OB 拒单 / 部分成交后引擎登记幻影仓位」。
  7. **止损/止盈档位在开仓时冻结**：sl_dist 与 RR 档位存入 Position，
     不再每根 bar 用当前 ATR 重算（波动扩张时止损自动变宽的 bug）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from v2.indicators import adx, rsi
from v2.risk import RegimeDetector, HtfFilter, ConflictFilter
from v2.strategies import REGISTRY, Signal, StrategyBase

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
    # —— 开仓时冻结的风险参数（P0-4）——
    sl_dist: float = 0.0                                   # 止损距离（价格单位）
    rr_list: List[float] = field(default_factory=lambda: list(_DEFAULT_RR))
    batch_ratios: List[float] = field(default_factory=lambda: list(_DEFAULT_BATCH))
    atr_entry: float = 0.0                                 # 入场 ATR
    entry_bar_ts: float = 0.0                              # 入场 K 线时间
    last_decay_ts: float = 0.0                             # 上次 time_decay 收紧时间


class StrategyEngine:
    """v2 策略引擎。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.strat_cfg = cfg.get("strategy", {})

        # Regime
        self.regime = RegimeDetector(self.strat_cfg)
        # HTF filter
        self.htf = HtfFilter(self.strat_cfg)
        # Conflict filter
        self.conflict = ConflictFilter(self.strat_cfg)

        # 加载策略插件（自动发现 v2/strategies 下所有实现）
        from v2.strategies.registry import all_strategies, list_names
        available = all_strategies()
        enabled = list(self.strat_cfg.get("enabled_strategies") or ["vol", "mr", "rng", "ewmac"])
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
                self.strategies[name] = cls(self.strat_cfg)
            except Exception as e:
                print(f"[StrategyEngine] 策略 {name} 实例化失败: {e}")
        if missing:
            print(
                f"[StrategyEngine] 未找到策略 {missing}；"
                f"已发现: {list_names()}"
            )
        if not self.strategies:
            print(
                f"[StrategyEngine] 警告: 无任何已启用策略被加载。"
                f"enabled={enabled} discovered={list_names()}"
            )

        # 持仓
        self.positions: Dict[str, Position] = {}
        # 冷却/熔断
        self._cooldown_until: Dict[str, float] = {}
        self._consec_loss: Dict[str, int] = {}
        self._cb_until: Dict[str, float] = {}
        # 冲突记录
        self._recent_signals: Dict[str, List[Dict]] = {}
        # 信号周期
        self.signal_bar = str(self.strat_cfg.get("signal_bar", "1H"))
        # Kelly 历史
        self._pnl_history: Dict[str, List[float]] = {}
        # 模拟时间
        self._sim_ts: Optional[float] = None

        # —— 配置开关 ——
        self.min_conf = float(self.strat_cfg.get("min_open_confidence", 0.55))
        self.stop_loss_mult = float(self.strat_cfg.get("stop_loss_atr_mult", 1.8))
        self.trail_mult = float(self.strat_cfg.get("trail_atr_mult", 2.0))
        self.timeout_sec = float(self.strat_cfg.get("position_timeout_sec", 86400))
        self.exit_mode = str(self.strat_cfg.get("exit_mode", "timeout"))  # timeout | time_decay
        self.decay_step_hours = float(self.strat_cfg.get("time_decay_step_hours", 6.0))
        self.decay_step_frac = float(self.strat_cfg.get("time_decay_step_frac", 0.25))
        # 资金费率 tilt（P2-3）：向付费方向收紧入场置信度
        self.funding_tilt = bool(self.strat_cfg.get("funding_tilt_enabled", True))
        self.funding_tilt_conf = float(self.strat_cfg.get("funding_tilt_conf", 0.05))

        enabled_names = list(self.strategies.keys())
        log_msg = f"StrategyEngine v2 | strategies={enabled_names} | bar={self.signal_bar}"
        print(log_msg)

    def set_sim_time(self, ts: float) -> None:
        self._sim_ts = ts

    def _now(self) -> float:
        return self._sim_ts if self._sim_ts is not None else time.time()

    def _timeout_sec_for_position(self, mode: str) -> float:
        """Return per-strategy timeout seconds, falling back to global timeout."""
        timeout_sec = float(self.timeout_sec)
        mode = str(mode or "").strip()
        if not mode:
            return timeout_sec

        bars = self.strat_cfg.get(f"{mode}_timeout_bars")
        if bars is None and mode == "rng":
            # 兼容 range_* 参数前缀。
            bars = self.strat_cfg.get("range_timeout_bars")
        if bars is not None and str(bars).strip() != "":
            try:
                n_bars = float(bars)
            except (TypeError, ValueError):
                n_bars = 0.0
            if n_bars > 0:
                bar = self.strat_cfg.get("_main_bar") or self.strat_cfg.get("main_bar") or self.signal_bar
                return n_bars * _bar_to_seconds(str(bar))

        sec = self.strat_cfg.get(f"{mode}_timeout_sec")
        if sec is None and mode == "rng":
            sec = self.strat_cfg.get("range_timeout_sec")
        if sec is not None and str(sec).strip() != "":
            try:
                value = float(sec)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value

        return timeout_sec

    # ---- 信号生成 ----
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

        **不登记持仓**：返回 open 信号后由回测层真实成交并调用
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

        # 冷却/熔断
        if self._in_cooldown(inst_id):
            return Signal(action="hold", direction="flat", reason="cooldown")

        # Regime
        regime = self.regime.detect(df, last)

        # HTF 过滤
        htf_bias = self.htf.bias(df_htf)

        # Kelly
        kelly = self._kelly_factor(inst_id, regime.atr_pct)

        # 候选信号
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

        # HTF 硬挡（P0-2）：过滤后无同向候选直接返回 hold。
        # 策略可设 skip_htf=True 以豁免。
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

        # 排序：置信度最高的优先
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

        # 冲突过滤
        now = self._now()
        conflict = self.conflict.check(
            best.direction, best.strategy, best.confidence,
            self._recent_signals.get(inst_id, []), now,
        )
        if conflict:
            return Signal(action="hold", direction="flat", reason=conflict)

        # 记录
        self._record_signal(inst_id, best, now)

        # 注意：不在此处登记 self.positions —— 等回测层 confirm_fill
        return best

    # ---- 持仓管理 ----
    def _manage_position(
        self, inst_id: str, pos: Position, df: pd.DataFrame,
        last: float, specs: dict,
    ) -> Signal:
        """持仓管理: 止损 / 分批止盈 / trailing / 超时(或时间衰减) / 兼容旧 MR RSI 归位。

        止损距离与止盈档位在开仓时冻结（pos.sl_dist / pos.rr_list），
        不随当前 ATR 漂移。用最后一根已收盘 bar 的 high/low 做 intrabar
        保守触发：同根 bar 同时触及 SL 与 TP 时假设先触 SL。
        """
        direction = pos.direction
        entry = pos.entry_long if direction == "long" else pos.entry_short

        if entry <= 0:
            return Signal(action="hold", direction=direction, reason="invalid entry")

        # 冻结的止损距离（回退到当前 ATR 仅用于历史遗留 Position）
        sl_dist = pos.sl_dist if pos.sl_dist > 0 else last * 0.01
        rr_list = pos.rr_list or _DEFAULT_RR
        batch_ratios = pos.batch_ratios or _DEFAULT_BATCH

        now = self._now()
        elapsed = max(0.0, now - float(pos.open_ts or 0.0))

        # 当前 bar OHLC（df 最后一行 = 已收盘信号 bar）
        row = df.iloc[-1]
        bar_open = float(row.get("open", last) or last)
        bar_high = float(row.get("high", last) or last)
        bar_low = float(row.get("low", last) or last)

        # —— 退出模式：硬超时 或 时间衰减止损（P2-5）——
        # time_decay：首次越过超时立即收紧一档，之后每 decay_step_hours 再收紧，
        # 避免每根 bar 连续收紧。
        # 可按策略设置 {name}_timeout_bars/sec；未设置则使用 position_timeout_sec。
        timeout_sec = self._timeout_sec_for_position(pos.mode)
        if elapsed >= timeout_sec - 1e-9:
            if self.exit_mode == "time_decay":
                step_sec = max(60.0, float(self.decay_step_hours) * 3600.0)
                last_d = float(pos.last_decay_ts or 0.0)
                if last_d <= 0 or (now - last_d >= step_sec - 1e-9):
                    self._tighten_trail(pos, direction, entry)
                    pos.last_decay_ts = now
            else:
                # timeout 在当前评估时点执行，交给回测层使用当前 bar open，
                # 不复用上一根已收盘 K 线的 open。
                return self._close_signal(inst_id, pos, f"timeout {elapsed:.0f}s")

        # —— 止损（intrabar 保守：bar_low/bar_high 触及即触发）——
        if direction == "long":
            sl_price = entry - sl_dist
            if pos.trail_stop is not None:
                sl_price = max(sl_price, pos.trail_stop)
            if bar_low <= sl_price:
                # 保守成交价：min(sl_price, bar_open)（跳空按更差价）
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

        # —— 分批止盈（档位冻结）——
        batch = pos.tp_batch
        if batch < len(rr_list):
            tp_price = (entry + sl_dist * rr_list[batch]) if direction == "long" \
                else (entry - sl_dist * rr_list[batch])
            hit = (bar_high >= tp_price) if direction == "long" else (bar_low <= tp_price)
            if hit:
                # 同根 bar 同时触及 SL/TP 时，SL 已在上面提前返回 → 此处只处理 TP
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

        # RADX 公开规则退出：RSI(2) < ADX(2)
        if pos.mode == "radx" and direction == "long" and len(df) >= 20:
            rsi2 = float(rsi(df["close"], 2).iloc[-1])
            adx2 = float(adx(df, 2).iloc[-1])
            if pd.notna(rsi2) and pd.notna(adx2) and rsi2 < adx2:
                return self._close_signal(inst_id, pos, f"RADX exit RSI2<{adx2:.1f} rsi={rsi2:.1f}")

        # Trailing update
        if pos.partial_done:
            trail = entry + (last - entry) * 0.5 if direction == "long" else entry - (entry - last) * 0.5
            if pos.trail_stop is None:
                pos.trail_stop = trail
            elif direction == "long":
                pos.trail_stop = max(pos.trail_stop, trail)
            else:
                pos.trail_stop = min(pos.trail_stop, trail)

        return Signal(action="hold", direction=direction, reason=f"holding {pos.mode}")

    def _tighten_trail(self, pos: Position, direction: str, entry: float) -> None:
        """时间衰减：把止损向盈亏平衡（entry）方向收紧一档（P2-5）。

        多：止损在 entry 下方，收紧 = 上移（提高 trail_stop）
        空：止损在 entry 上方，收紧 = 下移（降低 trail_stop）
        不超过 entry（保本封顶）。
        """
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

    # ---- 成交确认（P0-3 / P0-5）----
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

        # 冻结止损距离（价格单位）与 RR 档位
        sl_dist = 0.0
        if signal.stop_loss > 0:
            sl_dist = abs(fill_px - signal.stop_loss)
        if sl_dist <= 0:
            sl_dist = max(signal.atr * self.stop_loss_mult, fill_px * 0.005)

        # 从 tp_batches(价格) 反推 RR 档位；无则用策略默认
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
        self.positions[inst_id] = pos

    def confirm_partial_close(self, inst_id: str, filled_sz: float) -> None:
        """分批止盈真实成交后：扣减引擎内部仓位并推进档位（P0-5）。"""
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
            # 首档后锁保本，后续档再放宽到 +1R
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

    # ---- 辅助 ----
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
