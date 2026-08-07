# -*- coding: utf-8 -*-
"""策略参数加载器。

config.yaml 不再保存任何策略专属参数（don_xxx / vol_xxx / mr_xxx / ...），
每个策略的参数全部归入该策略的目录里::

    v3/strategies/<name>/
        <name>.py     # 策略实现
        params.yaml   # 默认参数（裸 key，无前缀）
        README.md     # 参数说明

加载流程:
  1. 从 config.yaml → strategy: 段读 enabled_strategies
  2. 对每个启用的策略，定位 v3/strategies/<name>/params.yaml
  3. 把 strategy 段下的跨策略共享字段按 SHARED_KEYS 白名单透传给策略
  4. 合并 --extra-config FILE 覆盖
  5. 合并 --params NAME=KEY=VAL 单点覆盖

错误语义:
  - 启用的策略在 v3/strategies/ 下既没有 <name>/<name>.py 也没有 <name>.py
    → 启动时报错
  - <name>/params.yaml 存在但 YAML 解析失败 → 报错
  - config.yaml → strategy: 下出现"策略专属前缀"字段 → 启动时报错，
    提示用户已迁到 v3/strategies/<name>/params.yaml
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None


def strategies_root() -> str:
    """v3/strategies/ 的真实目录（loader 和 registry 共用）。"""
    return os.path.dirname(os.path.abspath(__file__))


# 策略专属前缀门禁：config.yaml 残留前缀时启动报错
_STRATEGY_OWNED_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "don": (
        "don_period", "don_min_adx", "don_vol_ratio", "don_max_wick_ratio",
        "don_position_pct", "don_sl_atr_mult", "don_tp_atr_mult",
        "don_slippage_pct", "don_fee_rt", "don_short_penalty", "don_calc_window",
    ),
    "vol": (
        "vol_bb_period", "vol_bbw_squeeze", "vol_volume_mult",
        "vol_rsi_low", "vol_rsi_high", "vol_breakout_buffer",
        "vol_position_pct", "vol_sl_atr_mult", "vol_tp_atr_mult",
        "vol_min_expectancy", "vol_slippage_pct", "vol_fee_rt",
        "vol_short_penalty", "vol_squeeze_hold",
    ),
    "mr": (
        "mr_bb_period", "mr_rsi_period", "mr_rsi_oversold", "mr_rsi_overbought",
        "mr_max_adx", "mr_ema_filter", "mr_ema_atr_mult",
        "mr_position_pct", "mr_sl_atr_mult", "mr_tp_atr_mult",
        "mr_min_expectancy", "mr_slippage_pct", "mr_fee_rt",
    ),
    "ewmac": (
        "ewmac_fast_1", "ewmac_slow_1", "ewmac_fast_2", "ewmac_slow_2",
        "ewmac_threshold", "ewmac_min_adx",
        "ewmac_position_pct", "ewmac_sl_atr_mult", "ewmac_tp_atr_mult",
        "ewmac_slippage_pct", "ewmac_fee_rt",
    ),
    "macd": (
        "macd_fast", "macd_slow", "macd_signal",
        "macd_rsi_low", "macd_rsi_high", "macd_min_adx",
        "macd_position_pct", "macd_sl_atr_mult", "macd_tp_atr_mult",
        "macd_slippage_pct", "macd_fee_rt",
    ),
    "radx": (
        "radx_sma_period", "radx_ema_period", "radx_rsi_period", "radx_adx_period",
        "radx_position_pct", "radx_sl_atr_mult", "radx_tp_r",
        "radx_min_confidence", "radx_min_atr_pct", "radx_max_atr_pct",
        "radx_fee_rt", "radx_slippage_pct", "radx_long_only",
    ),
    "ming": (),  # ming 全部用裸 key，不允许任何 ming_xxx 前缀
}

_TOP_LEVEL_OWNED_PREFIXES = tuple(_STRATEGY_OWNED_PREFIXES.keys())


# 跨策略共享字段白名单：留在 config.yaml → strategy: 段下，不迁到 params.yaml
SHARED_STRATEGY_KEYS: frozenset = frozenset({
    "name", "signal_bar", "min_open_confidence", "bt_exec_at_open",
    "signal_on_closed_bar", "bt_embargo_bars",
    "execution_order_type", "aggressive_limit_ticks",
    "aggressive_limit_tick_fallback", "enable_ob_fill", "bt_slippage_stress",
    "regime_adx_trend", "regime_adx_exit", "regime_adx_chop", "regime_squeeze_threshold",
    "sizing_mode", "target_daily_vol",
    "htf_enabled", "htf_bar", "htf_ema_fast", "htf_ema_slow",
    "htf_min_gap", "htf_lookback", "signal_lookback",
    "conflict_window_min",
    "stop_loss_atr_mult", "take_profit_atr_mult", "trail_atr_mult",
    "position_timeout_sec", "exit_mode",
    "time_decay_step_hours", "time_decay_step_frac",
    "funding_settlement_hours", "funding_tilt_enabled",
    "funding_tilt_conf", "funding_max_abs",
    "net_delta_enabled", "net_delta_limit_mult",
    "_engine", "_config_path", "_main_bar", "main_bar",
    "enabled_strategies", "auto_plan_enabled",
    "_auto_plan_bar", "_auto_plan_original",
})


def strategy_params_path(name: str, root: Optional[str] = None) -> str:
    """返回 <root>/<name>/params.yaml 的绝对路径。"""
    root = root or strategies_root()
    return os.path.join(root, name, "params.yaml")


def strategy_module_path(name: str, root: Optional[str] = None) -> Optional[str]:
    """如果存在 <root>/<name>/<name>.py 返回其绝对路径，否则 None。"""
    root = root or strategies_root()
    p = os.path.join(root, name, f"{name}.py")
    return p if os.path.isfile(p) else None


def load_strategy_params(name: str, root: Optional[str] = None) -> Dict[str, Any]:
    """读取 v3/strategies/<name>/params.yaml，缺失返回空 dict。"""
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取策略 params.yaml。请 pip install pyyaml")
    p = strategy_params_path(name, root=root)
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} 必须是 mapping（key/value），收到: {type(data).__name__}")
    return data


def assert_no_legacy_keys(cfg: dict, *, where: str = "config.yaml") -> List[str]:
    """检查 cfg 里是否还残留旧的前缀字段（应已迁到 params.yaml）。

    返回冲突列表。启动时由 v3.run_backtest.load_config 调一次，
    发现冲突立即 raise。
    """
    conflicts: List[str] = []
    strat = (cfg.get("strategy") or {}) if isinstance(cfg, dict) else {}
    for owner, keys in _STRATEGY_OWNED_PREFIXES.items():
        for k in keys:
            if k in strat:
                conflicts.append(
                    f"{where}.strategy.{k}  (belongs to strategy '{owner}', "
                    f"moved to v3/strategies/{owner}/params.yaml)"
                )
    if isinstance(cfg, dict):
        for k in cfg.keys():
            for prefix in _TOP_LEVEL_OWNED_PREFIXES:
                if k.startswith(f"{prefix}_"):
                    conflicts.append(
                        f"{where}.{k}  (top-level strategy-owned key, "
                        f"moved to v3/strategies/{prefix}/params.yaml)"
                    )
                    break
    return conflicts


def _normalize_overrides(items: Iterable[str]) -> List[Tuple[str, str, str]]:
    """解析 --params NAME=KEY=VAL 形式（也接受 NAME:KEY=VAL）。"""
    out: List[Tuple[str, str, str]] = []
    for raw in items or ():
        s = str(raw).strip()
        if not s:
            continue
        # 归一化: 形如 "name:key=val" 转换成 "name=key=val"
        if "=" in s:
            idx = s.index("=")
            lhs = s[:idx]
            rhs = s[idx + 1:]
            if ":" in lhs:
                name, key = lhs.split(":", 1)
                s = f"{name}={key}={rhs}"
        parts = s.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"--params 参数格式应为 NAME=KEY=VAL 或 NAME:KEY=VAL，收到: {raw!r}")
        name, key, val = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not name or not key:
            raise ValueError(f"--params 名称/键不能为空: {raw!r}")
        out.append((name, key, val))
    return out


def _coerce_scalar(val: str) -> Any:
    """把字符串 VAL 解析为 int / float / bool / str / None 之一。"""
    s = val.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def build_strategy_cfg(
    name: str,
    base_strategy_cfg: Dict[str, Any],
    *,
    extra_config_path: str = "",
    cli_overrides: Optional[Iterable[str]] = None,
    root: Optional[str] = None,
    warn: bool = True,
) -> Dict[str, Any]:
    """为某个策略构造 cfg dict。

    合并顺序（后者覆盖前者）:
      1. v3/strategies/<name>/params.yaml — 默认值
      2. config.yaml → strategy: 段（共享字段）— 跨策略共用
      3. --extra-config FILE 顶层覆盖
      4. --params name=key=val 单点覆盖

    warn=True 时：--extra-config / --params 拼错名字或 key 时打 stderr warning
    （不强 raise，兼容嵌入调用；CLI 模式下用户能直接看到拼写错误）
    """
    import sys

    base = dict(base_strategy_cfg or {})
    params = load_strategy_params(name, root=root)
    merged: Dict[str, Any] = {**params, **base}

    if extra_config_path:
        if yaml is None:
            raise RuntimeError("缺少 PyYAML，无法读取 --extra-config")
        if not os.path.isfile(extra_config_path):
            raise FileNotFoundError(f"--extra-config 指定的文件不存在: {extra_config_path}")
        with open(extra_config_path, "r", encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}
        if not isinstance(extra, dict):
            raise ValueError(f"--extra-config 必须是 mapping: {extra_config_path}")
        known = set(params.keys()) | set(base.keys()) | set(SHARED_STRATEGY_KEYS)
        unknown = set(extra.keys()) - known
        if unknown and warn:
            print(
                f"[loader] WARNING: --extra-config 含未在 {name} 的 params.yaml / strategy 段 / 共享字段里登记的 "
                f"key {len(unknown)} 个，例如: {sorted(unknown)[:3]}",
                file=sys.stderr,
            )
        merged.update(extra)

    for sname, key, val in _normalize_overrides(cli_overrides):
        if sname != name:
            continue
        if warn and key not in params and key not in base:
            print(
                f"[loader] WARNING: --params {sname}:{key}={val} 的 key 不在"
                f" {name}/params.yaml 也不在 strategy 段，已强行写入（可能拼错）",
                file=sys.stderr,
            )
        merged[key] = _coerce_scalar(val)

    return merged


def all_strategy_param_keys() -> Dict[str, Tuple[str, ...]]:
    """返回 {name: (key, key, ...)}，仅用于文档/测试检查。"""
    return {k: tuple(v) for k, v in _STRATEGY_OWNED_PREFIXES.items()}


# 兼容旧 key（仅在 YAML 完全没有目标 key 时回退）— 留空供未来"旧名 → 新名"兼容入口
LEGACY_FALLBACK: Dict[str, Dict[str, str]] = {}


__all__ = [
    "strategies_root",
    "strategy_params_path",
    "strategy_module_path",
    "load_strategy_params",
    "assert_no_legacy_keys",
    "build_strategy_cfg",
    "all_strategy_param_keys",
    "SHARED_STRATEGY_KEYS",
    "LEGACY_FALLBACK",
]
