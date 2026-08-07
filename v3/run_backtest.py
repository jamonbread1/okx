# -*- coding: utf-8 -*-
"""回测入口 — 直接驱动 backtest.multi_engine。

用法:
  python -m v3.run_backtest --bar 1H
  python -m v3.run_backtest --bar 1H --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
  python -m v3.run_backtest --bar 1H --only vol
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import setup_logger

log = setup_logger("run_backtest")


def _pkg_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = "") -> dict:
    """加载唯一配置文件 config.yaml（项目根目录）。

    路径优先级：显式传入 > 项目根 config.yaml。相对路径先相对 cwd，
    不存在则相对项目根。
    """
    import yaml

    if not config_path:
        config_path = os.path.join(_pkg_root(), "config.yaml")
    elif not os.path.isabs(config_path):
        cand = config_path
        if not os.path.isfile(cand):
            cand = os.path.join(_pkg_root(), config_path)
        config_path = cand

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"找不到配置文件: {config_path}\n请在项目根目录放置 config.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误（需要 mapping）: {config_path}")

    cfg.setdefault("strategy", {})
    cfg.setdefault("risk", {})
    cfg.setdefault("universe", {})
    cfg["strategy"]["_engine"] = True
    cfg["_config_path"] = os.path.abspath(config_path)
    return cfg


# 周期 → 推荐策略集合（自动规划时使用）
# 注意: 旧 auto_plan 仍引用已删的 rng; 这里保留为兼容性入口, 但 --only 不会被覆盖
_ALL_BUILTIN_STRATEGIES = ["vol", "mr", "ewmac", "macd", "don"]


def plan_strategies_for_bar(bar: str, enabled: list[str]) -> list[str]:
    """按周期自动规划策略集合（仅在用户请求"全策略"时生效）。

    显式子集和 --only 不会被覆盖。
    """
    requested = [str(x).strip().lower() for x in (enabled or []) if str(x).strip()]
    if not requested:
        requested = list(_ALL_BUILTIN_STRATEGIES)
    req_set = set(requested)
    all_set = set(_ALL_BUILTIN_STRATEGIES)
    wants_all = bool(req_set & {"all", "__all__"}) or all_set.issubset(req_set)
    if not wants_all:
        return requested

    b = str(bar or "").strip()
    if b in ("1D",):
        return ["macd", "don"]
    if b in ("4H", "2H"):
        return ["vol", "ewmac", "macd", "don"]
    if b in ("1H",):
        return list(_ALL_BUILTIN_STRATEGIES)
    if b in ("30m", "15m"):
        return ["vol", "ewmac", "macd"]
    return ["vol"]


def apply_auto_strategy_plan(cfg: dict, bar: str) -> list[str]:
    st = cfg.setdefault("strategy", {})
    enabled = list(st.get("enabled_strategies") or [])
    if not bool(st.get("auto_plan_enabled", True)):
        return enabled
    planned = plan_strategies_for_bar(bar, enabled)
    if planned != [str(x).strip().lower() for x in enabled if str(x).strip()]:
        st["enabled_strategies"] = planned
        st["_auto_plan_bar"] = bar
        st["_auto_plan_original"] = enabled
    return list(st.get("enabled_strategies") or [])


def main():
    ap = argparse.ArgumentParser(description="回测入口（multi_engine + v3.engine 直连）")
    ap.add_argument("--config", default="", help="默认: 项目根目录 config.yaml")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--is-ratio", type=float, default=0.7)
    ap.add_argument("--bar", default="1H",
                    choices=["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"])
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--out-dir", default="data/backtest_results")
    ap.add_argument("--only", default="",
                    help="只启用一个策略插件（name，见 --list-strategies）")
    ap.add_argument("--list-strategies", action="store_true",
                    help="列出已发现的策略插件并退出")
    args = ap.parse_args()

    if args.list_strategies:
        from v3.strategies.registry import all_strategies, validate_class
        found = all_strategies()
        print("=== 已发现策略插件 ===")
        if not found:
            print("  (无)")
        for name, cls in sorted(found.items()):
            errs = validate_class(cls)
            flag = "OK" if not errs else ("INVALID: " + "; ".join(errs))
            reg = getattr(cls, "required_regime", "?")
            print(f"  {name:16s}  class={cls.__name__:24s}  regime={reg:6s}  [{flag}]")
        print(f"\n共 {len(found)} 个。自定义策略写法见项目根 README.md")
        return

    cfg = load_config(args.config)

    if args.only:
        only = args.only.strip().lower()
        from v3.strategies.registry import list_names
        valid = list_names()
        if only not in valid:
            log.error(f"无效策略 {only}，已发现: {valid}")
            sys.exit(1)
        cfg["strategy"]["enabled_strategies"] = [only]
        log.info(f"仅启用: {only}")

    if not args.only:
        planned = apply_auto_strategy_plan(cfg, args.bar)
        if cfg.get("strategy", {}).get("_auto_plan_original") is not None:
            log.info(
                f"自动策略规划 bar={args.bar}: "
                f"{cfg['strategy'].get('_auto_plan_original')} -> {planned}"
            )

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    from backtest.multi_engine import run_multi_from_parquet
    from backtest.metrics import check_overfitting

    start = args.start.strip() or None
    end = args.end.strip() or None

    log.info(f"配置: {cfg.get('_config_path', '')}")
    log.info(f"回测 bar={args.bar} IS={args.is_ratio:.0%} "
             f"strategies={cfg['strategy'].get('enabled_strategies', [])} "
             f"range={start or 'min'}→{end or 'max'}")

    is_m, oos_m, cmp = run_multi_from_parquet(
        cfg,
        symbols=symbols,
        is_ratio=args.is_ratio,
        main_bar=args.bar,
        out_dir=args.out_dir,
        start=start,
        end=end,
    )

    # 把 bars 字段换算成 IS/OOS 实际天数（用于 CAGR 衰减）
    _bar_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600,
                    "4H": 14400, "1D": 86400, "1W": 604800}
    _bar_sec = _bar_seconds.get(args.bar, 86400)
    _is_days = float(is_m.get("bars", 0) or 0) * _bar_sec / 86400.0
    _oos_days = float(oos_m.get("bars", 0) or 0) * _bar_sec / 86400.0
    if _is_days <= 0:
        _is_days = 365 * 4
    if _oos_days <= 0:
        _oos_days = 365 * 2

    overfit = check_overfitting(is_m, oos_m, is_days=_is_days, oos_days=_oos_days)

    print("\n========== v3 组合 样本内 70% ==========")
    for k, v in is_m.items():
        if k in ("per_symbol", "signal_stats", "attribution"):
            continue
        print(f"  {k}: {v}")
    if isinstance(is_m.get("attribution"), dict):
        print("  attribution(per_strategy):", json.dumps(is_m["attribution"].get("per_strategy", {}),
                                                         ensure_ascii=False))
        print("  attribution(per_direction):", is_m["attribution"].get("per_direction"))
    print("\n========== v3 组合 样本外 30% ==========")
    for k, v in oos_m.items():
        if k in ("per_symbol", "signal_stats", "attribution"):
            continue
        print(f"  {k}: {v}")
    if isinstance(oos_m.get("attribution"), dict):
        print("  attribution(per_strategy):", json.dumps(oos_m["attribution"].get("per_strategy", {}),
                                                         ensure_ascii=False))
        print("  attribution(per_direction):", oos_m["attribution"].get("per_direction"))
    print("\n", cmp.to_string(index=False))

    print("\n========== 过拟合检查 ==========")
    print(f"  ok: {overfit['ok']}")
    print(f"  overfit_score: {overfit['overfit_score']}  (0=稳, 1=严重)")
    print(f"  advice: {overfit['advice']}")
    for w in overfit.get("warnings") or []:
        print(f"  ! {w}")

    print(f"\n明细: {args.out_dir}/MULTI_*")


if __name__ == "__main__":
    main()
