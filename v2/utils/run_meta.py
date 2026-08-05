# -*- coding: utf-8 -*-
"""回测 run 元数据：run_id、config_hash、一致性校验。"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def config_hash(cfg: dict) -> str:
    """对策略相关配置做短哈希，便于对比两次回测是否同参。"""
    keys = ("strategy", "risk", "universe", "capital_usdt", "leverage")
    slim = {k: cfg.get(k) for k in keys if k in cfg}
    # 去掉运行时字段
    st = dict(slim.get("strategy") or {})
    st.pop("_v2_engine", None)
    slim["strategy"] = st
    h = hashlib.sha256(_stable_json(slim).encode("utf-8")).hexdigest()
    return h[:12]


def new_run_meta(
    cfg: dict,
    *,
    strategy: str = "",
    symbol: str = "",
    bar: str = "",
    phase: str = "",
    extra: Optional[dict] = None,
) -> Dict[str, Any]:
    meta = {
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg),
        "config_path": cfg.get("_config_path", ""),
        "strategy": strategy,
        "symbol": symbol,
        "bar": bar,
        "phase": phase,
        "capital_usdt": float(cfg.get("capital_usdt", 0) or 0),
        "enabled_strategies": list((cfg.get("strategy") or {}).get("enabled_strategies") or []),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_meta(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def assert_trades_match_summary(
    trades_pnl_sum: float,
    summary_total_pnl: float,
    trade_count: int,
    summary_count: int,
    tol: float = 0.05,
) -> None:
    if trade_count != summary_count:
        raise AssertionError(
            f"成交笔数不一致: trades={trade_count} summary={summary_count}"
        )
    if abs(trades_pnl_sum - summary_total_pnl) > tol:
        raise AssertionError(
            f"PnL 不一致: trades_sum={trades_pnl_sum:.4f} summary={summary_total_pnl:.4f}"
        )
