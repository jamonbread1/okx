# -*- coding: utf-8 -*-
"""精度处理、缓存、辅助函数"""

import json
import os
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Optional


def round_to_precision(value: float, precision: float, mode: str = "down") -> float:
    """按 tick / lot 精度取整，保证结果是 precision 的整数倍"""
    if precision <= 0:
        return value
    d_val = Decimal(str(value))
    d_prec = Decimal(str(precision))
    if mode == "down":
        # 先除再取整再乘，避免 quantize 对非 10 次幂精度的问题
        n = (d_val / d_prec).to_integral_value(rounding=ROUND_DOWN)
        return float(n * d_prec)
    n = (d_val / d_prec).to_integral_value(rounding=ROUND_HALF_UP)
    return float(n * d_prec)


def format_size(value: float, lot_sz: float) -> str:
    """把数量格式化成符合 lotSz 的字符串，避免科学计数法和多余小数"""
    rounded = round_to_precision(value, lot_sz, "down")
    if rounded <= 0:
        rounded = lot_sz
    # 根据 lot_sz 决定小数位
    d_lot = Decimal(str(lot_sz))
    decimals = max(0, -d_lot.as_tuple().exponent)
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(rounded)


def format_price(value: float, tick_sz: float) -> str:
    """价格格式化"""
    rounded = round_to_precision(value, tick_sz, "half")
    d_tick = Decimal(str(tick_sz))
    decimals = max(0, -d_tick.as_tuple().exponent)
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(rounded)


def make_cl_ord_id(prefix: str = "g") -> str:
    """生成符合 OKX 规范的 clOrdId：仅字母数字，最长 32 位"""
    import time
    import random
    # 只用 a-z 0-9
    ts = int(time.time() * 1000) % 100000000
    rnd = random.randint(1000, 9999)
    cid = f"{prefix}{ts}{rnd}"
    return cid[:32]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class SimpleCache:
    """简单内存 + 文件缓存，减少重复请求"""

    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 300):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)
        self._mem = {}

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        ttl = ttl if ttl is not None else self.default_ttl
        # 内存
        if key in self._mem:
            ts, data = self._mem[key]
            if time.time() - ts < ttl:
                return data
        # 文件
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if time.time() - obj.get("_ts", 0) < ttl:
                    self._mem[key] = (obj["_ts"], obj["data"])
                    return obj["data"]
            except Exception:
                pass
        return None

    def set(self, key: str, data: Any):
        ts = time.time()
        self._mem[key] = (ts, data)
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"_ts": ts, "data": data}, f, ensure_ascii=False)
        except Exception:
            pass
