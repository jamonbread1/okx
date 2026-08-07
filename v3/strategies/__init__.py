# -*- coding: utf-8 -*-
"""策略插件包。

自定义策略
----------
1. 在 ``v3/strategies/`` 新建 ``my_strategy.py``（不要以 ``_`` 开头）
2. 继承 ``StrategyBase``，设置唯一 ``name`` 并实现 ``generate``
3. 在 ``config.yaml`` → ``enabled_strategies`` 加入你的 ``name``
4. ``python -m v3.run_backtest --list-strategies`` 确认已发现

公共 API
--------
- Signal, StrategyBase
- register_strategy / register
- REGISTRY（动态代理，始终反映最新发现结果）
- list_strategies / get_strategy / reload_strategies
"""
from __future__ import annotations

from v3.strategies.base import Signal, StrategyBase
from v3.strategies.registry import (
    all_strategies,
    create,
    discover,
    get,
    list_names,
    register,
    register_strategy,
    reload,
    validate_class,
)

# 导入时自动发现 v3/strategies 下所有插件
discover()

# 显式触发子包模块加载（discover() 默认只扫顶层，
# 子包下的 .py 模块需要主动 import 才能注册）
import v3.strategies.don.don  # noqa: F401
import v3.strategies.ewmac.ewmac  # noqa: F401
import v3.strategies.macd.macd  # noqa: F401
import v3.strategies.ming.ming  # noqa: F401
import v3.strategies.mr.mr  # noqa: F401
import v3.strategies.radx.radx  # noqa: F401

# 再 discover 一次（让显式 import 后的注册生效）
discover()

# 兼容旧代码：REGISTRY 像 dict 一样用
class _RegistryProxy(dict):
    def __getitem__(self, key):
        ensure = get(key)
        if ensure is None:
            raise KeyError(key)
        return ensure

    def get(self, key, default=None):  # type: ignore[override]
        cls = get(key)
        return cls if cls is not None else default

    def keys(self):
        return list_names()

    def items(self):
        return all_strategies().items()

    def values(self):
        return all_strategies().values()

    def __contains__(self, key):
        return get(key) is not None

    def __iter__(self):
        return iter(list_names())

    def __len__(self):
        return len(list_names())


REGISTRY = _RegistryProxy()

# 便捷别名
list_strategies = list_names
get_strategy = get
reload_strategies = reload

__all__ = [
    "Signal",
    "StrategyBase",
    "REGISTRY",
    "register",
    "register_strategy",
    "discover",
    "reload",
    "reload_strategies",
    "list_names",
    "list_strategies",
    "get",
    "get_strategy",
    "create",
    "all_strategies",
    "validate_class",
]
