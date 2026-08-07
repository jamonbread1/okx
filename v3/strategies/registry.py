# -*- coding: utf-8 -*-
"""策略插件注册表与自动发现。

协议（写自定义策略时必须遵守）
================================

1. 在 ``v3/strategies/`` 下新建 ``你的文件.py``（文件名不要以 ``_`` 开头）。
2. 继承 ``StrategyBase``，设置类属性 ``name``（全局唯一、小写建议）。
3. 实现 ``generate(...)``，返回 ``Signal`` 或 ``None``。
4. （可选）用 ``@register_strategy`` 装饰；即使不写，导入时也会自动扫描注册。
5. 在 ``config.yaml`` 的 ``enabled_strategies`` 里写入 ``name`` 即可参与回测。

最小示例与字段说明见项目根 ``README.md``。

发现规则
--------
- 扫描 ``v3.strategies`` 包内所有非 ``_`` 开头、非 base/registry 的模块
- 收集其中 ``StrategyBase`` 的具体子类，且 ``name`` 非空、非 ``"base"``
- 同名后注册覆盖先注册（后加载的插件可覆盖内置）
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import traceback
from typing import Dict, List, Optional, Type

from v3.strategies.base import StrategyBase

log = logging.getLogger("strategy_registry")

# name -> class
_REGISTRY: Dict[str, Type[StrategyBase]] = {}
_DISCOVERED = False


def register(cls: Type[StrategyBase], *, overwrite: bool = True) -> Type[StrategyBase]:
    """注册一个策略类。可作为装饰器：@register 或 @register_strategy。"""
    if not isinstance(cls, type) or not issubclass(cls, StrategyBase):
        raise TypeError(f"策略必须继承 StrategyBase，收到: {cls!r}")
    name = str(getattr(cls, "name", "") or "").strip()
    if not name or name == "base":
        raise ValueError(f"策略类 {cls.__name__} 必须设置非空 name（且不能是 'base'）")
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"策略名冲突: {name} 已注册为 {_REGISTRY[name].__name__}")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        log.info("策略插件覆盖: %s  %s -> %s", name, _REGISTRY[name].__name__, cls.__name__)
    _REGISTRY[name] = cls
    return cls


# 装饰器别名
register_strategy = register


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def get(name: str) -> Optional[Type[StrategyBase]]:
    ensure_discovered()
    return _REGISTRY.get(name)


def all_strategies() -> Dict[str, Type[StrategyBase]]:
    ensure_discovered()
    return dict(_REGISTRY)


def list_names() -> List[str]:
    ensure_discovered()
    return sorted(_REGISTRY.keys())


def create(name: str, cfg: dict) -> StrategyBase:
    """按名称实例化策略。"""
    cls = get(name)
    if cls is None:
        known = ", ".join(list_names()) or "(无)"
        raise KeyError(f"未知策略 '{name}'。已发现: {known}")
    return cls(cfg)


def validate_class(cls: Type[StrategyBase]) -> List[str]:
    """静态检查策略类是否符合协议，返回错误列表（空=通过）。"""
    errs: List[str] = []
    if not isinstance(cls, type) or not issubclass(cls, StrategyBase):
        return ["不是 StrategyBase 子类"]
    name = str(getattr(cls, "name", "") or "").strip()
    if not name or name == "base":
        errs.append("缺少有效 name")
    if not callable(getattr(cls, "generate", None)):
        errs.append("缺少 generate 方法")
    else:
        # 未覆盖仍是基类 raise 的也可以，运行时会爆；这里只警告
        if getattr(cls.generate, "__qualname__", "").startswith("StrategyBase."):
            errs.append("未重写 generate()")
    regime = str(getattr(cls, "required_regime", "any"))
    if regime not in ("any", "trend", "chop", "mixed"):
        errs.append(f"required_regime 非法: {regime}")
    return errs


def _register_classes_from(full_name: str, mod, log) -> None:
    """扫描一个已加载的模块，把其中 StrategyBase 子类注册到 _REGISTRY。"""
    for attr in dir(mod):
        if attr.startswith("_"):
            continue
        obj = getattr(mod, attr, None)
        if not isinstance(obj, type):
            continue
        try:
            if not issubclass(obj, StrategyBase) or obj is StrategyBase:
                continue
        except TypeError:
            continue
        # 必须定义在该模块内，避免重复扫到 import 进来的类
        if getattr(obj, "__module__", "") != full_name:
            continue
        errs = validate_class(obj)
        if errs:
            log.warning("跳过策略 %s.%s: %s", full_name, obj.__name__, "; ".join(errs))
            continue
        register(obj, overwrite=True)


def discover(package_name: str = "v3.strategies", *, force: bool = False) -> Dict[str, Type[StrategyBase]]:
    """扫描包内模块并注册所有合法策略插件。"""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return dict(_REGISTRY)

    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        log.error("无法导入策略包 %s: %s", package_name, e)
        _DISCOVERED = True
        return dict(_REGISTRY)

    paths = getattr(pkg, "__path__", None)
    if not paths:
        _DISCOVERED = True
        return dict(_REGISTRY)

    skip = {"base", "registry"}
    seen_subpackages: set = set()
    for modinfo in pkgutil.iter_modules(paths):
        modname = modinfo.name
        if modname.startswith("_") or modname in skip:
            continue
        full = f"{package_name}.{modname}"
        try:
            mod = importlib.import_module(full)
        except Exception:
            log.warning("策略模块加载失败 %s:\n%s", full, traceback.format_exc())
            continue
        # 如果 modinfo 标记为子包（子目录），递归扫描子包内的 .py 模块
        if modinfo.ispkg and mod not in seen_subpackages:
            seen_subpackages.add(mod)
            sub_paths = getattr(mod, "__path__", None)
            if sub_paths:
                for sub_info in pkgutil.iter_modules(sub_paths):
                    sub_name = sub_info.name
                    if sub_name.startswith("_"):
                        continue
                    sub_full = f"{full}.{sub_name}"
                    try:
                        sub_mod = importlib.import_module(sub_full)
                    except Exception:
                        log.warning(
                            "策略子模块加载失败 %s:\n%s",
                            sub_full, traceback.format_exc(),
                        )
                        continue
                    _register_classes_from(sub_full, sub_mod, log)
        # 顶层模块也注册
        _register_classes_from(full, mod, log)

    _DISCOVERED = True
    log.debug("策略发现完成: %s", list_names())
    return dict(_REGISTRY)


def ensure_discovered() -> None:
    if not _DISCOVERED:
        discover()


def reload() -> Dict[str, Type[StrategyBase]]:
    """清空并重新扫描（开发自定义策略时热加载用）。"""
    global _DISCOVERED
    _REGISTRY.clear()
    _DISCOVERED = False
    return discover(force=True)
