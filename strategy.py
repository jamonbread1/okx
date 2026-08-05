# -*- coding: utf-8 -*-
"""
统一信号接口（P0-1）。

`from strategy import StrategyEngine` 现在直接得到 v2 引擎（v2.engine.StrategyEngine），
原生签名：
    generate_signal(inst_id, df, df_htf, capital, leverage, specs, funding_rate)

v2 引擎不再需要 fetcher（df 由调用方传入），也不再登记幻影仓位：
真实成交后由调用方调用 confirm_fill / confirm_partial_close / confirm_close。

`TradeSignal` 作为 v2 Signal 的兼容别名，供旧调用方使用。
"""
from __future__ import annotations

from v2.engine import StrategyEngine
from v2.strategies.base import Signal as TradeSignal

__all__ = ["StrategyEngine", "TradeSignal"]
