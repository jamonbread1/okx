# -*- coding: utf-8 -*-
"""v2 技术指标核心 — 纯 pandas/numpy 实现。

所有函数接受 pd.Series 或 pd.DataFrame，返回 pd.Series。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均（span 参数）。"""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数 (Wilder's smoothing)。

    参考: backtrader / freqtrade 标准实现。
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (macd_line, signal_line, histogram)。

    参考: je-suis-tm/quant-trading MACD Oscillator。
    """
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    macd_line = e_fast - e_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真实范围 (Wilder's smoothing)。

    df 需包含 high/low/close 列。
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均方向指数 (Wilder's smoothing)。

    参考: backtrader 标准实现。
    """
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_val = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_val.replace(0, np.nan))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# 布林带 & 布林带宽度
# ---------------------------------------------------------------------------

def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林带 (lower, mid, upper)。

    参考: freqtrade qtpylib.bollinger_bands。
    """
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    lower = mid - num_std * std
    upper = mid + num_std * std
    return lower, mid, upper


def bbw(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """布林带宽度 (upper - lower) / mid。"""
    lower, mid, upper = bollinger_bands(series, period, num_std)
    return (upper - lower) / mid.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 唐奇安通道
# ---------------------------------------------------------------------------

def donchian(
    df: pd.DataFrame,
    period: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """唐奇安通道 (lower, mid, upper) — 用前 period 根 K 线（不含当前）。

    参考: v1 MomentumAlgo (Donchian breakout)。
    """
    hh = df["high"].shift(1).rolling(window=period, min_periods=period).max()
    ll = df["low"].shift(1).rolling(window=period, min_periods=period).min()
    mid = (hh + ll) / 2
    return ll, mid, hh


# ---------------------------------------------------------------------------
# 随机指标 & Williams %R
# ---------------------------------------------------------------------------

def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """随机指标 (K, D)。"""
    hh = df["high"].rolling(window=k_period).max()
    ll = df["low"].rolling(window=k_period).min()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R。"""
    hh = df["high"].rolling(window=period).max()
    ll = df["low"].rolling(window=period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


# ---------------------------------------------------------------------------
# 成交量指标
# ---------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    """能量潮 (On Balance Volume)。

    参考: je-suis-tm/quant-trading。
    """
    sign = np.sign(df["close"].diff())
    sign.iloc[0] = 0
    return (sign * df["vol"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """成交量加权平均价 (当日累计)。

    参考: freqtrade。
    """
    tp = typical_price(df)
    cum_tp_vol = (tp * df["vol"]).cumsum()
    cum_vol = df["vol"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# EWMAC (指数加权移动平均交叉) — Rob Carver / pysystemtrade
# ---------------------------------------------------------------------------

def ewmac(series: pd.Series, fast: int = 16, slow: int = 64) -> pd.Series:
    """EWMAC — 指数加权移动平均交叉。

    参考: Rob Carver pysystemtrade ewmac.py。
    核心思想: 快慢 EMA 差值作为趋势信号，归一化后可跨品种比较。
    """
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    raw = e_fast - e_slow
    # 归一化：除以价格，使信号跨品种可比
    return raw / series.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 交叉检测
# ---------------------------------------------------------------------------

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """series1 上穿 series2。"""
    prev = series1.shift(1) < series2.shift(1)
    curr = series1 > series2
    return prev & curr


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """series1 下穿 series2。"""
    prev = series1.shift(1) > series2.shift(1)
    curr = series1 < series2
    return prev & curr


# ---------------------------------------------------------------------------
# Heikin-Ashi 蜡烛
# ---------------------------------------------------------------------------

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi 蜡烛图（向量化实现，与逐行递推数学等价）。

    参考: je-suis-tm/quant-trading Heikin-Ashi backtest。
    """
    ha = pd.DataFrame(index=df.index)

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["close"] = ha_close

    # 递推 ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    # 等价于对 shift(1) 后的 ha_close 做 alpha=0.5、adjust=False 的 EWM：
    #   y[i] = 0.5*y[i-1] + 0.5*x[i]，x[i] = ha_close[i-1]
    # 首值种子为 (open[0] + close[0]) / 2。完全向量化，O(n) → 常数级。
    shifted = ha_close.shift(1)
    shifted.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    ha["open"] = shifted.ewm(alpha=0.5, adjust=False).mean()

    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def typical_price(df: pd.DataFrame) -> pd.Series:
    """典型价格 (H+L+C)/3。"""
    return (df["high"] + df["low"] + df["close"]) / 3


def median_price(df: pd.DataFrame) -> pd.Series:
    """中间价 (H+L)/2。"""
    return (df["high"] + df["low"]) / 2
