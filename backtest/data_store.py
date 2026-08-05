# -*- coding: utf-8 -*-
"""
历史K线存储与加载。
- 优先读本地 CSV（data/backtest/）
- 若本机有网络，可用 OKX 公开接口拉取（无需 API Key）
- 无数据时可用合成行情做流程验证
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from logger import setup_logger

log = setup_logger("bt_data")

BAR_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1H": "1H",
    "4H": "4H",
}


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["ts", "open", "high", "low", "close", "vol"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"K线缺少列: {c}")
    out = df[need].copy()
    out["ts"] = pd.to_datetime(out["ts"])
    for c in ["open", "high", "low", "close", "vol"]:
        out[c] = out[c].astype(float)
    out = out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return out


def save_candles_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _ensure_df(df).to_csv(path, index=False)
    log.info(f"已保存 K线 {path} rows={len(df)}")


def load_candles_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _ensure_df(df)


def fetch_okx_candles_public(
    inst_id: str,
    bar: str = "15m",
    limit: int = 100,
    max_batches: int = 20,
) -> pd.DataFrame:
    """OKX 公开接口分页拉历史K线（无需 API Key）。多域名 + SSL 重试。"""
    import time as _time
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except Exception:
        from requests.packages.urllib3.util.retry import Retry  # type: ignore

    bases = ["https://www.okx.com", "https://aws.okx.com"]
    path = "/api/v5/market/history-candles"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "okx-grid-bot-backtest/5.0",
        "Accept": "application/json",
    })
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)

    rows: List[list] = []
    after = ""
    last_err: Optional[Exception] = None
    base_i = 0

    for batch_i in range(max_batches):
        params = {"instId": inst_id, "bar": bar, "limit": str(min(int(limit), 100))}
        if after:
            params["after"] = after

        batch: List[list] = []
        ok = False
        for attempt in range(6):
            base = bases[(base_i + attempt) % len(bases)]
            url = base + path
            try:
                r = session.get(url, params=params, timeout=(10, 30))
                if r.status_code != 200:
                    last_err = RuntimeError(f"HTTP {r.status_code} {url}")
                    _time.sleep(0.6 * (attempt + 1))
                    continue
                data = r.json()
                if str(data.get("code")) != "0":
                    last_err = RuntimeError(
                        f"OKX code={data.get('code')} msg={data.get('msg')} url={url}"
                    )
                    _time.sleep(0.5)
                    continue
                batch = data.get("data") or []
                ok = True
                base_i = bases.index(base) if base in bases else base_i
                break
            except requests.exceptions.SSLError as e:
                last_err = e
                log.warning(f"SSL 异常 {inst_id} {bar} attempt={attempt+1}: {e}")
                _time.sleep(1.2 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                last_err = e
                log.warning(f"网络异常 {inst_id} {bar} attempt={attempt+1}: {e}")
                _time.sleep(0.9 * (attempt + 1))
            except Exception as e:
                last_err = e
                log.warning(f"拉取异常 {inst_id} {bar} attempt={attempt+1}: {e}")
                _time.sleep(0.5)

        if not ok:
            if rows:
                log.warning(
                    f"分页中断 {inst_id} {bar}，已有 {len(rows)} 根，提前结束: {last_err}"
                )
                break
            raise RuntimeError(
                f"无法拉取 {inst_id} {bar}。常见原因：网络/代理/防火墙/SSL。"
                f"可尝试：1) 系统代理 2) pip install -U certifi urllib3 requests "
                f"3) 换网络。原始错误: {last_err}"
            )

        if not batch:
            break
        rows.extend(batch)
        after = str(batch[-1][0])
        if len(batch) < 100:
            break
        _time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
    )
    df = df[["ts", "open", "high", "low", "close", "vol"]]
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True).dt.tz_convert(None)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    log.info(
        f"拉取成功 {inst_id} {bar} 共 {len(df)} 根 | {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}"
    )
    return df


def generate_synthetic(
    inst_id: str = "BTC-USDT-SWAP",
    n: int = 3000,
    bar: str = "15m",
    start_price: float = 60000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """合成几何布朗 + 偶尔趋势段，仅用于无网时验证回测流程。"""
    rng = np.random.default_rng(seed + hash(inst_id + bar) % 10000)
    # 15m → 约 96 根/天
    freq = {"1m": "1min", "5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h"}.get(bar, "15min")
    ts = pd.date_range("2025-01-01", periods=n, freq=freq)
    mu = 0.00002
    sigma = 0.0018 if bar in ("1m", "5m") else 0.004
    rets = rng.normal(mu, sigma, size=n)
    # 嵌入两段趋势
    rets[n // 5 : n // 5 + n // 10] += 0.0012
    rets[3 * n // 5 : 3 * n // 5 + n // 12] -= 0.0010
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0015, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0015, n))
    vol = rng.uniform(50, 500, n) * (1 + np.abs(rets) * 80)
    df = pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close, "vol": vol}
    )
    log.info(f"合成数据 {inst_id} {bar} n={n} start={start_price}")
    return df


# 常见合约规格（回测用）
INST_SPECS = {
    "BTC-USDT-SWAP": {"ctVal": 0.01, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.1},
    "ETH-USDT-SWAP": {"ctVal": 0.1, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.01},
    "SOL-USDT-SWAP": {"ctVal": 1.0, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.001},
    "BNB-USDT-SWAP": {"ctVal": 0.01, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.01},
    "DOGE-USDT-SWAP": {"ctVal": 1000.0, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.00001},
    "XRP-USDT-SWAP": {"ctVal": 10.0, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.0001},
}


def default_spec(inst_id: str) -> Dict:
    if inst_id in INST_SPECS:
        return dict(INST_SPECS[inst_id])
    return {"ctVal": 0.01, "lotSz": 0.01, "minSz": 0.01, "tickSz": 0.01}


class CandleStore:
    """多周期 K 线仓库：{inst_id: {bar: DataFrame}}"""

    def __init__(self):
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}

    def set(self, inst_id: str, bar: str, df: pd.DataFrame) -> None:
        self.data.setdefault(inst_id, {})[bar] = _ensure_df(df)

    def get(self, inst_id: str, bar: str) -> Optional[pd.DataFrame]:
        return self.data.get(inst_id, {}).get(bar)

    def symbols(self) -> List[str]:
        return list(self.data.keys())

    def time_range(self, inst_id: str, bar: str = "15m"):
        df = self.get(inst_id, bar)
        if df is None or df.empty:
            return None, None
        return df["ts"].iloc[0], df["ts"].iloc[-1]

    def load_dir(self, root: str) -> None:
        """读取 data/backtest/{inst}/{bar}.csv"""
        if not os.path.isdir(root):
            return
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if name.endswith(".csv"):
                # 单文件: BTC-USDT-SWAP_15m.csv
                base = name[:-4]
                if "_" in base:
                    inst, bar = base.rsplit("_", 1)
                    self.set(inst, bar, load_candles_csv(p))
            elif os.path.isdir(p):
                for fn in os.listdir(p):
                    if fn.endswith(".csv"):
                        bar = fn[:-4]
                        self.set(name, bar, load_candles_csv(os.path.join(p, fn)))

    def save_dir(self, root: str) -> None:
        for inst, bars in self.data.items():
            d = os.path.join(root, inst)
            os.makedirs(d, exist_ok=True)
            for bar, df in bars.items():
                save_candles_csv(df, os.path.join(d, f"{bar}.csv"))



# 各周期每天大约K线数（用于「回测天数」→ 拉取批次数）
BARS_PER_DAY = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "1H": 24,
    "4H": 6,
}


def days_to_batches(days: float, bar: str = "15m", per_batch: int = 100) -> int:
    """回测天数 → OKX 分页批次数（每批最多 100 根）。"""
    days = max(1.0, float(days))
    bpd = BARS_PER_DAY.get(bar, 96)
    need = int(days * bpd * 1.05) + 20  # 略多一点余量
    return max(1, (need + per_batch - 1) // per_batch)


def prepare_store(
    symbols: List[str],
    bars: Optional[List[str]] = None,
    data_dir: str = "data/backtest",
    use_synthetic: bool = False,
    n_bars: int = 600,
    fetch: bool = False,
    max_batches: int = 0,
    days: float = 30.0,
    force_refetch: bool = False,
) -> CandleStore:
    bars = bars or ["1m", "15m", "1H", "4H"]
    store = CandleStore()
    if not force_refetch:
        store.load_dir(data_dir)

    errors = []
    for inst in symbols:
        for bar in bars:
            existing = store.get(inst, bar)
            if (not force_refetch) and existing is not None and len(existing) > 50:
                log.info(f"使用本地缓存 {inst} {bar} rows={len(existing)}")
                continue

            if fetch:
                try:
                    # 优先用「天数」换算批次数；若显式传了 max_batches>0 则取较大者
                    auto = days_to_batches(days, bar=bar)
                    batches = max(int(max_batches), auto) if max_batches and max_batches > 0 else auto
                    # 1m 数据量极大：按天数换算但硬顶 200 批（约 14 天 1m）
                    if bar == "1m":
                        batches = min(batches, max(days_to_batches(min(days, 14), "1m"), 40))
                    log.info(f"计划拉取 {inst} {bar}: ~{days}天 → {batches} 批")
                    df = fetch_okx_candles_public(inst, bar=bar, max_batches=batches)
                    if len(df) > 50:
                        store.set(inst, bar, df)
                        continue
                    raise RuntimeError(f"返回过少 rows={len(df)}")
                except Exception as e:
                    msg = f"拉取失败 {inst} {bar}: {e}"
                    log.error(msg)
                    errors.append(msg)
                    if not use_synthetic:
                        continue

            if use_synthetic:
                px0 = {
                    "BTC-USDT-SWAP": 65000,
                    "ETH-USDT-SWAP": 3500,
                    "SOL-USDT-SWAP": 140,
                }.get(inst, 100.0)
                n = n_bars if bar != "1m" else min(n_bars * 4, 2000)
                store.set(inst, bar, generate_synthetic(inst, n=n, bar=bar, start_price=px0))
            elif store.get(inst, bar) is None:
                log.error(f"无数据且未启用合成: {inst} {bar}")

    required = [b for b in ("15m", "1H", "4H") if b in bars]
    for inst in symbols:
        for b in required:
            df = store.get(inst, b)
            if df is None or len(df) < 80:
                detail = "; ".join(errors[:6]) if errors else "无本地CSV且拉取失败"
                tip = (
                    "缺少有效K线 {inst} {b}（需要真实数据）。{detail}\n"
                    "处理建议:\n"
                    "  1) 浏览器能否打开 https://www.okx.com\n"
                    "  2) pip install -U certifi requests urllib3\n"
                    "  3) 代理: set HTTPS_PROXY=http://127.0.0.1:端口\n"
                    "  4) 删除 data/backtest 下错误缓存 csv 后重试\n"
                    "  5) --no-hft 时不拉 1m，更快更稳"
                ).format(inst=inst, b=b, detail=detail)
                raise RuntimeError(tip)

    store.save_dir(data_dir)
    return store
