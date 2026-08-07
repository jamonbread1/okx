# -*- coding: utf-8 -*-
"""
历史资金费率入库 / 查询（OKX public funding-rate-history）

写入与 K 线同一 SQLite：data/okx_history/bars.db → 表 funding
可从 2022-03 起按需拉取（OKX 分页）。
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from logger import setup_logger

log = setup_logger("funding_store")

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # backtest/
    root = os.path.dirname(here)
    if os.path.isfile(os.path.join(root, "config.yaml")) or os.path.isfile(os.path.join(root, "config.json")):
        return root
    return root


DB_PATH = os.path.join(_project_root(), "data", "okx_history", "bars.db")
API = "https://www.okx.com/api/v5/public/funding-rate-history"


def _conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    c = sqlite3.connect(db_path)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS funding (
            symbol TEXT NOT NULL,
            funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL,
            PRIMARY KEY (symbol, funding_time)
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_funding_sym_ts ON funding(symbol, funding_time)"
    )
    c.commit()
    return c


def fetch_funding_history(
    inst_id: str,
    max_pages: int = 200,
    pause: float = 0.15,
    min_time: Optional[int] = None,
) -> List[Dict]:
    """分页拉取资金费率历史（新→旧）。

    min_time 为增量模式：当某页出现 funding_time <= min_time 的旧数据时停止，
    只拉取比库内最新结算更晚的费率，避免全量重下。
    """
    rows: List[Dict] = []
    after = ""
    for _ in range(max_pages):
        params = {"instId": inst_id, "limit": "100"}
        if after:
            params["after"] = after
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if str(data.get("code")) != "0":
            raise RuntimeError(f"OKX funding API: {data}")
        batch = data.get("data") or []
        if not batch:
            break
        for x in batch:
            ft = int(x.get("fundingTime") or 0)
            rows.append({
                "symbol": inst_id,
                "funding_time": ft,
                "funding_rate": float(x.get("realizedRate") or x.get("fundingRate") or 0),
            })
        after = str(batch[-1].get("fundingTime") or "")
        # 增量：已触及库内已有时间 → 停止，避免继续翻旧页
        if min_time is not None and int(batch[-1].get("fundingTime") or 0) <= min_time:
            break
        if len(batch) < 100:
            break
        time.sleep(pause)
    # 去重
    uniq = {(r["symbol"], r["funding_time"]): r for r in rows if r["funding_time"] > 0}
    out = list(uniq.values())
    out.sort(key=lambda x: x["funding_time"])
    log.info(f"{inst_id} 资金费率拉取 {len(out)} 条")
    return out


def write_funding(rows: List[Dict], db_path: str = DB_PATH) -> int:
    if not rows:
        return 0
    c = _conn(db_path)
    try:
        c.executemany(
            "INSERT OR REPLACE INTO funding (symbol, funding_time, funding_rate) VALUES (?,?,?)",
            [(r["symbol"], r["funding_time"], r["funding_rate"]) for r in rows],
        )
        c.commit()
        return len(rows)
    finally:
        c.close()


def load_funding_series(
    symbol: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    c = _conn(db_path)
    try:
        q = "SELECT funding_time, funding_rate FROM funding WHERE symbol=?"
        params: list = [symbol]
        if start_ms is not None:
            q += " AND funding_time>=?"
            params.append(int(start_ms))
        if end_ms is not None:
            q += " AND funding_time<=?"
            params.append(int(end_ms))
        q += " ORDER BY funding_time"
        df = pd.read_sql_query(q, c, params=params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["funding_time"], unit="ms")
        return df
    finally:
        c.close()


def rate_at(
    symbol: str,
    ts: pd.Timestamp,
    db_path: str = DB_PATH,
    cache: Optional[Dict] = None,
) -> float:
    """取 ts 之前最近一条资金费率（结算后生效的那档）。"""
    key = symbol
    if cache is not None and key in cache:
        arr_t, arr_r = cache[key]
        if arr_t is None or len(arr_t) == 0:
            return 0.0
        t = int(pd.Timestamp(ts).timestamp() * 1000)
        import numpy as np
        i = int(np.searchsorted(arr_t, t, side="right") - 1)
        if i < 0:
            return 0.0
        return float(arr_r[i])
    c = _conn(db_path)
    try:
        t = int(pd.Timestamp(ts).timestamp() * 1000)
        cur = c.execute(
            "SELECT funding_rate FROM funding WHERE symbol=? AND funding_time<=? "
            "ORDER BY funding_time DESC LIMIT 1",
            (symbol, t),
        )
        row = cur.fetchone()
        return float(row[0]) if row else 0.0
    finally:
        c.close()


def preload_cache(symbols: List[str], db_path: str = DB_PATH) -> Dict:
    cache = {}
    for s in symbols:
        df = load_funding_series(s, db_path=db_path)
        if df.empty:
            cache[s] = (None, None)
        else:
            cache[s] = (
                df["funding_time"].astype("int64").values,
                df["funding_rate"].astype(float).values,
            )
        log.info(f"funding cache {s}: {0 if df.empty else len(df)} rows")
    return cache


def latest_funding_time(symbol: str, db_path: str = DB_PATH) -> Optional[int]:
    """库内该 symbol 最新一条 funding_time（毫秒），无则 None。"""
    c = _conn(db_path)
    try:
        cur = c.execute(
            "SELECT MAX(funding_time) FROM funding WHERE symbol=?", (symbol,)
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        c.close()


def build_funding_for_symbols(
    symbols: List[str],
    db_path: str = DB_PATH,
    max_pages: int = 300,
    incremental: bool = True,
) -> None:
    """下载并写入资金费率。

    incremental=True（默认）：只拉取比库内最新更晚的费率，节省 API 与时间；
    incremental=False：全量重下（force）。
    """
    for s in symbols:
        try:
            min_time = None
            if incremental:
                min_time = latest_funding_time(s, db_path=db_path)
            rows = fetch_funding_history(s, max_pages=max_pages, min_time=min_time)
            n = write_funding(rows, db_path=db_path)
            mode = "incremental" if incremental else "full"
            log.info(f"写入 {s} funding {n} 条 → {db_path} ({mode})")
        except Exception as e:
            log.error(f"{s} funding 失败: {e}")
