# -*- coding: utf-8 -*-
"""
OKX 历史 K 线统一入口（v2.3）

一个文件完成：下载 / 校验 / 入库(build) / 增量更新 / 状态查询。
直接调用 OKX 公开接口 /api/v5/market/history-candles，按周期下载，
不再依赖成交 tick 聚合，也不再生成合成 K 线。

能力：
  1) download  — 按品种+周期分页下载历史 K 线
  2) validate  — 检查时间连续性、OHLC 合法性、重复时间戳
  3) build     — 写入 parquet + SQLite（PRIMARY KEY 防重，INSERT OR REPLACE）
  4) update    — 从库内最新时间戳向「更近」方向增量补全
  5) status    — 列出库内覆盖范围
  6) funding   — 可选同步资金费率

示例：
  # 首次全量下载并入库（默认 1H,4H + 常用辅助周期）
  python tools/data_manager.py --symbols BTC-USDT-SWAP,ETH-USDT-SWAP --bars 1H,4H --days 180

  # 增量更新（自动从库内最新往新方向拉）
  python tools/data_manager.py --update --symbols BTC-USDT-SWAP,ETH-USDT-SWAP

  # 仅查看库状态
  python tools/data_manager.py --status

  # 下载 + 校验 + 入库 + 资金费率
  python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1m,5m,15m,1H,4H --days 90 --funding

  # 强制重下某区间（覆盖库内同 ts）
  python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1H --start 2025-01-01 --end 2025-06-01 --force
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logger import setup_logger

log = setup_logger("data_manager")

BASE_URLS = ["https://www.okx.com", "https://aws.okx.com"]
CANDLES_PATH = "/api/v5/market/history-candles"
def _pkg_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # tools/
    root = os.path.dirname(here)
    if os.path.isfile(os.path.join(root, "config.yaml")) or os.path.isfile(os.path.join(root, "config.json")):
        return root
    return root


_PKG_ROOT = _pkg_root()
DB_PATH_DEFAULT = os.path.join(_PKG_ROOT, "data", "okx_history", "bars.db")
PQ_ROOT = os.path.join(_PKG_ROOT, "data", "okx_history", "parquet")

# OKX 支持的 bar 与本地统一命名
SUPPORTED_BARS = [
    "1m", "3m", "5m", "15m", "30m",
    "1H", "2H", "4H", "6H", "12H",
    "1D", "1W", "1M",
]
DEFAULT_BARS = ["1m", "5m", "15m", "1H", "4H"]

# 各周期每天约多少根（用于 days → 目标根数估算）
BARS_PER_DAY: Dict[str, float] = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
    "1H": 24, "2H": 12, "4H": 6, "6H": 4, "12H": 2,
    "1D": 1, "1W": 1 / 7, "1M": 1 / 30,
}

BAR_ALIASES = {
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
    "60m": "1H", "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D", "1w": "1W", "1Mth": "1M",
}


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "okx-quant-v2.3/data-manager",
        "Accept": "application/json",
    })
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    return s


_SESS: Optional[requests.Session] = None


def _get_sess() -> requests.Session:
    global _SESS
    if _SESS is None:
        _SESS = _session()
    return _SESS


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

def normalize_bar(bar: str) -> str:
    b = (bar or "").strip()
    return BAR_ALIASES.get(b, b)


def fetch_history_candles(
    inst_id: str,
    bar: str = "1H",
    total: int = 2000,
    after_ms: Optional[int] = None,
    before_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    pause: float = 0.12,
) -> pd.DataFrame:
    """
    分页下载 OKX 历史 K 线（从新往旧翻页）。

    参数语义（对齐 OKX）：
      - after_ms: 分页游标，只返回 ts < after 的更旧数据；首屏可为空=从最新开始
      - before_ms: 只返回 ts > before 的更新数据（少用）
      - until_ms: 时间下界，当本批最旧一根 ts <= until_ms 时停止（用于拉到 start）

    注意：after_ms 只做游标，不要和下界 until_ms 混用。

    返回列：ts, open, high, low, close, vol  （ts 为 naive UTC datetime）
    """
    bar = normalize_bar(bar)
    if bar not in SUPPORTED_BARS:
        raise ValueError(f"不支持的周期 {bar}，可选: {SUPPORTED_BARS}")

    sess = _get_sess()
    all_rows: List[list] = []
    after = str(after_ms) if after_ms else ""
    before = str(before_ms) if before_ms else ""
    base_i = 0
    last_err: Optional[Exception] = None
    # 页数上限按目标根数估算，并留余量
    max_pages = max(1, (int(total) + 99) // 100) + 20

    for _page in range(max_pages):
        if len(all_rows) >= total:
            break
        params: Dict[str, str] = {
            "instId": inst_id,
            "bar": bar,
            "limit": "100",
        }
        if after:
            params["after"] = after
        if before:
            params["before"] = before

        batch: List[list] = []
        ok = False
        for attempt in range(6):
            base = BASE_URLS[(base_i + attempt) % len(BASE_URLS)]
            url = base + CANDLES_PATH
            try:
                r = sess.get(url, params=params, timeout=(10, 30))
                if r.status_code != 200:
                    last_err = RuntimeError(f"HTTP {r.status_code} {url}")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                data = r.json()
                if str(data.get("code")) != "0":
                    last_err = RuntimeError(
                        f"OKX code={data.get('code')} msg={data.get('msg')}"
                    )
                    time.sleep(0.4)
                    continue
                batch = data.get("data") or []
                ok = True
                base_i = BASE_URLS.index(base) if base in BASE_URLS else base_i
                break
            except requests.exceptions.RequestException as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
            except Exception as e:
                last_err = e
                time.sleep(0.4)

        if not ok:
            if all_rows:
                log.warning(f"{inst_id} {bar} 分页中断，已有 {len(all_rows)} 根: {last_err}")
                break
            raise RuntimeError(
                f"无法拉取 {inst_id} {bar}: {last_err}\n"
                "建议: 检查网络 / 代理 / pip install -U certifi requests urllib3"
            )

        if not batch:
            break

        all_rows.extend(batch)

        # OKX 返回新→旧；继续往更早翻：after = 本批最旧一根
        oldest = int(batch[-1][0])
        after = str(oldest)

        # 已翻过时间下界（start）则停
        if until_ms is not None and oldest <= int(until_ms):
            break
        # 本批不足 100 说明没有更早数据
        if len(batch) < 100:
            break
        time.sleep(pause)

    if not all_rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])

    df = pd.DataFrame(
        all_rows,
        columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"],
    )
    df = df[["ts", "open", "high", "low", "close", "vol"]].copy()
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["ts"], keep="last")
    df = df.sort_values("ts").reset_index(drop=True)
    log.info(
        f"下载 {inst_id} {bar}: {len(df)} 根 | "
        f"{df['ts'].iloc[0] if len(df) else '-'} → {df['ts'].iloc[-1] if len(df) else '-'}"
    )
    return df


def download_range(
    inst_id: str,
    bar: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    days: float = 0,
    max_bars: int = 0,
) -> pd.DataFrame:
    """
    按时间窗口下载。优先 start/end；否则 days；否则默认最近 60 天。
    从「end」往「start」方向翻页，直到覆盖或无更早数据。
    """
    bar = normalize_bar(bar)
    now = datetime.utcnow()
    if end is None:
        end = now
    if start is None:
        if days and days > 0:
            start = end - timedelta(days=float(days))
        else:
            start = end - timedelta(days=60)

    if start >= end:
        raise ValueError(f"start({start}) 必须早于 end({end})")

    bpd = BARS_PER_DAY.get(bar, 24)
    need = int((end - start).total_seconds() / 86400.0 * bpd * 1.08) + 50
    if max_bars > 0:
        need = min(need, max_bars)
    # 单次接口软顶：避免一次请求无限翻（1m 很长时仍可分多次 --update）
    need = min(need, 200_000)

    # naive datetime 按 UTC 解释
    start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)

    # 从最新往旧翻；until_ms=start 作为真正的时间下界（与 after 游标分离）
    df = fetch_history_candles(
        inst_id,
        bar=bar,
        total=need,
        after_ms=None,          # 空游标 = 从最新一根开始
        until_ms=start_ms,      # 翻到 start 为止
    )
    if df.empty:
        return df

    # 裁剪到 [start, end]
    df = df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] <= pd.Timestamp(end))]
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    return df


def download_newer(
    inst_id: str,
    bar: str,
    since_ts: datetime,
    max_bars: int = 0,
) -> pd.DataFrame:
    """
    增量：拉 since_ts 之后（更近）的 K 线。

    从「现在」往更早翻页，直到最旧一根 <= since_ts 或达到上限。
    max_bars<=0 时按周期自动估算（至少覆盖 since→now，并加余量）。
    """
    bar = normalize_bar(bar)
    since_ts = pd.Timestamp(since_ts).to_pydatetime().replace(tzinfo=None)
    now = datetime.utcnow()
    if since_ts >= now:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])

    bpd = BARS_PER_DAY.get(bar, 24)
    span_days = max(1.0, (now - since_ts).total_seconds() / 86400.0)
    auto = int(span_days * bpd * 1.15) + 50
    if max_bars and max_bars > 0:
        need = min(int(max_bars), 80_000)
    else:
        need = min(max(auto, 200), 80_000)

    since_ms = int(pd.Timestamp(since_ts).replace(tzinfo=timezone.utc).timestamp() * 1000)
    # 从最新往旧翻，直到翻过 since
    df = fetch_history_candles(
        inst_id, bar=bar, total=need, after_ms=None, until_ms=since_ms,
    )
    if df.empty:
        return df
    df = df[df["ts"] > pd.Timestamp(since_ts)]
    return df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def validate_candles(
    df: pd.DataFrame,
    bar: str = "1H",
    strict: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    校验并清洗 K 线：
      - 去重时间戳
      - OHLC 合法性（high>=low, high>=open/close, low<=open/close, 价格>0）
      - 按周期估算缺口（仅报告，不填补）
    返回 (清洗后 df, 报告 dict)
    """
    report: Dict = {
        "rows_in": 0,
        "rows_out": 0,
        "dup_ts": 0,
        "bad_ohlc": 0,
        "gaps": 0,
        "ok": True,
        "messages": [],
    }
    if df is None or df.empty:
        report["ok"] = False
        report["messages"].append("空数据")
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"]), report

    report["rows_in"] = len(df)
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    n0 = len(out)
    out = out.drop_duplicates(subset=["ts"], keep="last")
    report["dup_ts"] = n0 - len(out)

    for c in ["open", "high", "low", "close", "vol"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    bad = (
        (out["open"] <= 0) | (out["high"] <= 0) | (out["low"] <= 0) | (out["close"] <= 0)
        | (out["high"] < out["low"])
        | (out["high"] < out["open"]) | (out["high"] < out["close"])
        | (out["low"] > out["open"]) | (out["low"] > out["close"])
        | out[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    report["bad_ohlc"] = int(bad.sum())
    out = out.loc[~bad].copy()
    out = out.sort_values("ts").reset_index(drop=True)

    # 缺口检测
    bar = normalize_bar(bar)
    bpd = BARS_PER_DAY.get(bar, 24)
    if bpd > 0 and len(out) >= 3:
        expected_delta = pd.Timedelta(days=1) / bpd
        deltas = out["ts"].diff().dropna()
        # 允许 1.5 倍周期内算正常
        gaps = deltas > expected_delta * 1.6
        report["gaps"] = int(gaps.sum())
        if report["gaps"] > 0:
            report["messages"].append(
                f"检测到约 {report['gaps']} 处时间缺口（>{expected_delta}×1.6）"
            )

    report["rows_out"] = len(out)
    if report["dup_ts"]:
        report["messages"].append(f"去除重复 ts {report['dup_ts']} 条")
    if report["bad_ohlc"]:
        report["messages"].append(f"去除非法 OHLC {report['bad_ohlc']} 条")
    if len(out) < 10:
        report["ok"] = False
        report["messages"].append(f"有效 K 线过少 ({len(out)})")
    if strict and report["gaps"] > max(5, len(out) // 50):
        report["ok"] = False
        report["messages"].append("缺口过多（strict 模式）")

    return out, report


# ---------------------------------------------------------------------------
# 入库（build）— 防重复 PRIMARY KEY + INSERT OR REPLACE
# ---------------------------------------------------------------------------

def _ensure_db(db_path: str):
    from backtest.trade_pipeline import _ensure_db as _e
    return _e(db_path)


def upsert_bars_to_db(
    symbol: str,
    bar: str,
    df: pd.DataFrame,
    db_path: str = DB_PATH_DEFAULT,
) -> int:
    """
    增量友好写入：INSERT OR REPLACE，不先 DELETE 全表。
    PRIMARY KEY (symbol, bar, ts) 保证无重复。
    写完后刷新 meta 覆盖范围。
    """

    if df is None or df.empty:
        return 0
    bar = normalize_bar(bar)
    conn = _ensure_db(db_path)
    try:
        rows = []
        for r in df.itertuples(index=False):
            ts = pd.Timestamp(r.ts)
            ts_i = int(ts.timestamp())
            rows.append((
                symbol, bar, ts_i,
                float(r.open), float(r.high), float(r.low), float(r.close),
                float(getattr(r, "vol", 0) or 0),
                float(getattr(r, "buy_vol", 0) or 0),
                float(getattr(r, "sell_vol", 0) or 0),
                float(getattr(r, "cvd", 0) or 0),
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(symbol,bar,ts,open,high,low,close,vol,buy_vol,sell_vol,cvd) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        # 刷新 meta
        cur = conn.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM ohlcv WHERE symbol=? AND bar=?",
            (symbol, bar),
        )
        n_bars, ts_min, ts_max = cur.fetchone()
        conn.execute(
            "INSERT OR REPLACE INTO meta (symbol,bar,n_bars,ts_min,ts_max,built_at) "
            "VALUES (?,?,?,?,?,datetime('now'))",
            (symbol, bar, int(n_bars or 0), ts_min, ts_max),
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def write_parquet(symbol: str, bar: str, df: pd.DataFrame, pq_root: str = PQ_ROOT) -> str:
    """合并写入 parquet（若已有则与旧数据合并去重）。"""
    bar = normalize_bar(bar)
    out_dir = os.path.join(pq_root, symbol)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bars_{bar}.parquet")
    if os.path.exists(path) and not df.empty:
        try:
            old = pd.read_parquet(path)
            old["ts"] = pd.to_datetime(old["ts"])
            df = (
                pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset=["ts"], keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        except Exception as e:
            log.warning(f"合并旧 parquet 失败，将覆盖: {e}")
    cols = [c for c in ["ts", "open", "high", "low", "close", "vol", "buy_vol", "sell_vol", "cvd"] if c in df.columns]
    if "buy_vol" not in df.columns:
        df = df.copy()
        df["buy_vol"] = 0.0
        df["sell_vol"] = 0.0
        df["cvd"] = 0.0
        cols = ["ts", "open", "high", "low", "close", "vol", "buy_vol", "sell_vol", "cvd"]
    try:
        df[cols].to_parquet(path, index=False)
    except Exception:
        alt = path.replace(".parquet", ".pkl")
        df[cols].to_pickle(alt)
        log.info(f"写入 {alt} rows={len(df)}")
        return alt
    log.info(f"写入 {path} rows={len(df)}")
    return path


def db_latest_ts(symbol: str, bar: str, db_path: str = DB_PATH_DEFAULT) -> Optional[pd.Timestamp]:
    """库内该 symbol+bar 最新一根时间，无则 None。"""
    import sqlite3
    if not os.path.exists(db_path):
        return None
    bar = normalize_bar(bar)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND bar=?", (symbol, bar)
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return pd.to_datetime(int(row[0]), unit="s")
        return None
    finally:
        conn.close()


def db_count_duplicates(db_path: str = DB_PATH_DEFAULT) -> int:
    """检测 ohlcv 表是否存在重复 (symbol,bar,ts) — 正常应为 0。"""
    import sqlite3
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT symbol, bar, ts, COUNT(*) c FROM ohlcv "
            "  GROUP BY symbol, bar, ts HAVING c > 1"
            ")"
        )
        return int(cur.fetchone()[0] or 0)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 高层流程：download + validate + build / update
# ---------------------------------------------------------------------------

def process_symbol_bar(
    symbol: str,
    bar: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    days: float = 0,
    update: bool = False,
    force: bool = False,
    db_path: str = DB_PATH_DEFAULT,
    write_pq: bool = True,
    strict: bool = False,
) -> Dict:
    """
    单品种单周期完整流程。
    update=True：从库内最新向新方向补；库空则退化为全量 days/start-end。
    """
    bar = normalize_bar(bar)
    result = {"symbol": symbol, "bar": bar, "rows": 0, "action": "", "ok": False, "msg": ""}

    latest = db_latest_ts(symbol, bar, db_path=db_path)

    if update and latest is not None and not force:
        log.info(f"[update] {symbol} {bar} 从 {latest} 向新方向补全")
        raw = download_newer(symbol, bar, since_ts=latest.to_pydatetime())
        result["action"] = "update"
        if raw.empty:
            result["ok"] = True
            result["msg"] = "已是最新"
            log.info(f"{symbol} {bar}: 已是最新，无需更新")
            return result
    else:
        result["action"] = "download"
        if update and latest is None:
            log.info(f"[update] {symbol} {bar} 库内无数据，退化为全量下载")
        if force and latest is not None:
            log.info(f"[force] {symbol} {bar} 强制重下")
        # 未指定范围时给默认天数，避免 days=0 只拉到默认 60 天且调用方不知情
        use_days = days if days and days > 0 else (0 if (start or end) else 90)
        raw = download_range(symbol, bar, start=start, end=end, days=use_days)

    if raw.empty:
        result["msg"] = "无数据返回"
        log.warning(f"{symbol} {bar}: 无数据")
        return result

    clean, report = validate_candles(raw, bar=bar, strict=strict)
    for m in report.get("messages") or []:
        log.info(f"  validate {symbol} {bar}: {m}")

    if clean.empty:
        result["msg"] = "校验后为空"
        return result

    n = upsert_bars_to_db(symbol, bar, clean, db_path=db_path)
    if write_pq:
        write_parquet(symbol, bar, clean)

    result["rows"] = n
    result["ok"] = report.get("ok", True)
    result["msg"] = f"写入 {n} 根 | dup_removed={report.get('dup_ts', 0)} gaps={report.get('gaps', 0)}"
    log.info(f"{symbol} {bar}: {result['msg']}")
    return result


def run_pipeline(
    symbols: List[str],
    bars: List[str],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    days: float = 0,
    update: bool = False,
    force: bool = False,
    db_path: str = DB_PATH_DEFAULT,
    write_pq: bool = True,
    funding: bool = False,
    strict: bool = False,
) -> List[Dict]:
    results = []
    for i, sym in enumerate(symbols):
        for bar in bars:
            log.info(f"==== [{i + 1}/{len(symbols)}] {sym} {bar} ====")
            try:
                r = process_symbol_bar(
                    sym, bar,
                    start=start, end=end, days=days,
                    update=update, force=force,
                    db_path=db_path, write_pq=write_pq, strict=strict,
                )
                results.append(r)
            except Exception as e:
                log.exception(f"{sym} {bar} 失败: {e}")
                results.append({
                    "symbol": sym, "bar": bar, "rows": 0,
                    "action": "error", "ok": False, "msg": str(e),
                })

    # 重复检测
    dups = db_count_duplicates(db_path)
    if dups > 0:
        log.error(f"数据库存在 {dups} 组重复 (symbol,bar,ts)！请检查 schema")
    else:
        log.info("数据库重复检测通过：0 组重复")

    if funding:
        try:
            from backtest.funding_store import build_funding_for_symbols
            build_funding_for_symbols(symbols, db_path=db_path, incremental=not force)
        except Exception as e:
            log.error(f"资金费率同步失败: {e}")

    return results


def print_status(db_path: str = DB_PATH_DEFAULT, symbols: Optional[List[str]] = None) -> None:
    from backtest.trade_pipeline import list_db_coverage
    abs_db = os.path.abspath(db_path)
    print(f"DB: {abs_db}")
    print(f"exists: {os.path.isfile(db_path)}")
    cov = list_db_coverage(db_path)
    if cov.empty:
        print(f"数据库为空或不存在: {abs_db}")
        return
    if symbols:
        cov = cov[cov["symbol"].isin(symbols)]
    print("\n=== bars.db 覆盖 ===")
    print(cov.to_string(index=False))
    dups = db_count_duplicates(db_path)
    print(f"\n重复键组数: {dups}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_dt(s: str, end_of_day: bool = False) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    if "T" in s or " " in s:
        dt = datetime.fromisoformat(s.replace(" ", "T").replace("Z", ""))
    else:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=None)


def main():
    ap = argparse.ArgumentParser(
        description="OKX K 线统一管理：下载 / 校验 / 入库 / 增量更新（v2.3）"
    )
    ap.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP")
    ap.add_argument(
        "--bars",
        default=",".join(DEFAULT_BARS),
        help=f"逗号分隔周期，可选: {','.join(SUPPORTED_BARS)}",
    )
    ap.add_argument("--days", type=float, default=0, help="回溯天数（无 start/end 时生效）")
    ap.add_argument("--start", default="", help="YYYY-MM-DD")
    ap.add_argument("--end", default="", help="YYYY-MM-DD")
    ap.add_argument("--update", action="store_true", help="从库内最新向新方向增量更新")
    ap.add_argument("--force", action="store_true", help="强制重下（覆盖同 ts）")
    ap.add_argument("--status", action="store_true", help="仅打印库覆盖状态")
    ap.add_argument("--funding", action="store_true", help="同步资金费率")
    ap.add_argument("--no-parquet", action="store_true", help="不写 parquet，仅 SQLite")
    ap.add_argument("--strict", action="store_true", help="校验更严格（缺口过多则标记失败）")
    ap.add_argument("--db", default=DB_PATH_DEFAULT)
    ap.add_argument("--validate-only", action="store_true",
                    help="只下载并校验，不入库（调试用）")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bars = [normalize_bar(b.strip()) for b in args.bars.split(",") if b.strip()]
    for b in bars:
        if b not in SUPPORTED_BARS:
            log.error(f"不支持的周期 {b}，可选: {SUPPORTED_BARS}")
            sys.exit(1)

    # 相对路径的 --db 锚定到发行包根，避免 cwd 漂移
    db_path = args.db
    if db_path and not os.path.isabs(db_path):
        db_path = os.path.join(_PKG_ROOT, db_path)

    if args.status:
        print_status(db_path, symbols)
        return

    start = _parse_dt(args.start)
    end = _parse_dt(args.end, end_of_day=True)
    days = float(args.days) if args.days else 0
    if not args.update and not start and not end and not days:
        days = 90  # 默认 90 天
        log.info("未指定时间范围，默认 --days 90")

    if args.validate_only:
        for sym in symbols:
            for bar in bars:
                raw = download_range(sym, bar, start=start, end=end, days=days)
                clean, report = validate_candles(raw, bar=bar, strict=args.strict)
                print(f"{sym} {bar}: in={report['rows_in']} out={report['rows_out']} "
                      f"dup={report['dup_ts']} bad={report['bad_ohlc']} gaps={report['gaps']} "
                      f"ok={report['ok']}")
        return

    results = run_pipeline(
        symbols, bars,
        start=start, end=end, days=days,
        update=args.update, force=args.force,
        db_path=db_path, write_pq=not args.no_parquet,
        funding=args.funding, strict=args.strict,
    )

    print("\n===== 结果汇总 =====")
    for r in results:
        flag = "OK" if r.get("ok") else "FAIL"
        print(f"  [{flag}] {r['symbol']} {r['bar']} | {r.get('action')} | "
              f"rows={r.get('rows')} | {r.get('msg')}")

    print_status(db_path, symbols)
    print(f"\n完成。库: {os.path.abspath(db_path)}")
    print("parquet: data/okx_history/parquet/{SYMBOL}/bars_*.parquet")


if __name__ == "__main__":
    main()
