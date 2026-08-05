# -*- coding: utf-8 -*-
"""
OKX 官方历史成交 → 多周期 K 线 + SQLite 库（低内存流式）

目录：
  data/okx_history/raw/{SYMBOL}/*.csv|.zip   （构建后可删除）
  data/okx_history/parquet/{SYMBOL}/bars_*.parquet
  data/okx_history/bars.db                  （统一 SQLite，回测首选）

常用周期一次生成：1m 3m 5m 15m 30m 1H 2H 4H 1D
回测可按 日期~日期 + 自选周期 切片读取。
"""
from __future__ import annotations

import gc
import os
import sqlite3
from glob import glob
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from logger import setup_logger

log = setup_logger("trade_pipeline")

def _project_root() -> str:
    """发行包根目录（含 config.yaml / v2 / backtest），不依赖启动时 cwd。"""
    here = os.path.dirname(os.path.abspath(__file__))  # .../backtest
    root = os.path.dirname(here)  # package root
    # 若从源码树移动，向上找带 config.json 的目录
    probe = root
    for _ in range(3):
        if os.path.isfile(os.path.join(probe, "config.yaml")) or os.path.isfile(os.path.join(probe, "config.json")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return root


_ROOT = _project_root()
RAW_ROOT = os.path.join(_ROOT, "data", "okx_history", "raw")
PQ_ROOT = os.path.join(_ROOT, "data", "okx_history", "parquet")
DB_PATH = os.path.join(_ROOT, "data", "okx_history", "bars.db")

DEFAULT_CHUNK = 80_000

# 用户可选的常用 K 线周期（构建时全部拟合进库）
COMMON_BARS: List[str] = ["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "1D"]

# bar 名 → pandas floor 规则
BAR_FREQ: Dict[str, str] = {
    "1s": "1s",
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "1D": "1D",
}

# 文件名映射
BAR_FILE: Dict[str, str] = {b: f"bars_{b}.parquet" for b in COMMON_BARS}
BAR_FILE["1s"] = "bars_1s.parquet"


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    ts_c = _find_col(df, ["timestamp", "ts", "created_time", "time", "datetime", "trade_time"])
    px_c = _find_col(df, ["price", "px", "trade_price"])
    sz_c = _find_col(df, ["size", "sz", "quantity", "qty", "amount", "trade_size"])
    side_c = _find_col(df, ["side", "side_code", "direction"])
    if not ts_c or not px_c or not sz_c:
        return pd.DataFrame(columns=["ts", "price", "size", "side"])
    out = pd.DataFrame({
        "ts": df[ts_c],
        "price": pd.to_numeric(df[px_c], errors="coerce"),
        "size": pd.to_numeric(df[sz_c], errors="coerce"),
    })
    if side_c:
        side = df[side_c].astype(str).str.lower()
        out["side"] = np.where(side.str.contains("buy|b|^1$"), "buy", "sell")
    else:
        out["side"] = "unknown"
    out = out.dropna(subset=["ts", "price", "size"])
    out = out[(out["price"] > 0) & (out["size"] > 0)]
    if out.empty:
        return out
    if np.issubdtype(out["ts"].dtype, np.number):
        sample = float(out["ts"].iloc[0])
        unit = "ms" if sample > 1e12 else ("s" if sample > 1e9 else "ms")
        out["ts"] = pd.to_datetime(out["ts"], unit=unit, utc=True).dt.tz_convert(None)
    else:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["ts"])
    return out


def _write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        alt = path.replace(".parquet", ".pkl")
        df.to_pickle(alt)
        log.info(f"写入 {alt} rows={len(df)}")
        return
    log.info(f"写入 {path} rows={len(df)}")


def _read_table(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    alt = path.replace(".parquet", ".pkl")
    if os.path.exists(alt):
        return pd.read_pickle(alt)
    raise FileNotFoundError(path)


def _path_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(path.replace(".parquet", ".pkl"))


def list_raw_files(symbol: str, raw_dir: Optional[str] = None) -> List[str]:
    raw_dir = raw_dir or os.path.join(RAW_ROOT, symbol)
    if not os.path.isdir(raw_dir):
        return []
    files = sorted(
        set(
            glob(os.path.join(raw_dir, "*.csv"))
            + glob(os.path.join(raw_dir, "*.csv.gz"))
            + glob(os.path.join(raw_dir, "*.CSV"))
        )
    )
    seen = set()
    uniq = []
    for fp in files:
        base = os.path.basename(fp).lower()
        if base in seen:
            continue
        seen.add(base)
        uniq.append(fp)
    return uniq


def iter_trade_chunks(
    symbol: str,
    raw_dir: Optional[str] = None,
    chunksize: int = DEFAULT_CHUNK,
) -> Iterator[pd.DataFrame]:
    """逐文件、逐块产生标准化成交，内存只保留一块。"""
    raw_dir = raw_dir or os.path.join(RAW_ROOT, symbol)
    uniq = list_raw_files(symbol, raw_dir)
    if not uniq:
        raise FileNotFoundError(
            f"未找到 {raw_dir}\n请先 download 或手动放入 Trade history CSV。"
        )

    for fp in uniq:
        log.info(f"流式读取 {os.path.basename(fp)}")
        try:
            reader = pd.read_csv(fp, chunksize=chunksize, low_memory=True)
        except Exception as e:
            log.warning(f"无法读取 {fp}: {e}")
            continue
        for i, chunk in enumerate(reader):
            norm = _normalize_chunk(chunk)
            if norm.empty:
                continue
            yield norm
            if (i + 1) % 20 == 0:
                gc.collect()
        del reader
        gc.collect()


def _merge_bar_maps(acc: Dict[pd.Timestamp, dict], chunk: pd.DataFrame, freq: str) -> None:
    """把一块成交合并进 bar 累加器（只存每个 bucket 的 OHLCV，不存 tick）。"""
    if chunk.empty:
        return
    rule = BAR_FREQ.get(freq, freq)
    ts = chunk["ts"].dt.floor(rule)
    px = chunk["price"].values
    sz = chunk["size"].values
    side = chunk["side"].values
    tmp = pd.DataFrame({"bucket": ts, "price": px, "size": sz, "side": side})
    for bucket, g in tmp.groupby("bucket", sort=False):
        prices = g["price"].values
        sizes = g["size"].values
        buy = float(sizes[g["side"].values == "buy"].sum())
        sell = float(sizes[g["side"].values == "sell"].sum())
        o, h, l, c = float(prices[0]), float(prices.max()), float(prices.min()), float(prices[-1])
        v = float(sizes.sum())
        if bucket in acc:
            a = acc[bucket]
            a["high"] = max(a["high"], h)
            a["low"] = min(a["low"], l)
            a["close"] = c
            a["vol"] += v
            a["buy_vol"] += buy
            a["sell_vol"] += sell
        else:
            acc[bucket] = {
                "open": o, "high": h, "low": l, "close": c,
                "vol": v, "buy_vol": buy, "sell_vol": sell,
            }


def _acc_to_df(acc: Dict[pd.Timestamp, dict]) -> pd.DataFrame:
    if not acc:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "vol", "buy_vol", "sell_vol", "cvd"]
        )
    rows = []
    for ts in sorted(acc.keys()):
        a = acc[ts]
        rows.append({
            "ts": ts,
            "open": a["open"], "high": a["high"], "low": a["low"], "close": a["close"],
            "vol": a["vol"], "buy_vol": a["buy_vol"], "sell_vol": a["sell_vol"],
        })
    df = pd.DataFrame(rows)
    df["cvd"] = (df["buy_vol"] - df["sell_vol"]).cumsum()
    return df


# ---------------------------------------------------------------------------
# SQLite 统一库
# ---------------------------------------------------------------------------

def _ensure_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol   TEXT NOT NULL,
            bar      TEXT NOT NULL,
            ts       INTEGER NOT NULL,
            open     REAL,
            high     REAL,
            low      REAL,
            close    REAL,
            vol      REAL,
            buy_vol  REAL,
            sell_vol REAL,
            cvd      REAL,
            PRIMARY KEY (symbol, bar, ts)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_bar_ts ON ohlcv(symbol, bar, ts)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            symbol    TEXT NOT NULL,
            bar       TEXT NOT NULL,
            n_bars    INTEGER,
            ts_min    INTEGER,
            ts_max    INTEGER,
            built_at  TEXT,
            PRIMARY KEY (symbol, bar)
        )
        """
    )
    conn.commit()
    return conn


def write_bars_to_db(
    symbol: str,
    bar: str,
    df: pd.DataFrame,
    db_path: str = DB_PATH,
    replace_all: bool = False,
) -> int:
    """把某品种某周期 K 线写入 SQLite。

    - 默认增量友好：INSERT OR REPLACE（PRIMARY KEY 防重，不先清空）
    - replace_all=True：先 DELETE 该 symbol+bar 再写入（全量重建）
    """
    if df is None or df.empty:
        return 0
    conn = _ensure_db(db_path)
    try:
        if replace_all:
            conn.execute("DELETE FROM ohlcv WHERE symbol=? AND bar=?", (symbol, bar))
        rows = []
        for r in df.itertuples(index=False):
            ts = pd.Timestamp(r.ts)
            ts_i = int(ts.timestamp())
            rows.append((
                symbol, bar, ts_i,
                float(r.open), float(r.high), float(r.low), float(r.close),
                float(r.vol),
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
        # meta 以库内真实 MIN/MAX/COUNT 为准，避免增量写入后范围错误
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


def load_bars_from_db(
    symbol: str,
    bar: str = "1m",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """从 SQLite 按日期区间读取 K 线。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"缺少数据库 {db_path}，请先 build")
    conn = sqlite3.connect(db_path)
    try:
        q = "SELECT ts,open,high,low,close,vol,buy_vol,sell_vol,cvd FROM ohlcv WHERE symbol=? AND bar=?"
        params: list = [symbol, bar]
        if start is not None:
            q += " AND ts>=?"
            params.append(int(pd.Timestamp(start).timestamp()))
        if end is not None:
            q += " AND ts<=?"
            params.append(int(pd.Timestamp(end).timestamp()))
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, conn, params=params)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
        return df
    finally:
        conn.close()


def _norm_bar(bar: str) -> str:
    aliases = {
        "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
        "60m": "1H", "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W",
    }
    b = (bar or "").strip()
    return aliases.get(b, b)


def db_has_symbol_bar(symbol: str, bar: str, db_path: str = DB_PATH) -> bool:
    """库内是否有该品种+周期（meta 优先，ohlcv 兜底）。"""
    db_path = db_path or DB_PATH
    if not os.path.isfile(db_path):
        return False
    bar = _norm_bar(bar)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT n_bars FROM meta WHERE symbol=? AND bar=?", (symbol, bar)
        )
        row = cur.fetchone()
        if row and row[0] and int(row[0]) > 10:
            return True
        # meta 缺失/过期时直接数 ohlcv
        cur = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND bar=?", (symbol, bar)
        )
        n = int((cur.fetchone() or [0])[0] or 0)
        return n > 10
    except Exception:
        return False
    finally:
        conn.close()


def list_db_coverage(db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["symbol", "bar", "n_bars", "ts_min", "ts_max"])
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM meta ORDER BY symbol, bar", conn)
        if not df.empty:
            df["ts_min"] = pd.to_datetime(df["ts_min"], unit="s")
            df["ts_max"] = pd.to_datetime(df["ts_max"], unit="s")
        return df
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------

def cleanup_raw_files(symbol: str, raw_dir: Optional[str] = None) -> Tuple[int, float]:
    """
    删除 raw 目录下的大体积原文件（csv / zip / gz）。
    返回 (删除文件数, 释放 MB)。
    """
    raw_dir = raw_dir or os.path.join(RAW_ROOT, symbol)
    if not os.path.isdir(raw_dir):
        return 0, 0.0
    patterns = ["*.csv", "*.CSV", "*.csv.gz", "*.zip", "*.ZIP", "*.gz"]
    removed = 0
    bytes_freed = 0
    seen = set()
    for pat in patterns:
        for fp in glob(os.path.join(raw_dir, pat)):
            if fp in seen:
                continue
            seen.add(fp)
            try:
                sz = os.path.getsize(fp)
                os.remove(fp)
                removed += 1
                bytes_freed += sz
                log.info(f"已删除原文件 {os.path.basename(fp)} ({sz / 1e6:.1f} MB)")
            except OSError as e:
                log.warning(f"删除失败 {fp}: {e}")
    # 若目录空了也清掉空目录
    try:
        if os.path.isdir(raw_dir) and not os.listdir(raw_dir):
            os.rmdir(raw_dir)
    except OSError:
        pass
    mb = bytes_freed / 1e6
    log.info(f"{symbol} 清理完成: 删除 {removed} 个文件，释放约 {mb:.1f} MB")
    return removed, mb


def build_symbol_parquet(
    symbol: str,
    raw_dir: Optional[str] = None,
    out_root: Optional[str] = None,
    chunksize: int = DEFAULT_CHUNK,
    save_trades: bool = False,
    trade_part_rows: int = 500_000,
    bars: Optional[Sequence[str]] = None,
    write_db: bool = True,
    db_path: str = DB_PATH,
    cleanup_raw: bool = True,
) -> str:
    """
    流式：CSV chunks → 一次聚合全部常用周期 → parquet + SQLite。
    cleanup_raw=True 时处理成功后删除 raw 下 csv/zip。
    """
    from backtest.progress import ProgressBar

    bars = list(bars) if bars else list(COMMON_BARS)
    for b in bars:
        if b not in BAR_FREQ:
            raise ValueError(f"不支持的周期 {b}，可选: {list(BAR_FREQ.keys())}")

    out_root = out_root or PQ_ROOT
    out_dir = os.path.join(out_root, symbol)
    os.makedirs(out_dir, exist_ok=True)

    accs: Dict[str, Dict[pd.Timestamp, dict]] = {b: {} for b in bars}

    trade_buf: List[pd.DataFrame] = []
    trade_buf_rows = 0
    part_i = 0
    trades_dir = os.path.join(out_dir, "trades")
    total_ticks = 0

    if save_trades:
        os.makedirs(trades_dir, exist_ok=True)

    # 估算进度：按文件数
    files = list_raw_files(symbol, raw_dir)
    n_files = max(len(files), 1)
    pbar = ProgressBar(total=n_files, desc=f"build {symbol}")
    file_done = 0

    for chunk in iter_trade_chunks(symbol, raw_dir=raw_dir, chunksize=chunksize):
        n = len(chunk)
        total_ticks += n
        for b in bars:
            _merge_bar_maps(accs[b], chunk, b)

        if save_trades:
            trade_buf.append(chunk)
            trade_buf_rows += n
            if trade_buf_rows >= trade_part_rows:
                part = pd.concat(trade_buf, ignore_index=True)
                part = part.sort_values("ts")
                part_path = os.path.join(trades_dir, f"part_{part_i:04d}.pkl")
                part.to_pickle(part_path)
                log.info(f"成交分卷 {part_path} rows={len(part)}")
                part_i += 1
                trade_buf.clear()
                trade_buf_rows = 0
                del part
                gc.collect()

        del chunk
        if total_ticks % (chunksize * 20) == 0:
            gc.collect()
            # 粗估：每处理若干 tick 推进一步（文件级更准靠 log 文件切换）
            buckets_info = " ".join(f"{b}={len(accs[b])}" for b in bars[:4])
            pbar.set(
                min(file_done, n_files - 1),
                suffix=f"ticks≈{total_ticks // 1000}k {buckets_info}",
            )

    # 文件级进度在 iter 内无法精确，结束时拉满
    pbar.set(n_files, suffix=f"ticks≈{total_ticks // 1000}k done")

    if save_trades and trade_buf:
        part = pd.concat(trade_buf, ignore_index=True)
        part_path = os.path.join(trades_dir, f"part_{part_i:04d}.pkl")
        part.to_pickle(part_path)
        log.info(f"成交分卷 {part_path} rows={len(part)}")
        del part, trade_buf
        gc.collect()

    bucket_summary = " ".join(f"{b}={len(accs[b])}" for b in bars)
    log.info(f"{symbol} 流式处理 tick≈{total_ticks} | buckets {bucket_summary}")

    for b in bars:
        df = _acc_to_df(accs[b])
        fname = BAR_FILE.get(b, f"bars_{b}.parquet")
        _write_table(df, os.path.join(out_dir, fname))
        if write_db and not df.empty:
            n_w = write_bars_to_db(symbol, b, df, db_path=db_path)
            log.info(f"DB 写入 {symbol} {b} rows={n_w}")
        del df
        del accs[b]
        gc.collect()

    marker = os.path.join(out_dir, "BUILD_OK.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"ticks≈{total_ticks}\nbars={','.join(bars)}\n")

    if cleanup_raw:
        cleanup_raw_files(symbol, raw_dir=raw_dir)

    return out_dir


def build_all(
    symbols: List[str],
    chunksize: int = DEFAULT_CHUNK,
    save_trades: bool = False,
    bars: Optional[Sequence[str]] = None,
    write_db: bool = True,
    cleanup_raw: bool = True,
) -> None:
    for i, s in enumerate(symbols):
        log.info(f"==== 构建 [{i + 1}/{len(symbols)}] {s} ====")
        try:
            build_symbol_parquet(
                s,
                chunksize=chunksize,
                save_trades=save_trades,
                bars=bars,
                write_db=write_db,
                cleanup_raw=cleanup_raw,
            )
        except Exception as e:
            log.error(f"{s} 构建失败: {e}")


def load_bars(
    symbol: str,
    bar: str = "1m",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    prefer_db: bool = True,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    读取 K 线：优先 SQLite（支持日期切片），否则回退 parquet/pkl。
    bar 支持 COMMON_BARS 全部周期。
    """
    bar = _norm_bar(bar)

    if prefer_db and os.path.exists(db_path) and db_has_symbol_bar(symbol, bar, db_path):
        return load_bars_from_db(symbol, bar, start=start, end=end, db_path=db_path)

    name = BAR_FILE.get(bar, f"bars_{bar}.parquet")
    path = os.path.join(PQ_ROOT, symbol, name)
    if not _path_exists(path):
        raise FileNotFoundError(
            f"缺少 {path}（且 DB 无 {symbol}/{bar}）\n"
            f"请先: python tools/data_manager.py --symbols {symbol} --bars {bar} --days 90"
        )
    df = _read_table(path)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def load_trades_iter(symbol: str, part_dir: Optional[str] = None) -> Iterator[pd.DataFrame]:
    """分卷读取成交（若 build 时 save_trades=True）。"""
    part_dir = part_dir or os.path.join(PQ_ROOT, symbol, "trades")
    if not os.path.isdir(part_dir):
        raise FileNotFoundError(f"无成交分卷目录 {part_dir}（默认构建不保存全量 tick）")
    parts = sorted(glob(os.path.join(part_dir, "part_*.pkl")))
    for fp in parts:
        df = pd.read_pickle(fp)
        yield df
        del df
        gc.collect()


def load_trade_csvs(symbol: str, raw_dir: Optional[str] = None) -> pd.DataFrame:
    """警告：会合并全部 tick，仅小样本调试使用。"""
    log.warning("load_trade_csvs 会占用大量内存，请优先用 build_symbol_parquet 流式构建")
    parts = list(iter_trade_chunks(symbol, raw_dir=raw_dir))
    if not parts:
        return pd.DataFrame(columns=["ts", "price", "size", "side"])
    return pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates()


def trades_to_bars(trades: pd.DataFrame, freq: str = "1m") -> pd.DataFrame:
    """freq 可为 bar 名（1m/5m/…）或 pandas rule（1min/5min）。"""
    acc: Dict[pd.Timestamp, dict] = {}
    key = freq if freq in BAR_FREQ else freq
    # 若传入 1min 等 rule，直接作为 floor 用
    if key not in BAR_FREQ:
        BAR_FREQ_TMP = key  # noqa — _merge uses BAR_FREQ.get(freq, freq)
        _merge_bar_maps(acc, trades, BAR_FREQ_TMP)
    else:
        _merge_bar_maps(acc, trades, key)
    return _acc_to_df(acc)
