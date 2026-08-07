# -*- coding: utf-8 -*-
"""[兼容入口] 资金费率下载。推荐: python tools/data_manager.py --funding --symbols ..."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.funding_store import DB_PATH, build_funding_for_symbols, load_funding_series
from logger import setup_logger

log = setup_logger("build_funding")


def main():
    ap = argparse.ArgumentParser(
        description="[兼容] 仅下载资金费率；推荐: python tools/data_manager.py --funding"
    )
    ap.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP")
    ap.add_argument("--max-pages", type=int, default=300)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="全量重下（非增量）")
    args = ap.parse_args()

    db = args.db
    if db and not os.path.isabs(db):
        db = os.path.join(ROOT, db)

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.list:
        for s in syms:
            df = load_funding_series(s, db_path=db)
            print(s, "EMPTY" if df.empty else f"{len(df)} {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
        return

    log.info(f"下载资金费率 → {os.path.abspath(db)}")
    build_funding_for_symbols(
        syms, db_path=db, max_pages=args.max_pages, incremental=not args.force
    )


if __name__ == "__main__":
    main()
