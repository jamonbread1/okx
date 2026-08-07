# -*- coding: utf-8 -*-
"""
[兼容入口] v3 起请使用统一工具：

  python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1H,4H --days 90
  python tools/data_manager.py --update --symbols BTC-USDT-SWAP
  python tools/data_manager.py --status
  python tools/data_manager.py --funding --symbols BTC-USDT-SWAP

本文件保留为薄包装，转发到 data_manager / funding_store。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logger import setup_logger
from tools.data_manager import DB_PATH_DEFAULT, print_status

log = setup_logger("download_history")


def main():
    ap = argparse.ArgumentParser(
        description="[兼容] 转发到 tools/data_manager.py；推荐直接使用 data_manager"
    )
    ap.add_argument("--symbols", default="BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP")
    ap.add_argument("--bars", default="1m,5m,15m,1H,4H")
    ap.add_argument("--days", type=float, default=90)
    ap.add_argument("--months", type=int, default=0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--update-fee", action="store_true")
    ap.add_argument("--funding-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--db", default=DB_PATH_DEFAULT)
    args = ap.parse_args()

    db = args.db
    if db and not os.path.isabs(db):
        db = os.path.join(ROOT, db)

    if args.list or args.status:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        print_status(db, symbols)
        return

    if args.funding_only or args.update_fee:
        from backtest.funding_store import build_funding_for_symbols
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        build_funding_for_symbols(symbols, db_path=db, incremental=not args.force)
        return

    days = float(args.days)
    if args.months and args.months > 0:
        days = max(days, args.months * 30)

    argv = [
        "data_manager.py",
        "--symbols", args.symbols,
        "--bars", args.bars,
        "--db", db,
    ]
    if args.update:
        argv.append("--update")
    if args.force:
        argv.append("--force")
    if args.start:
        argv.extend(["--start", args.start])
    if args.end:
        argv.extend(["--end", args.end])
    if days and not args.update:
        argv.extend(["--days", str(days)])
    if args.update_fee:
        argv.append("--funding")

    log.info("转发 → tools/data_manager.py " + " ".join(argv[1:]))
    sys.argv = argv
    from tools.data_manager import main as dm_main
    dm_main()


if __name__ == "__main__":
    main()
