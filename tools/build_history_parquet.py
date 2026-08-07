# -*- coding: utf-8 -*-
"""
[兼容入口] v3 起请使用统一工具：

  python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1H,4H --days 90
  python tools/data_manager.py --status

本文件保留为薄包装，转发到 data_manager（直接下载各周期 K 线并入库，
不再从成交 tick 聚合）。
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

log = setup_logger("build_history")


def main():
    ap = argparse.ArgumentParser(
        description="[兼容] 转发到 tools/data_manager.py"
    )
    ap.add_argument("--symbol", default="")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--bars", default="1m,5m,15m,1H,4H")
    ap.add_argument("--days", type=float, default=90)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--list-db", action="store_true")
    ap.add_argument("--db", default=DB_PATH_DEFAULT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chunksize", type=int, default=0)
    ap.add_argument("--save-trades", action="store_true")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    db = args.db
    if db and not os.path.isabs(db):
        db = os.path.join(ROOT, db)

    if args.list_db:
        print_status(db)
        return

    syms = []
    if args.symbol:
        syms.append(args.symbol.strip())
    if args.symbols:
        syms.extend([s.strip() for s in args.symbols.split(",") if s.strip()])
    if not syms:
        log.error("请指定 --symbol 或 --symbols")
        sys.exit(1)

    argv = [
        "data_manager.py",
        "--symbols", ",".join(syms),
        "--bars", args.bars,
        "--db", db,
        "--days", str(args.days),
    ]
    if args.start:
        argv.extend(["--start", args.start])
    if args.end:
        argv.extend(["--end", args.end])
    if args.force:
        argv.append("--force")
    if args.no_db:
        argv.append("--validate-only")

    log.info("转发 → tools/data_manager.py " + " ".join(argv[1:]))
    sys.argv = argv
    from tools.data_manager import main as dm_main
    dm_main()


if __name__ == "__main__":
    main()
