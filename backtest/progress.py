# -*- coding: utf-8 -*-
"""轻量进度条（不依赖 tqdm），build / 回测共用。"""
from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 28,
        min_interval: float = 0.25,
    ):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.min_interval = min_interval
        self.n = 0
        self._t0 = time.time()
        self._last_print = 0.0
        self._closed = False

    def update(self, n: int = 1, suffix: str = "") -> None:
        self.n = min(self.n + n, self.total)
        now = time.time()
        if self.n < self.total and (now - self._last_print) < self.min_interval:
            return
        self._last_print = now
        self._render(suffix)

    def set(self, n: int, suffix: str = "") -> None:
        self.n = max(0, min(int(n), self.total))
        now = time.time()
        if self.n < self.total and (now - self._last_print) < self.min_interval:
            return
        self._last_print = now
        self._render(suffix)

    def _render(self, suffix: str = "") -> None:
        ratio = self.n / self.total
        filled = int(self.width * ratio)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = max(time.time() - self._t0, 1e-6)
        rate = self.n / elapsed
        eta = (self.total - self.n) / rate if rate > 1e-9 else 0.0
        msg = (
            f"\r{self.desc} |{bar}| {self.n}/{self.total} "
            f"({ratio * 100:5.1f}%) {rate:.1f}/s ETA {eta:.0f}s"
        )
        if suffix:
            msg += f" {suffix}"
        sys.stdout.write(msg[:120])
        sys.stdout.flush()
        if self.n >= self.total:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sys.stdout.write("\n")
        sys.stdout.flush()


def progress_log(log, i: int, total: int, every: int = 50, prefix: str = "") -> None:
    """每隔 every 步打一条日志进度（适合无 stdout 的环境）。"""
    if total <= 0:
        return
    if i == 0 or i >= total - 1 or (i + 1) % max(1, every) == 0:
        pct = 100.0 * (i + 1) / total
        log.info(f"{prefix}进度 {i + 1}/{total} ({pct:.1f}%)")
