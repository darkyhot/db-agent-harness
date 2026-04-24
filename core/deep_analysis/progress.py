"""Progress reporter — thin adapter over the CLI status line.

The CLI's status_print (cli/interface.py) rewrites a single terminal line. Here
we accept a callback with the same signature so the pipeline can emit progress
without importing CLI internals, which keeps the module testable headless.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from core.deep_analysis.logging_setup import get_logger

ProgressCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


class ProgressReporter:
    """Emits status updates and mirrors them to the logger."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._cb = callback or _noop
        self._logger = get_logger()
        self._started_at = time.time()
        self._current_stage = ""
        self._total_stages = 0
        self._stage_index = 0

    def set_stages(self, total: int) -> None:
        self._total_stages = total
        self._stage_index = 0

    def stage(self, name: str) -> None:
        """Enter a top-level stage (profiling / hypothesis generation / ...)."""
        self._stage_index += 1
        self._current_stage = name
        elapsed = int(time.time() - self._started_at)
        prefix = f"[{self._stage_index}/{self._total_stages}] " if self._total_stages else ""
        msg = f"{prefix}{name} ({elapsed}s)"
        self._cb(msg)
        self._logger.info("STAGE %s", name)

    def sub(self, message: str) -> None:
        """Fine-grained sub-stage message inside the current stage."""
        elapsed = int(time.time() - self._started_at)
        stage = f"{self._current_stage} — " if self._current_stage else ""
        self._cb(f"{stage}{message} ({elapsed}s)")
        self._logger.debug("SUB %s | %s", self._current_stage, message)

    def hypothesis(self, idx: int, total: int, title: str) -> None:
        """Per-hypothesis progress line."""
        elapsed = int(time.time() - self._started_at)
        short = title if len(title) <= 80 else title[:77] + "..."
        self._cb(f"Гипотеза [{idx}/{total}] {short} ({elapsed}s)")
        self._logger.info("HYPOTHESIS %d/%d %s", idx, total, title)

    def warn(self, message: str) -> None:
        self._logger.warning(message)

    def clear(self) -> None:
        self._cb("")
