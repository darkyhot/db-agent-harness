"""Dedicated logger for deep analysis pipeline.

All records go both to the shared agent.log (for unified debugging) and to a
per-run deep_analysis.log inside the run's output directory. This makes it easy
to diff runs and share a single log when reporting bugs.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "deep_analysis"


def get_logger() -> logging.Logger:
    """Get the shared deep_analysis logger. Idempotent."""
    return logging.getLogger(_LOGGER_NAME)


def attach_run_log(output_dir: Path) -> logging.FileHandler:
    """Attach a per-run log file to the deep_analysis logger.

    Returns the handler so the caller can detach it when the run ends, keeping
    the shared logger clean between runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "deep_analysis.log"
    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return handler


def detach_run_log(handler: logging.FileHandler) -> None:
    logger = get_logger()
    logger.removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass
