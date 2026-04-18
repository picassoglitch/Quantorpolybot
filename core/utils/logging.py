"""Loguru wrapper. Console + rotating file + JSONL audit trail."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from core.utils.config import get_config, root_dir

_INITIALISED = False


def setup_logging() -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    cfg = get_config()
    log_dir = Path(cfg.get("log_dir", default="logs"))
    if not log_dir.is_absolute():
        log_dir = root_dir() / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level:<7}</level> "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_dir / "polybot.log",
        level=level,
        rotation="20 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_dir / "audit.jsonl",
        level="INFO",
        rotation="50 MB",
        retention="30 days",
        serialize=True,
        enqueue=True,
        filter=lambda record: record["extra"].get("audit") is True,
    )
    _INITIALISED = True


def audit(event: str, **fields: object) -> None:
    """Structured audit log line. Always written to audit.jsonl."""
    logger.bind(audit=True).info(event, **fields)
