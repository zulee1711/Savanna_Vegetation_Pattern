from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    base_dir: str | Path = "logs",
    run_date: str | None = None,
    log_name: str = "full_run.log",
    level: int = logging.INFO,
) -> Path:
    """
    Create one logging file per chosen day.

    Example output:
        logs/20260604/full_run.log
    """

    if run_date is None:
        run_date = datetime.now().strftime("%Y%m%d")

    log_dir = Path(base_dir) / run_date
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_name

    # Clear existing handlers so repeated setup does not duplicate logs
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.info("Logging started")
    logging.info("Log file: %s", log_path)

    return log_path