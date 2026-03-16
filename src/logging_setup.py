from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(*, enable: bool, level: str, log_to_file: bool, logs_dir: Path) -> None:
    """Enables logging for the project to run so output can be stored in a file rather than relying on terminal printing for debugging.

    Args:
        enable (bool): Whether to use logging.
        level (str): The level to allow logging for. Is a minimum level, so if sett to level.ERROR, level.WARNING would not log.
        log_to_file (bool): Whether to save output to file.
        logs_dir (Path): The path to save output file to.
    """
    if not enable:
        logging.disable(logging.CRITICAL)
        return

    logs_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_to_file:
        logfile = logs_dir / "rag.log"
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
    )
