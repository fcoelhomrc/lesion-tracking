import logging
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Shared console instance for coordinated Rich output
console = Console(stderr=True)


def get_logger(
    name: str,
    level: LogLevel = "INFO",
    *,
    show_path: bool = True,
    show_time: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = RichHandler(
        console=console,
        show_path=show_path,
        show_time=show_time,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger
