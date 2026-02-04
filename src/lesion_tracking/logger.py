import logging
from time import perf_counter
from typing import Callable, Literal

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


def track_runtime(logger: logging.Logger | None, buffer: list):
    def decorator(f: Callable):
        def f_timed(*args, **kwargs):
            start = perf_counter()
            outputs = f(*args, **kwargs)
            elapsed = perf_counter() - start  # seconds
            if logger:
                logger.info(f"Callable {f.__name__} executed in {elapsed:.4f} seconds")
            buffer.append(elapsed)
            return outputs

        return f_timed

    return decorator
