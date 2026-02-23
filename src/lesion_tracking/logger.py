import logging
from time import perf_counter
from typing import Callable, Literal

from rich.console import Console
from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Shared console instance for coordinated Rich output
console = Console(stderr=True)


def setup_logging(level: LogLevel = "INFO") -> None:
    """Configure the root logger and force all existing loggers to propagate."""
    handler = RichHandler(
        console=console,
        show_path=True,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Force all existing loggers to propagate through root
    # (strips handlers added by libraries like Lightning)
    for name in logging.root.manager.loggerDict:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True

    logging.captureWarnings(True)


def get_logger(name: str, level: LogLevel = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
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
