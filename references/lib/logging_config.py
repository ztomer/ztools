"""
Logging Configuration for ZTools
Provides structured logging with optional file output.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write logs to file
        console_output: Whether to output to console (default True)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting process")
    """
    logger = logging.getLogger(name)

    # Only configure if not already done
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Define a central log file under the user cache dir (never under the install
# prefix, which is read-only when installed via Homebrew/uv into a Cellar).
def _log_file() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ztools" / "debug.log"


LOG_FILE = _log_file()
DEBUG_MODE = "--debug" in sys.argv


# Module-level loggers for common use. The file handler is only attached in
# debug mode so a normal run never touches disk under a read-only install.
def _make_logger(name: str) -> logging.Logger:
    log_file = LOG_FILE if DEBUG_MODE else None
    return get_logger(
        name,
        level="DEBUG" if DEBUG_MODE else "INFO",
        log_file=log_file,
        console_output=DEBUG_MODE,
    )


lib_logger = _make_logger("lib")
osaurus_logger = _make_logger("lib.osaurus")
mlx_logger = _make_logger("lib.mlx")
validators_logger = _make_logger("lib.validators")
content_logger = _make_logger("lib.content")
