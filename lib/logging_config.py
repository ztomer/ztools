"""
Logging Configuration for ZTools
Provides structured logging with optional file output.
"""

import logging
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

    return logger


# Module-level loggers for common use
lib_logger = get_logger("lib", level="INFO")
osaurus_logger = get_logger("lib.osaurus", level="INFO")
mlx_logger = get_logger("lib.mlx", level="INFO")
validators_logger = get_logger("lib.validators", level="INFO")
content_logger = get_logger("lib.content", level="INFO")
