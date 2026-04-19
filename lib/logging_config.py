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

    logger.propagate = False
    return logger


# Define a central log file in the project root
LOG_FILE = Path(__file__).parent.parent / "logs" / "debug.log"
DEBUG_MODE = "--debug" in sys.argv

# Module-level loggers for common use
lib_logger = get_logger("lib", level="DEBUG" if DEBUG_MODE else "INFO", log_file=LOG_FILE, console_output=DEBUG_MODE)
osaurus_logger = get_logger("lib.osaurus", level="DEBUG" if DEBUG_MODE else "INFO", log_file=LOG_FILE, console_output=DEBUG_MODE)
mlx_logger = get_logger("lib.mlx", level="DEBUG" if DEBUG_MODE else "INFO", log_file=LOG_FILE, console_output=DEBUG_MODE)
validators_logger = get_logger("lib.validators", level="DEBUG" if DEBUG_MODE else "INFO", log_file=LOG_FILE, console_output=DEBUG_MODE)
content_logger = get_logger("lib.content", level="DEBUG" if DEBUG_MODE else "INFO", log_file=LOG_FILE, console_output=DEBUG_MODE)
