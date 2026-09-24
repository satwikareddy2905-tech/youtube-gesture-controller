"""Logging module for the YouTube Gesture Controller.

Provides a centralized logging configuration that outputs to both the console
and a rotating log file. Automatically handles log directory creation and
supports debug-level configurations from the main settings.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Flag to guarantee logging setup runs exactly once
_logging_configured = False


def setup_logging(
    log_dir: str = "logs",
    log_file_name: str = "app.log",
    debug: bool = False,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> None:
    """Configures the root logger with console and rotating file handlers.

    Args:
        log_dir: Directory where log files will be saved.
        log_file_name: Name of the active log file.
        debug: If True, sets level to DEBUG. Otherwise, defaults to INFO.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of rotated log files to retain.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Ensure log directory exists
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Fallback to stdout only if directory cannot be created
        print(f"Warning: Failed to create log directory '{log_dir}': {e}. Logging to console only.")

    # Select logging level
    level = logging.DEBUG if debug else logging.INFO

    # Format: Timestamp - Log Level - Module Name - Message
    log_format = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    formatter = logging.Formatter(log_format)

    # Retrieve and configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clean existing handlers to prevent duplicates (e.g. during re-imports)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    if log_path.exists():
        file_path = log_path / log_file_name
        try:
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Failed to setup file logger at '{file_path}': {e}. Logging to console only.")

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Factory function to retrieve a configured logger for a specific module.

    Initializes root logging config automatically if it has not been configured.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    global _logging_configured
    if not _logging_configured:
        try:
            # Lazy import to avoid circular dependency issues
            from config import AppConfig
            config = AppConfig()
            debug_mode = config.DEBUG
        except Exception:
            debug_mode = False
        setup_logging(debug=debug_mode)

    return logging.getLogger(name)
