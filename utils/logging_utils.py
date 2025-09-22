import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import time


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Configures logging for the application.

    Args:
        config (Optional[Dict[str, Any]]): Logging configuration dictionary.
            Expected keys:
            - 'level' (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING'). Defaults to 'INFO'.
            - 'format' (str): Logging format string. Defaults to a standard format.
            - 'datefmt' (str): Date format string. Defaults to ISO 8601 format.
            - 'log_file' (Optional[str]): Path to a log file. If provided, logs will also be written here.
                                           Path can be relative to project root defined in base config, or absolute.
            - 'log_dir' (Optional[str]): Directory to store log files if log_file is just a name.
    """
    if config is None:
        config = {}

    log_level = config.get("level", "INFO").upper()
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_fmt = config.get("datefmt", "%Y-%m-%d %H:%M:%S")

    root_logger = logging.getLogger()

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(log_format, datefmt=date_fmt)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    log_file_path = config.get("log_file")
    log_dir = config.get("log_dir")

    if log_file_path:
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            if not Path(log_file_path).is_absolute() and "/" not in log_file_path and "\\" not in log_file_path:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_filename = f"{Path(log_file_path).stem}_{timestamp}.log"
                full_log_path = log_dir_path / log_filename
            else:
                full_log_path = Path(log_file_path)
                full_log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            full_log_path = Path(log_file_path)
            full_log_path.parent.mkdir(parents=True, exist_ok=True)

        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == full_log_path.resolve() for h in root_logger.handlers):
            try:
                file_handler = logging.FileHandler(full_log_path, mode="a")
                file_formatter = logging.Formatter(log_format, datefmt=date_fmt)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                logging.info("Logging to file: %s", full_log_path)
            except Exception as e:
                logging.error("Failed to create log file handler for %s: %s", full_log_path, e, exc_info=True)
        else:
            logging.debug("File handler for %s already exists.", full_log_path)

    try:
        level = getattr(logging, log_level, logging.INFO)
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        logging.info("Logging level set to: %s", log_level)
    except AttributeError:
        logging.error("Invalid logging level: %s. Defaulting to INFO.", log_level)
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers:
            handler.setLevel(logging.INFO)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)