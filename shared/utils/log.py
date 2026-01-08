# Logging Utility module
# -----------------------
# Purpose:
# - Centralized logging setup for all modules (fetchers, evaluator, etc.).
# - Provides consistent formatting and log level configuration.

# Future Enhancements:
# - Add file logging, JSON logging, or cloud log integration.
# - Add support for log context (e.g., job ID, record ID).

import logging


def setup_logger(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a configured logger for a module.

    Args:
        module_name (str): The name of the module (typically __name__)
        level (int): The logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(level=level)
    logger = logging.getLogger(module_name)
    return logger
