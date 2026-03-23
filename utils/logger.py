"""
Logging setup utility.
"""
import logging
import os
import sys
from typing import Dict

def setup_logger(config: Dict) -> logging.Logger:
    """
    Configure and return a logger based on config dict.
    Expected keys: level, file (optional)
    """
    level = getattr(logging, config.get("level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger("AI_Cover_Generator")
    logger.setLevel(level)
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if specified
    log_file = config.get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger