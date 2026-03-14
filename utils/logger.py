"""
utils/logger.py
---------------
Simple coloured console logger so terminal output is easy to read
during the classroom demo.
"""

import logging
import sys
import os

# ANSI colour codes
COLOURS = {
    "aggregator": "\033[94m",   # blue
    "client_1":   "\033[92m",   # green
    "client_2":   "\033[93m",   # yellow
    "client_3":   "\033[95m",   # magenta
    "reset":      "\033[0m",
}


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create and return a logger that writes to both the console and a log file.

    Args:
        name    : e.g. 'aggregator', 'client_1'
        log_dir : folder where .log files are saved

    Returns:
        A configured logging.Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(name)-12s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler (coloured)
    colour = COLOURS.get(name, "")
    reset  = COLOURS["reset"]

    class ColourFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            return f"{colour}{msg}{reset}"

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColourFormatter(
        fmt="[%(asctime)s] [%(name)-12s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    ))

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
