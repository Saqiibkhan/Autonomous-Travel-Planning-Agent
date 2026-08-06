"""
Structured logging setup.

We want every log line to tell us: timestamp, logger name (so we know which
module -- e.g. weather_tool, reflection -- emitted it), level, and message.
This matters a lot for this project specifically, because later on the
reflection node's whole job is to reason about *why* something failed, and
readable logs are how you (and we, debugging together) will actually see
that reasoning happen instead of it being a black box.
"""

import logging
import sys

from src.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]  # avoid duplicate handlers on reload

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-scoped logger.

    Usage in any module:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("places_tool: found 12 attractions for Hunza")
    """
    _configure_root_logger()
    return logging.getLogger(name)
