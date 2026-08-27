"""Logging helpers for the XBID hybrid trader.

The purpose of this module is to centralise logging configuration.  By
encapsulating logging setup here we avoid accidentally configuring loggers
multiple times across different entry points.  See :func:`setup_logging` for
details.
"""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Initialise the root logger with a sensible default format.

    Parameters
    ----------
    level:
        Logging level for the root logger.  Typical values are ``logging.INFO``
        or ``logging.DEBUG``.

    This function should be called once at the beginning of a script or Jupyter
    session.  It configures the root logger; after calling it child loggers
    created via ``logging.getLogger(__name__)`` will inherit this configuration.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )