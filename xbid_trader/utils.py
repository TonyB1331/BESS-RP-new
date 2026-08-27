"""Utility functions for the XBID hybrid trader.

This module provides helper routines for reproducibility and configuration
loading.  Keeping these simple functions centralised avoids code duplication
throughout the project.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random number generators.

    Parameters
    ----------
    seed:
        An integer seed.  Using a fixed seed ensures reproducible simulations
        across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested dictionary.

    Parameters
    ----------
    path:
        Path to the YAML file on disk.

    Returns
    -------
    dict
        A nested dictionary representing the parsed YAML configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_dicts(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Values in ``updates`` override those in ``base``.  If a value is itself a
    dictionary in both base and updates the merge proceeds recursively.  The
    original dictionaries are not modified.
    """
    if updates is None:
        return base.copy()
    result = base.copy()
    for key, value in updates.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a basic logging format for the library.

    This helper should be called from scripts before using the library.  It
    installs a root logger with a simple message format.  Individual modules
    create child loggers through ``logging.getLogger(__name__)``.

    Parameters
    ----------
    level:
        Numeric logging level.  Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )