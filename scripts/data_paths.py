"""Robust data-file resolution for the XBID pipeline.

Why this exists
---------------
The three Excel inputs have, in practice, shown up under inconsistent names:

  * the balancing-market file was shipped as ``Balanacin_Market_Data.xlsx``
    (misspelled) while the code expected ``Balancing_Market_Data.xlsx``;
  * the main thesis workbook has a Greek filename (``Διπλωματική_…``) that some
    archive tools transliterate/escape on extraction;
  * the imbalance-price file is sometimes singular/plural.

Hard-coded names meant that a missing/renamed file was *silently* swallowed by
a ``try/except`` and the pipeline degraded to its legacy fallback reward
without telling anyone.  This module fixes both problems:

  1. Files are located by **glob pattern** inside a data directory, so small
     spelling/locale differences no longer matter (no renaming required).
  2. A loud banner is printed whenever an expected input cannot be found, and
     an optional STRICT mode (env ``XBID_STRICT=1``) turns that into a hard
     error so research runs fail fast instead of reporting misleading numbers.

Environment overrides (absolute or relative paths) still win:
    XBID_EXCEL_PATH, XBID_BM_PATH, XBID_IMBALANCE_PATH, XBID_DATA_DIR
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("data_paths")

# Default data directory = the folder that holds this file (scripts/), so
# resolution is independent of the current working directory.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent

STRICT = os.environ.get("XBID_STRICT", "0") not in ("0", "", "false", "False")

# Glob patterns (first match wins). Patterns are intentionally permissive so
# that minor spelling/locale variants resolve without manual renaming.
EXCEL_PATTERNS     = ["*XBID_Trading.xlsx", "*XBID*Trading*.xlsx"]
BM_PATTERNS        = ["*Balanc*Market*Data*.xlsx", "*Balanac*Market*Data*.xlsx",
                      "*Market_Data.xlsx"]
IMBALANCE_PATTERNS = ["*Imbalance*Price*.xlsx", "Imbalance_Price.xlsx"]


def _data_dir() -> Path:
    return Path(os.environ.get("XBID_DATA_DIR", str(DEFAULT_DATA_DIR)))


def _glob_first(patterns: List[str], data_dir: Optional[Path] = None) -> Optional[Path]:
    base = data_dir or _data_dir()
    for pat in patterns:
        matches = sorted(base.glob(pat))
        if matches:
            return matches[0]
    return None


def _banner(lines: List[str]) -> None:
    width = max(len(s) for s in lines) + 4
    bar = "!" * width
    logger.warning("\n%s", bar)
    for s in lines:
        logger.warning("! %s", s.ljust(width - 4))
    logger.warning("%s", bar)


def resolve_path(
    env_var: str,
    patterns: List[str],
    label: str,
    required: bool = False,
) -> Optional[Path]:
    """Resolve a data file.

    Order: explicit env override → glob match in the data dir → None.
    A loud banner (and, under STRICT, a RuntimeError) is emitted when a
    ``required`` file cannot be found.
    """
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if p.exists():
            return p
        _banner([f"{label}: env {env_var}={override} does not exist."])
        if STRICT and required:
            raise RuntimeError(f"{label} not found (env override missing).")
        return None

    p = _glob_first(patterns)
    if p is not None:
        return p

    if required or True:  # always warn loudly on a miss
        msg = [
            f"{label}: NOT FOUND in {_data_dir()}",
            f"  searched patterns: {patterns}",
            "  → the pipeline will fall back to a DEGRADED path",
            "    (legacy reward / proxy prices). Results will NOT match the",
            f"    report. Set {env_var} or place the file in XBID_DATA_DIR.",
        ]
        _banner(msg)
        if STRICT and required:
            raise RuntimeError(f"{label} not found and XBID_STRICT=1.")
    return None


def excel_path(required: bool = True) -> Optional[Path]:
    return resolve_path("XBID_EXCEL_PATH", EXCEL_PATTERNS,
                        "Main workbook (ISP/DAM/XBID/IDA)", required=required)


def bm_path(required: bool = True) -> Optional[Path]:
    return resolve_path("XBID_BM_PATH", BM_PATTERNS,
                        "Balancing-market dual prices", required=required)


def imbalance_path(required: bool = True) -> Optional[Path]:
    return resolve_path("XBID_IMBALANCE_PATH", IMBALANCE_PATTERNS,
                        "Imbalance settlement prices", required=required)
