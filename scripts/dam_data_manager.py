"""HEnEx DAM price data manager for the XBID simulation.

Downloads, caches and serves the 96-slot Greece Mainland 15-min MCP prices
from the HEnEx public results portal for every trading day from a configurable
start date up to today (or any end date you specify).

URL pattern (from HEnEx documentation)::

    https://www.enexgroup.gr/documents/20126/200106/YYYYMMDD_EL-DAM_Results_EN_v01.xlsx

The manager tries v01 first, then v02, up to a configurable maximum version,
stopping at the first successful download.  Downloaded files are stored in a
local cache directory and never re-downloaded.

Typical usage::

    from scripts.dam_data_manager import DAMDataManager

    manager = DAMDataManager(cache_dir="scripts/dam_cache")
    manager.fetch_all()           # download everything not yet cached
    prices = manager.get_prices() # Dict[date, np.ndarray]  shape (96,) each

    # Iterate in chronological order for simulation
    for day, ref_prices in manager.iter_days():
        session.reset(reference_prices=ref_prices)
        ...
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import requests

# Re-use the existing loader from the same scripts directory
from dam_price_loader import load_greece_mainland_15min_mcp


logger = logging.getLogger(__name__)

# First date with 15-min MCP data on HEnEx
_DEFAULT_START = date(2025, 10, 1)

# HEnEx URL template — {date} is YYYYMMDD, {ver} is zero-padded version e.g. 01
_URL_TEMPLATE = (
    "https://www.enexgroup.gr/documents/20126/200106/"
    "{date}_EL-DAM_Results_EN_v{ver:02d}.xlsx"
)

# Maximum version number to try before giving up on a date
_MAX_VERSION = 5

# Seconds to wait between HTTP requests (be polite to the server)
_REQUEST_DELAY = 1.0

# HTTP request timeout in seconds
_TIMEOUT = 30


class DAMDataManager:
    """Downloads and caches HEnEx DAM 15-min MCP prices for Greece Mainland.

    Parameters
    ----------
    cache_dir:
        Directory where downloaded xlsx files are stored.  Created
        automatically if it does not exist.  Defaults to
        ``scripts/dam_cache`` relative to the current working directory.
    start_date:
        First date to fetch.  Defaults to 2025-10-01, when 15-min
        MCP data became available on HEnEx.
    end_date:
        Last date to fetch (inclusive).  Defaults to today at the time
        :meth:`fetch_all` is called.
    sheet_name:
        Sheet name inside the xlsx file containing the MCP data.
    """

    def __init__(
        self,
        cache_dir: str = "scripts/dam_cache",
        start_date: date = _DEFAULT_START,
        end_date: Optional[date] = None,
        sheet_name: str = "SPOT_Summary (SELL)",
    ) -> None:
        self.cache_dir   = Path(cache_dir)
        self.start_date  = start_date
        self.end_date    = end_date  # resolved to today() at fetch time
        self.sheet_name  = sheet_name

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory store: date → (96,) array of MCP prices
        self._prices: Dict[date, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Download interface
    # ------------------------------------------------------------------

    def load_from_excel(
        self,
        loader,
        overwrite: bool = False,
    ) -> None:
        """Load DAM prices from a :class:`HistoricalDataLoader` instance.

        Use this to fill dates not available from HEnEx downloads
        (e.g. 2025-10-01 → 2025-12-31 which may be missing from cache).

        Parameters
        ----------
        loader:
            A loaded :class:`HistoricalDataLoader` instance.
        overwrite:
            If ``True``, overwrite dates already in ``_prices``.
            Default ``False`` — only fills missing dates.
        """
        excel_prices = loader.get_dam_prices_by_day()
        added = 0
        for d, prices in excel_prices.items():
            if d not in self._prices or overwrite:
                self._prices[d] = prices
                added += 1
        logger.info(
            "Loaded %d days from Excel (%d total in manager)",
            added, len(self._prices),
        )

    def fetch_all(self, force_reload: bool = False) -> None:
        """Download and parse all missing dates from start_date to today.

        Already-cached files are skipped unless ``force_reload=True``.
        After completion, :meth:`get_prices` and :meth:`iter_days` are
        ready to use.

        Parameters
        ----------
        force_reload:
            If ``True``, re-download even dates that are already cached.
        """
        end = self.end_date or (date.today() + timedelta(days=1))
        current = self.start_date
        total = (end - current).days + 1
        downloaded = skipped = failed = 0

        logger.info(
            "Fetching DAM prices from %s to %s (%d days)",
            self.start_date, end, total,
        )

        while current <= end:
            # Skip dates already loaded (e.g. from Excel)
            if current in self._prices and not force_reload:
                current += timedelta(days=1)
                downloaded += 1
                continue

            prices = self._load_date(current, force_reload=force_reload)

            if prices is not None:
                self._prices[current] = prices
                downloaded += 1
            else:
                # Skip this day entirely — no forward fill
                logger.warning("%s: no data available — day skipped", current)
                failed += 1

            current += timedelta(days=1)

        logger.info(
            "Done. %d days loaded, %d missing/filled.", downloaded, failed
        )

    def _load_date(
        self, day: date, force_reload: bool = False
    ) -> Optional[np.ndarray]:
        """Return prices for ``day``, downloading if necessary.

        Returns ``None`` if the file cannot be obtained.
        """
        # Check cache first
        cached_path = self._cache_path(day)
        if cached_path.exists() and not force_reload:
            try:
                return load_greece_mainland_15min_mcp(
                    str(cached_path), sheet_name=self.sheet_name
                )
            except KeyError as exc:
                logger.warning(
                    "%s: sheet not found in cached file (%s) — day skipped", day, exc
                )
                return None
            except Exception as exc:
                logger.warning(
                    "%s: parse failed on cached file (%s) — day skipped", day, exc
                )
                return None

        # Try downloading, attempting versions v01 → v{MAX_VERSION}
        date_str = day.strftime("%Y%m%d")
        for ver in range(1, _MAX_VERSION + 1):
            url = _URL_TEMPLATE.format(date=date_str, ver=ver)
            try:
                logger.debug("GET %s", url)
                response = requests.get(url, timeout=_TIMEOUT)
                if response.status_code == 200:
                    cached_path.write_bytes(response.content)
                    time.sleep(_REQUEST_DELAY)
                    try:
                        prices = load_greece_mainland_15min_mcp(
                            str(cached_path), sheet_name=self.sheet_name
                        )
                        logger.info("Downloaded %s (v%02d)", day, ver)
                        return prices
                    except KeyError as exc:
                        # Downloaded OK but sheet missing — public holiday etc.
                        logger.warning(
                            "%s: downloaded but sheet not found (%s) — day skipped",
                            day, exc,
                        )
                        return None
                    except Exception as exc:
                        # Parse failed — keep the file for manual inspection
                        logger.warning("Parse error for %s v%02d: %s", day, ver, exc)
                        return None
                elif response.status_code == 404:
                    logger.debug("%s v%02d → 404, trying next version", day, ver)
                    time.sleep(_REQUEST_DELAY)
                else:
                    logger.warning(
                        "%s v%02d → HTTP %d", day, ver, response.status_code
                    )
                    time.sleep(_REQUEST_DELAY)
                    break  # unexpected status — don't keep trying versions

            except requests.RequestException as exc:
                logger.warning("Request failed for %s v%02d: %s", day, ver, exc)
                break

        return None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_prices(self) -> Dict[date, np.ndarray]:
        """Return all loaded prices as a ``{date: np.ndarray}`` dict.

        Call :meth:`fetch_all` first.  Each array has shape ``(96,)``
        and contains MCP values in €/MWh for the 96 quarter-hour slots.
        """
        return dict(self._prices)

    def iter_days(self) -> Iterator[Tuple[date, np.ndarray]]:
        """Yield ``(date, prices)`` pairs in chronological order.

        Only dates for which prices are available are yielded.
        """
        for day in sorted(self._prices):
            yield day, self._prices[day]

    def available_dates(self) -> list[date]:
        """Return a sorted list of dates for which prices are loaded."""
        return sorted(self._prices)

    def load_from_historical(
        self,
        loader,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> None:
        """Load DAM prices from a HistoricalDataLoader (Excel file).

        Use this to supplement the cache with Oct-Dec 2025 prices that
        are not available via HEnEx automated download.

        Parameters
        ----------
        loader:
            A loaded :class:`HistoricalDataLoader` instance.
        start:
            First date to import.  Default: first available in loader.
        end:
            Last date to import (inclusive).  Default: last available.

        Example
        -------
        ::

            from scripts.historical_data_loader import HistoricalDataLoader
            from scripts.dam_data_manager import DAMDataManager

            hist = HistoricalDataLoader("path/to/excel.xlsx")
            hist.load()

            manager = DAMDataManager(cache_dir="scripts/dam_cache")
            manager.load_from_historical(hist,
                                         start=date(2025, 10, 1),
                                         end=date(2025, 12, 31))
            manager.fetch_all()   # downloads Jan 2026 onwards from HEnEx
        """
        prices = loader.dam_prices_for_dam_manager(start=start, end=end)
        loaded = 0
        for d, arr in prices.items():
            if d not in self._prices:   # don't overwrite existing data
                self._prices[d] = arr
                loaded += 1
        logger.info(
            "Loaded %d days from HistoricalDataLoader (%s → %s)",
            loaded,
            min(prices) if prices else "-",
            max(prices) if prices else "-",
        )
        """Load all already-cached xlsx files without hitting the network.

        Useful if you want to work offline after a previous :meth:`fetch_all`.
        """
        loaded = 0
        for xlsx_file in sorted(self.cache_dir.glob("*_EL-DAM_Results_EN_v*.xlsx")):
            date_str = xlsx_file.name[:8]
            try:
                day = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                if day in self._prices:
                    continue  # already loaded
                prices = load_greece_mainland_15min_mcp(
                    str(xlsx_file), sheet_name=self.sheet_name
                )
                self._prices[day] = prices
                loaded += 1
            except Exception as exc:
                logger.warning("Could not load %s: %s", xlsx_file.name, exc)

        logger.info("Loaded %d days from cache.", loaded)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_path(self, day: date) -> Path:
        """Return the expected cache file path for a given date (v01)."""
        return self.cache_dir / f"{day.strftime('%Y%m%d')}_EL-DAM_Results_EN_v01.xlsx"

    def __len__(self) -> int:
        return len(self._prices)

    def __repr__(self) -> str:
        return (
            f"DAMDataManager(cache_dir='{self.cache_dir}', "
            f"dates={self.start_date}→{self.end_date or 'today'}, "
            f"loaded={len(self._prices)} days)"
        )
