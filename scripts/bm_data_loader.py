"""Loader for the IPTO Balancing Market data export.

Parses ``Balancing_Market_Data.xlsx`` and exposes per-day arrays of:
    - ``bm_up_price``     (€/MWh) — pay this if your portfolio is short
    - ``bm_down_price``   (€/MWh) — receive this if your portfolio is long
    - ``bm_up_energy``    (MW)    — system-wide upward activation
    - ``bm_down_energy``  (MW)    — system-wide downward activation
    - ``imb_price_single`` (€/MWh) — the BM price applicable to the supplier's
      residual under single-pricing (BM Up if system short, BM Down if long)

These prices govern the *real* economic cost of leaving residual imbalance
to settlement, and replace the XBID VWAP proxy that was previously used in
HistoricalScenarioProvider.imbalance_prices.

DST handling
------------
The Excel file appends " (2)" to duplicate timestamps during DST fall-back.
This loader strips that suffix transparently.

Trading day convention
----------------------
For consistency with the rest of the project (see HistoricalDataLoader),
slots with hour == 0 belong to the previous calendar day:
    01:00–23:45 of day D  → trading day D
    00:00–00:45 of day D+1 → trading day D
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Slots/day under normal Greek time
SLOTS_PER_DAY = 96


class BMDataLoader:
    """Loads and exposes per-day Balancing Market arrays.

    Parameters
    ----------
    xlsx_path:
        Path to ``Balancing_Market_Data.xlsx``.
    """

    def __init__(self, xlsx_path: str | Path) -> None:
        self.xlsx_path = Path(xlsx_path)
        self._loaded = False

        self._bm_up_price:    Dict[date, List[float]] = defaultdict(list)
        self._bm_down_price:  Dict[date, List[float]] = defaultdict(list)
        self._bm_up_energy:   Dict[date, List[float]] = defaultdict(list)
        self._bm_down_energy: Dict[date, List[float]] = defaultdict(list)
        self._slot_index:     Dict[date, List[int]]   = defaultdict(list)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read the Excel file and populate per-day buffers."""
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"BM data file not found: {self.xlsx_path}")

        df = pd.read_excel(self.xlsx_path)
        # Strip DST " (2)" suffix that appears on duplicate hours
        df["ts_clean"] = (
            df["Trading Period"].astype(str).str.replace(r" \(\d+\)$", "", regex=True)
        )
        df["ts"] = pd.to_datetime(df["ts_clean"])
        df = df.sort_values("ts").reset_index(drop=True)

        # Group by trading day with the same convention as HistoricalDataLoader
        for _, row in df.iterrows():
            ts: datetime = row["ts"].to_pydatetime()
            d = self._delivery_date(ts)
            self._bm_up_price[d].append(self._safe_float(row["BM Up Price"]))
            self._bm_down_price[d].append(self._safe_float(row["BM Down Price"]))
            self._bm_up_energy[d].append(self._safe_float(row["BM Up Energy"]))
            self._bm_down_energy[d].append(self._safe_float(row["BM Down Energy"]))

        self._loaded = True
        days = sorted(self._bm_up_price.keys())
        logger.info(
            "Loaded BM data for %d trading days (%s → %s)",
            len(days), days[0], days[-1],
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def available_dates(self) -> List[date]:
        self._require_loaded()
        return sorted(self._bm_up_price.keys())

    def get_bm_prices_by_day(self) -> Dict[date, Dict[str, np.ndarray]]:
        """Return per-day dict with keys:

            bm_up_price, bm_down_price, bm_up_energy, bm_down_energy,
            imb_price_single

        ``imb_price_single`` is the supplier-applicable settlement price
        under single pricing: BM Up when the system is short
        (Up energy > Down energy), BM Down otherwise.
        """
        self._require_loaded()
        out: Dict[date, Dict[str, np.ndarray]] = {}
        for d in self.available_dates():
            up_p = np.array(self._bm_up_price[d],   dtype=float)
            dn_p = np.array(self._bm_down_price[d], dtype=float)
            up_e = np.array(self._bm_up_energy[d],  dtype=float)
            dn_e = np.array(self._bm_down_energy[d], dtype=float)

            # Predominant direction: system is "short" when more upward
            # energy was activated than downward.
            short = up_e >= dn_e
            imb_single = np.where(short, up_p, dn_p)

            out[d] = {
                "bm_up_price":     up_p,
                "bm_down_price":   dn_p,
                "bm_up_energy":    up_e,
                "bm_down_energy":  dn_e,
                "imb_price_single": imb_single,
            }
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _delivery_date(ts: datetime) -> date:
        """Trading-day convention: hour==0 belongs to the previous day."""
        if ts.hour == 0:
            return (ts - timedelta(days=1)).date()
        return ts.date()

    @staticmethod
    def _safe_float(value) -> float:
        if value is None:
            return float("nan")
        try:
            v = float(value)
            if np.isnan(v):
                return float("nan")
            return v
        except (TypeError, ValueError):
            return float("nan")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() before accessing BM data")
