"""Historical scenario provider for RL training.

Rt specification
----------------
    DAM_position,i  = ISP1_D-1,i × market_share × slot_duration_h   [MWh]
    Rt,i            = (ISP_current,i − ISP1_D-1,i) × market_share × slot_duration_h

    Slots  0–63  (00:00–16:00) → ISP_current = ISP2_D
    Slots 64–95  (16:00–24:00) → ISP_current = ISP3_D

ISP forecasts are MW (instantaneous power); MWh per 15-min slot = MW × 0.25h.

Settlement prices
-----------------
The supplier's residual imbalance is settled at IPTO Balancing Market
prices, NOT at XBID VWAP.  When a ``bm_loader`` is supplied, the
scenario carries:

    bm_up_price[i]   — €/MWh, paid by suppliers who are short at slot i
    bm_down_price[i] — €/MWh, received by suppliers who are long at slot i
    imbalance_prices[i] — single-direction settlement price (BM Up if
                          system short, BM Down if long), used as the
                          mark-to-imbalance reference for hedging PnL
                          and shaping signals.

Without a ``bm_loader``, the legacy XBID-VWAP fallback is used so that
all existing call sites keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np


MARKET_SHARE:        float = 0.06    # supplier market share
SLOT_DURATION_HOURS: float = 0.25    # 15-minute delivery products


@dataclass
class HistoricalScenario:
    prices:            np.ndarray
    res_errors:        np.ndarray
    load_errors:       np.ndarray
    imbalance_prices:  np.ndarray
    xbid_volume:       np.ndarray
    dam_position:      np.ndarray
    rt:                np.ndarray
    # Optional dual-pricing fields — populated only when a BM loader is
    # supplied to the provider.  Default to None for back-compat.
    bm_up_price:       Optional[np.ndarray] = None
    bm_down_price:     Optional[np.ndarray] = None

    @property
    def n_products(self) -> int:
        return len(self.prices)


class HistoricalScenarioProvider:
    """Provides historical scenarios for RL training.

    Parameters
    ----------
    loader:
        A loaded HistoricalDataLoader instance (DAM, ISP, XBID stats).
    market_share:
        Supplier market share fraction.  Default 0.06 (6%).
    fallback_imbalance_spread:
        Fallback spread over DAM when BM and XBID prices are NaN.
    slot_duration_hours:
        Slot duration for MW → MWh conversion.  Default 0.25.
    bm_loader:
        Optional ``BMDataLoader`` instance.  When supplied, real IPTO
        Balancing Market prices replace the XBID VWAP proxy in
        ``imbalance_prices`` and the dual-pricing arrays
        ``bm_up_price`` / ``bm_down_price`` are populated.
    imbalance_loader:
        Optional ``ImbalancePriceLoader`` instance.  When supplied,
        the single imbalance settlement price from this file overrides
        ``imbalance_prices`` (taking priority over both BM-derived
        and XBID VWAP proxies).
    """

    def __init__(
        self,
        loader,
        market_share: float = MARKET_SHARE,
        fallback_imbalance_spread: float = 5.0,
        slot_duration_hours: float = SLOT_DURATION_HOURS,
        bm_loader=None,
        imbalance_loader=None,
        rt_morning_source: Optional[str] = None,
    ) -> None:
        self._loader              = loader
        self._market_share        = market_share
        self._fallback_spread     = fallback_imbalance_spread
        self._slot_duration_hours = slot_duration_hours
        self._unit_factor         = market_share * slot_duration_hours
        self._rt_morning_source   = (
            rt_morning_source if rt_morning_source is not None
            else os.environ.get("XBID_RT_MORNING", "today")
        )

        self._dam    = loader.get_dam_prices_by_day()
        self._errors = loader.get_forecast_errors_by_day()
        self._xbid   = loader.get_xbid_stats_by_day()

        # Optional BM data
        self._bm: Dict[date, Dict[str, np.ndarray]] = {}
        if bm_loader is not None:
            self._bm = bm_loader.get_bm_prices_by_day()

        # Optional imbalance settlement price
        self._imbalance_loader = imbalance_loader

        # Dataset is the intersection of DAM-priced days and ISP-error days.
        # When a BM loader is supplied, also restrict to days with BM data
        # so callers can rely on every returned scenario having the BM fields
        # populated.
        candidate_days = set(loader.available_dates())
        if self._bm:
            candidate_days &= set(self._bm.keys())
        self._days = sorted(candidate_days)

        self._day_idx: Dict[date, int] = {d: i for i, d in enumerate(self._days)}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_scenario(self, day: date) -> HistoricalScenario:
        if day not in self._dam:
            raise KeyError(
                f"Day {day} not found. "
                f"Available: {self._days[0]} → {self._days[-1]}"
            )

        dam    = self._dam[day]
        n      = len(dam)
        errors = self._errors[day]
        xbid   = self._xbid.get(day, {})
        bm     = self._bm.get(day, {})

        isp2_today = errors["isp2"]
        isp3_today = errors["isp3"]

        # ── Previous-day ISP1, ISP2 ──────────────────────────────────
        prev_isp1 = self._get_prev_isp(day, n, "isp1")
        # Morning intraday revision source for slots 0-63:
        #   "today"    -> today's ISP2 (realistic same-day morning revision)
        #   "prev_day" -> yesterday's ISP2 (legacy; caused a ~3x artificial
        #                 |Rt| jump at slot 64)
        if self._rt_morning_source == "prev_day":
            morning_ref = self._get_prev_isp(day, n, "isp2")
        else:
            morning_ref = isp2_today

        # ── DAM position = ISP1_D-1 × market_share × slot_h ──────────
        dam_position = prev_isp1 * self._unit_factor

        # ── Rt: slots 0–63 use ISP2_D, slots 64+ use ISP3_D ────────
        rt = np.empty(n, dtype=float)
        end_mid = min(64, n)
        rt[:end_mid] = (morning_ref[:end_mid] - prev_isp1[:end_mid]) * self._unit_factor
        if n > 64:
            rt[64:] = (isp3_today[64:n] - prev_isp1[64:n]) * self._unit_factor

        # ── Info-only state features (unscaled ISP − Metered) ───────
        isp1_err = errors["isp1_error"]
        isp2_err = errors["isp2_error"]
        isp3_err = errors["isp3_error"]

        load_errors = np.empty(n, dtype=float)
        load_errors[:min(32, n)]   = isp1_err[:min(32, n)]
        load_errors[32:min(64, n)] = isp2_err[32:min(64, n)]
        if n > 64:
            load_errors[64:] = isp3_err[64:n]
        res_errors = np.zeros(n, dtype=float)

        # ── Settlement prices ────────────────────────────────────────
        # Preference order:
        #   1. real BM Up/Down/single from IPTO data (when bm_loader supplied)
        #   2. XBID VWAP_sell as a (legacy) proxy
        #   3. DAM + fixed spread fallback
        bm_up_price:   Optional[np.ndarray] = None
        bm_down_price: Optional[np.ndarray] = None

        if bm:
            bm_up_price   = self._fit_length(bm["bm_up_price"],   n)
            bm_down_price = self._fit_length(bm["bm_down_price"], n)
            imb_single    = self._fit_length(bm["imb_price_single"], n)
            bad = ~np.isfinite(imb_single)
            if bad.any():
                imb_single = imb_single.copy()
                imb_single[bad] = dam[bad] + self._fallback_spread
            imbalance_prices = imb_single
        else:
            vwap_sell = xbid.get("vwap_sell", np.full(n, np.nan))
            vwap_sell = self._fit_length(vwap_sell, n)
            imbalance_prices = np.where(
                np.isnan(vwap_sell),
                dam + self._fallback_spread,
                vwap_sell,
            )

        # Override with dedicated imbalance price file if available
        if self._imbalance_loader is not None:
            imb_from_file = self._imbalance_loader.get_day(day)
            if imb_from_file is not None:
                imbalance_prices = self._fit_length(imb_from_file, n)

        # ── XBID volume ──────────────────────────────────────────────
        raw_vol = xbid.get("volume", np.full(n, np.nan))
        raw_vol = self._fit_length(raw_vol, n)
        xbid_volume = np.nan_to_num(raw_vol, nan=0.0)

        return HistoricalScenario(
            prices           = dam.copy(),
            res_errors       = res_errors,
            load_errors      = load_errors,
            imbalance_prices = imbalance_prices,
            xbid_volume      = xbid_volume,
            dam_position     = dam_position,
            rt               = rt,
            bm_up_price      = bm_up_price,
            bm_down_price    = bm_down_price,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
        """Pad / truncate ``arr`` to length ``n``.  Padding uses NaN."""
        a = np.asarray(arr, dtype=float)
        if len(a) >= n:
            return a[:n].copy()
        out = np.full(n, np.nan)
        out[:len(a)] = a
        return out

    def _get_prev_isp(self, day: date, n: int, which: str) -> np.ndarray:
        """ISP1/2/3 from the previous available day, padded to length n.
        Falls back to today's ISP1 if no previous day exists."""
        idx = self._day_idx.get(day, 0)
        if idx == 0:
            return self._errors[day]["isp1"].copy()[:n]

        prev_day    = self._days[idx - 1]
        prev_errors = self._errors.get(prev_day)
        if prev_errors is None:
            return self._errors[day]["isp1"].copy()[:n]

        series = prev_errors[which]
        if len(series) >= n:
            return series[:n].copy()
        result = np.empty(n, dtype=float)
        result[:len(series)] = series
        result[len(series):] = series[-1]
        return result

    def available_dates(self) -> List[date]:
        return self._days

    def __len__(self) -> int:
        return len(self._days)
