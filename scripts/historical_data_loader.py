"""Loader for historical XBID market data from the research Excel file.

Parses all three sheets and provides structured data per trading day:

Sheet 1 — Dataframe:
    Per-slot (15-min) consumption, load forecasts (ISP1/2/3),
    DAM prices, IDA1/2/3 prices, XBID VWAP/MIN/MAX.

Sheet 2 — XBID Data:
    Per-slot XBID trade statistics (VWAP, MIN, MAX, volume)
    separated by Buy/Sell side.

Sheet 3 — Past Data:
    MV/HV metered values from the previous week, used as
    forecast proxy.

Day convention (Greek local time):
    Each trading day runs from 01:00 on day D to 00:45 on day D+1.
    Slot timestamps with hour==0 (00:00-00:45) belong to the
    PREVIOUS calendar day.  Days at DST transitions have 100 slots
    (clocks go back) or 92 slots (clocks go forward).

Usage
-----
::

    loader = HistoricalDataLoader("scripts/Διπλωματική_-_XBID_Trading.xlsx")
    loader.load()

    # DAM prices for simulation reference prices
    dam = loader.get_dam_prices_by_day()       # {date: np.ndarray(n_slots,)}

    # XBID stats for calibration / validation
    xbid = loader.get_xbid_stats_by_day()      # {date: {vwap_buy, vwap_sell, ...}}

    # Load forecast errors for training signals
    errors = loader.get_forecast_errors_by_day()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpyxl

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Loads and structures historical market data from the Excel file.

    Parameters
    ----------
    xlsx_path:
        Path to the Excel file.
    """

    def __init__(self, xlsx_path: str) -> None:
        self.xlsx_path = Path(xlsx_path)
        self._loaded = False

        # Per-day data stores — each maps date → list of per-slot values
        self._dam_prices:     Dict[date, List[float]] = defaultdict(list)
        self._ida1_prices:    Dict[date, List[float]] = defaultdict(list)
        self._ida2_prices:    Dict[date, List[float]] = defaultdict(list)
        self._ida3_prices:    Dict[date, List[float]] = defaultdict(list)
        self._metered_total:  Dict[date, List[float]] = defaultdict(list)
        self._isp1_forecast:  Dict[date, List[float]] = defaultdict(list)
        self._isp2_forecast:  Dict[date, List[float]] = defaultdict(list)
        self._isp3_forecast:  Dict[date, List[float]] = defaultdict(list)
        self._xbid_vwap_buy:  Dict[date, Dict[int, float]] = defaultdict(dict)
        self._xbid_vwap_sell: Dict[date, Dict[int, float]] = defaultdict(dict)
        self._xbid_min:       Dict[date, Dict[int, float]] = defaultdict(dict)
        self._xbid_max:       Dict[date, Dict[int, float]] = defaultdict(dict)
        self._xbid_volume:    Dict[date, Dict[int, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse all sheets from the Excel file."""
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.xlsx_path}")

        logger.info("Loading historical data from %s ...", self.xlsx_path.name)
        wb = openpyxl.load_workbook(
            str(self.xlsx_path), read_only=True, data_only=True
        )

        self._parse_dataframe(wb["Dataframe"])
        self._parse_xbid_data(wb["XBID Data"])
        self._loaded = True

        days = sorted(self._dam_prices.keys())
        logger.info(
            "Loaded %d trading days (%s → %s)",
            len(days), days[0], days[-1],
        )

    def _delivery_date(self, ts: datetime) -> date:
        """Map a Greek-time timestamp to its trading day.

        Slots 01:00-23:45 on day D → belong to day D.
        Slots 00:00-00:45 on day D+1 → belong to day D.
        """
        if ts.hour == 0:
            return (ts - timedelta(days=1)).date()
        return ts.date()

    def _parse_dataframe(self, ws) -> None:
        """Parse Sheet 1 — Dataframe.

        Column layout (0-based):
        0:timestamp  1:LV_metered  2:MV_metered  3:HV_metered
        4:total_metered  5:ISP1  6:ISP2  7:ISP3
        8:MV_forecast  9:HV_forecast
        10:LV_ISP1  11:LV_ISP2  12:LV_ISP3
        13:DAM  14:IDA1  15:IDA2  16:IDA3
        17:XBID_VWAP  18:XBID_MIN  19:XBID_MAX
        """
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue  # skip header
            ts = row[0]
            if not isinstance(ts, datetime):
                continue

            d = self._delivery_date(ts)
            self._metered_total[d].append(self._safe_float(row[4]))
            self._isp1_forecast[d].append(self._safe_float(row[5]))
            self._isp2_forecast[d].append(self._safe_float(row[6]))
            self._isp3_forecast[d].append(self._safe_float(row[7]))
            self._dam_prices[d].append(self._safe_float(row[13]))
            self._ida1_prices[d].append(self._safe_float(row[14]))
            self._ida2_prices[d].append(self._safe_float(row[15]))
            self._ida3_prices[d].append(self._safe_float(row[16]))

    def _parse_xbid_data(self, ws) -> None:
        """Parse Sheet 2 — XBID Data.

        Column layout (0-based):
        0:TARGET  1:ZONE  2:SIDE  3:DDAY  4:ASSET  5:CLASS
        6:DELIVERY_DATETIME  7:CONTRACT_ID  8:DURATION
        9:VWAP  10:MIN  11:MAX  12:TOTAL_TRADES  13:VER
        """
        # Collect raw data
        raw: Dict[Tuple[date, datetime, str], dict] = {}
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue
            delivery_dt = row[6]
            side = row[2]
            if not isinstance(delivery_dt, datetime) or side not in ("Buy", "Sell"):
                continue
            d = delivery_dt.date()
            raw[(d, delivery_dt, side)] = {
                "vwap":   self._safe_float(row[9]),
                "min":    self._safe_float(row[10]),
                "max":    self._safe_float(row[11]),
                "volume": self._safe_float(row[12]),
            }

        # Build sorted slot index per day
        slots_by_day: Dict[date, set] = defaultdict(set)
        for (d, dt, _) in raw:
            slots_by_day[d].add(dt)
        sorted_slots = {d: sorted(dts) for d, dts in slots_by_day.items()}

        # Assign slot indices
        for (d, dt, side), vals in raw.items():
            slot_idx = sorted_slots[d].index(dt)
            if side == "Buy":
                self._xbid_vwap_buy[d][slot_idx]  = vals["vwap"]
            else:
                self._xbid_vwap_sell[d][slot_idx] = vals["vwap"]
                self._xbid_min[d][slot_idx]        = vals["min"]
                self._xbid_max[d][slot_idx]        = vals["max"]
                self._xbid_volume[d][slot_idx]     = vals["volume"]

    @staticmethod
    def _safe_float(val) -> float:
        """Convert cell value to float, NaN for missing/non-numeric."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_dam_prices_by_day(self) -> Dict[date, np.ndarray]:
        """DAM clearing prices per trading day.

        Returns
        -------
        Dict[date, np.ndarray]
            Shape ``(n_slots,)`` — 96 normally, 100 on DST-end day.
            Use as ``reference_prices`` for :class:`MarketEngine`.
        """
        self._require_loaded()
        return {d: np.array(v, dtype=float) for d, v in self._dam_prices.items()}

    def get_ida_prices_by_day(self) -> Dict[date, np.ndarray]:
        """IDA1/2/3 auction prices stacked as ``(n_slots, 3)`` array.

        Column 0 = IDA1, 1 = IDA2, 2 = IDA3.
        Use for benchmarking agent performance vs auctions.
        """
        self._require_loaded()
        result = {}
        for d in self._dam_prices:
            result[d] = np.column_stack([
                np.array(self._ida1_prices[d], dtype=float),
                np.array(self._ida2_prices[d], dtype=float),
                np.array(self._ida3_prices[d], dtype=float),
            ])
        return result

    def get_xbid_stats_by_day(self) -> Dict[date, Dict[str, np.ndarray]]:
        """XBID market statistics per trading day.

        Keys per day:
        - ``vwap_buy``  : buy-side VWAP per slot (€/MWh)
        - ``vwap_sell`` : sell-side VWAP per slot (€/MWh)
        - ``min_price`` : minimum trade price per slot (€/MWh)
        - ``max_price`` : maximum trade price per slot (€/MWh)
        - ``volume``    : total traded volume per slot (MW)

        NaN for slots with no XBID activity.

        Use for:
        - Background trader calibration (price_sigma, spread)
        - Validation: compare agent execution vs VWAP
        """
        self._require_loaded()
        result = {}
        for d, dam in self._dam_prices.items():
            n = len(dam)
            vwap_buy  = np.full(n, np.nan)
            vwap_sell = np.full(n, np.nan)
            mn = np.full(n, np.nan)
            mx = np.full(n, np.nan)
            vol = np.full(n, np.nan)

            for slot, val in self._xbid_vwap_buy.get(d, {}).items():
                if slot < n:
                    vwap_buy[slot] = val
            for slot, val in self._xbid_vwap_sell.get(d, {}).items():
                if slot < n:
                    vwap_sell[slot] = val
            for slot, val in self._xbid_min.get(d, {}).items():
                if slot < n:
                    mn[slot] = val
            for slot, val in self._xbid_max.get(d, {}).items():
                if slot < n:
                    mx[slot] = val
            for slot, val in self._xbid_volume.get(d, {}).items():
                if slot < n:
                    vol[slot] = val

            result[d] = {
                "vwap_buy":  vwap_buy,
                "vwap_sell": vwap_sell,
                "min_price": mn,
                "max_price": mx,
                "volume":    vol,
            }
        return result

    def get_forecast_errors_by_day(self) -> Dict[date, Dict[str, np.ndarray]]:
        """System-level load forecast errors per trading day.

        forecast_error = ISP_forecast - Metered_consumption

        Positive = ISP overestimated load (system has surplus tendency).
        Negative = ISP underestimated load (system has deficit tendency).

        Keys per day:
        - ``isp1_error``, ``isp2_error``, ``isp3_error`` : errors per slot
        - ``metered``  : actual metered consumption per slot (MW)
        - ``isp1``, ``isp2``, ``isp3`` : raw ISP forecasts (MW)

        NOTE: system-level, not supplier-level.  Use as training proxy
        until supplier-level forecasts are available.
        """
        self._require_loaded()
        result = {}
        for d in self._dam_prices:
            metered = np.array(self._metered_total[d], dtype=float)
            isp1    = np.array(self._isp1_forecast[d], dtype=float)
            isp2    = np.array(self._isp2_forecast[d], dtype=float)
            isp3    = np.array(self._isp3_forecast[d], dtype=float)
            result[d] = {
                "isp1_error": isp1 - metered,
                "isp2_error": isp2 - metered,
                "isp3_error": isp3 - metered,
                "metered":    metered,
                "isp1": isp1,
                "isp2": isp2,
                "isp3": isp3,
            }
        return result

    def available_dates(self) -> List[date]:
        """Sorted list of all available trading days."""
        self._require_loaded()
        return sorted(self._dam_prices.keys())

    def calibrate_engine_config(self) -> dict:
        """Compute calibrated background agent parameters from real XBID data.

        Uses XBID MIN/MAX/VOLUME statistics to set realistic price volatility
        and volume parameters for the background trading agents.

        Returns
        -------
        dict
            ``background_agent_configs`` dict ready to pass to
            :class:`~xbid_trader.market.market_engine.MarketEngine`.

        Calibration methodology
        -----------------------
        price_sigma (NoiseTrader):
            Half the median intraday spread (MAX - MIN) / 2.
            Represents typical price noise around mid.

        price_offset (LiquidityProvider):
            Median spread / 4 — tighter than noise trader.

        volume_mean (all agents):
            Median XBID traded volume per slot, scaled down per-agent
            since multiple agents share the total volume.
        """
        self._require_loaded()

        xbid_stats = self.get_xbid_stats_by_day()

        all_spreads = []
        all_volumes = []

        for d, stats in xbid_stats.items():
            mn  = stats["min_price"]
            mx  = stats["max_price"]
            vol = stats["volume"]

            spread = mx - mn
            valid_spread = spread[~np.isnan(spread) & (spread >= 0)]
            valid_vol    = vol[~np.isnan(vol) & (vol > 0)]

            all_spreads.extend(valid_spread.tolist())
            all_volumes.extend(valid_vol.tolist())

        if not all_spreads:
            logger.warning("No XBID spread data — returning default config")
            return {}

        spreads = np.array(all_spreads)
        volumes = np.array(all_volumes)

        # Use median to avoid influence of extreme outliers
        median_spread = float(np.median(spreads))
        median_volume = float(np.median(volumes))
        std_spread    = float(np.std(spreads))

        # Scale volume per agent — 4 background agents share total volume
        vol_per_agent = max(median_volume / 4.0, 0.5)

        logger.info(
            "XBID calibration: median_spread=%.2f€  std_spread=%.2f€  "
            "median_vol=%.2f MW  vol_per_agent=%.2f MW",
            median_spread, std_spread, median_volume, vol_per_agent,
        )

        return {
            "noise_trader": {
                "order_rate_per_minute": 20.0,
                "price_sigma":  round(max(median_spread / 2.0, 0.5), 2),
                "volume_mean":  round(vol_per_agent, 2),
                "volume_std":   round(vol_per_agent * 0.4, 2),
            },
            "liquidity_provider": {
                "order_rate_per_minute": 8.0,
                "price_offset": round(max(median_spread / 4.0, 0.25), 2),
                "volume_mean":  round(vol_per_agent * 1.5, 2),
                "volume_std":   round(vol_per_agent * 0.3, 2),
            },
            "urgency_trader": {
                "order_rate_per_minute": 6.0,
                "aggression_factor": 8.0,
                "volume_mean":  round(vol_per_agent, 2),
                "volume_std":   round(vol_per_agent * 0.4, 2),
            },
            "forecast_informed_trader": {
                "order_rate_per_minute": 6.0,
                "signal_sensitivity":   round(max(median_spread / 4.0, 0.1), 2),
                "volume_mean":  round(vol_per_agent, 2),
                "volume_std":   round(vol_per_agent * 0.2, 2),
            },
        }

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() before accessing data.")
