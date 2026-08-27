"""Loader for the *given* day-ahead (DAM) battery schedule.

Reads the results workbook produced by the day-ahead LP (``basic_constraints``)
— columns ``Quarter_Global, Date_Hour, Price, Charge_MW, Discharge_MW,
Energy_MWh`` — and exposes it to the RL environment as:

* ``dam_prices_by_day``  : ``{day: [price per 15-min slot]}`` — the reference
  price curve the synthetic XBID order book is anchored to;
* :class:`DAMScheduleProvider` : a scenario provider whose ``get_scenario(day)``
  returns the battery's committed DAM net position for that day, so the RL
  agent starts each episode already holding the real day-ahead schedule and
  only *re-optimises* it intraday.

Sign convention (matches :mod:`xbid_trader.market.battery`)::

    q_DAM_i = (Discharge_MW_i − Charge_MW_i) · dt        (MWh)
    +  → discharge / SELL commitment
    −  → charge    / BUY  commitment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SLOT_HOURS = 0.25


@dataclass
class DAMDayScenario:
    """Per-day scenario handed to :class:`~xbid_trader.market.xbid_env.XBIDEnv`."""

    imbalance_prices: np.ndarray        # value-of-energy reference (DAM price)
    battery_dam_schedule: np.ndarray    # committed net position q_DAM (MWh, signed)
    xbid_volume: Optional[np.ndarray] = None
    bm_up_price: Optional[np.ndarray] = None
    bm_down_price: Optional[np.ndarray] = None
    # Raw dispatch (for logging / diagnostics)
    charge_mw: Optional[np.ndarray] = None
    discharge_mw: Optional[np.ndarray] = None
    soe_mwh: Optional[np.ndarray] = None


class DAMScheduleProvider:
    """Serves :class:`DAMDayScenario` objects keyed by day string."""

    def __init__(self, by_day: Dict[str, DAMDayScenario]) -> None:
        self._by_day = by_day

    def get_scenario(self, day: str) -> DAMDayScenario:
        if day not in self._by_day:
            raise KeyError(day)
        return self._by_day[day]

    @property
    def days(self) -> List[str]:
        return sorted(self._by_day.keys())


def load_dam_dataset(
    path: str,
    sheet: str = "Results",
    months: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[float]], DAMScheduleProvider]:
    """Load the DAM workbook into ``(dam_prices_by_day, provider)``.

    Parameters
    ----------
    path:
        Path to the ``.xlsx`` file.
    sheet:
        Worksheet name (default ``"Results"``).
    months:
        Optional list of ``"YYYY-MM"`` strings to keep (e.g. ``["2026-04"]``
        for the April test, or ``["2026-01","2026-02","2026-03"]`` for the
        train split).  ``None`` keeps every day.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df["dt"] = pd.to_datetime(df["Date_Hour"], format="%d-%m-%Y %H:%M:%S")
    df["day"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["month"] = df["dt"].dt.strftime("%Y-%m")
    if months is not None:
        df = df[df["month"].isin(set(months))]
    df = df.sort_values("dt")

    dam_prices_by_day: Dict[str, List[float]] = {}
    by_day: Dict[str, DAMDayScenario] = {}

    for day, g in df.groupby("day"):
        g = g.sort_values("dt")
        price   = g["Price"].to_numpy(dtype=float)
        charge  = g["Charge_MW"].to_numpy(dtype=float)
        dischg  = g["Discharge_MW"].to_numpy(dtype=float)
        soe     = g["Energy_MWh"].to_numpy(dtype=float)
        # Net committed position per slot (MWh): + discharge / − charge
        q_dam = (dischg - charge) * SLOT_HOURS

        dam_prices_by_day[day] = price.tolist()
        by_day[day] = DAMDayScenario(
            imbalance_prices=price.copy(),
            battery_dam_schedule=q_dam,
            xbid_volume=np.zeros_like(price),
            charge_mw=charge,
            discharge_mw=dischg,
            soe_mwh=soe,
        )

    return dam_prices_by_day, DAMScheduleProvider(by_day)


def infer_battery_config(path: str, sheet: str = "Results") -> dict:
    """Infer battery parameters (power, capacity, bounds, efficiency, cycles)
    directly from the DAM workbook — handy for a sanity check / config dump."""
    df = pd.read_excel(path, sheet_name=sheet)
    df["dt"] = pd.to_datetime(df["Date_Hour"], format="%d-%m-%Y %H:%M:%S")
    df["day"] = df["dt"].dt.strftime("%Y-%m-%d")
    dt = SLOT_HOURS
    e, pch, pdh = (df.Energy_MWh.values, df.Charge_MW.values, df.Discharge_MW.values)
    nd, nc = [], []
    days = df["day"].values
    for h in range(1, len(e)):
        if days[h] != days[h - 1]:
            continue
        de = e[h] - e[h - 1]
        if pdh[h] > 0 and pch[h] == 0 and de < 0:
            nd.append(-pdh[h] * dt / de)
        if pch[h] > 0 and pdh[h] == 0 and de > 0:
            nc.append(de / (pch[h] * dt))
    emax = float(df.Energy_MWh.max())
    return {
        "power_mw": float(max(df.Charge_MW.max(), df.Discharge_MW.max())),
        "energy_mwh": emax,
        "soc_min_mwh": float(df.Energy_MWh.min()),
        "soc_max_mwh": emax,
        "eta_charge": round(float(np.median(nc)), 4) if nc else 0.95,
        "eta_discharge": round(float(np.median(nd)), 4) if nd else 0.95,
        "soc_initial_mwh": 50.0,
        "soc_target_mwh": 50.0,
        "max_cycles_per_day": 1.5,
    }
